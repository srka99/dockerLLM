import asyncio
import json
from threading import Thread

import websockets

from extensions.api.util import (
    build_parameters,
    try_start_cloudflared,
    with_api_lock
)
from modules import shared
from modules.chat import generate_chat_reply
from modules.text_generation import generate_reply
from websockets.server import serve

from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
import uvicorn

app = FastAPI()

PATH = '/api/v1/stream'


@with_api_lock
async def _handle_stream_message(websocket, message):
    message = json.loads(message)

    prompt = message['prompt']
    generate_params = build_parameters(message)
    stopping_strings = generate_params.pop('stopping_strings')
    generate_params['stream'] = True

    generator = generate_reply(
        prompt, generate_params, stopping_strings=stopping_strings, is_chat=False)

    # As we stream, only send the new bytes.
    skip_index = 0
    message_num = 0

    for a in generator:
        to_send = a[skip_index:]
        # partial unicode character, don't send it yet.
        if to_send is None or chr(0xfffd) in to_send:
            continue

        await websocket.send(json.dumps({
            'event': 'text_stream',
            'message_num': message_num,
            'text': to_send
        }))

        await asyncio.sleep(0)
        skip_index += len(to_send)
        message_num += 1

    await websocket.send(json.dumps({
        'event': 'stream_end',
        'message_num': message_num
    }))


@with_api_lock
async def _handle_chat_stream_message(websocket, message):
    body = json.loads(message)

    user_input = body['user_input']
    generate_params = build_parameters(body, chat=True)
    generate_params['stream'] = True
    regenerate = body.get('regenerate', False)
    _continue = body.get('_continue', False)

    generator = generate_chat_reply(
        user_input, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)

    message_num = 0
    for a in generator:
        await websocket.send(json.dumps({
            'event': 'text_stream',
            'message_num': message_num,
            'history': a
        }))

        await asyncio.sleep(0)
        message_num += 1

    await websocket.send(json.dumps({
        'event': 'stream_end',
        'message_num': message_num
    }))


async def _handle_connection(websocket, path):

    if path == '/api/v1/stream':
        async for message in websocket:
            await _handle_stream_message(websocket, message)

    elif path == '/api/v1/chat-stream':
        async for message in websocket:
            await _handle_chat_stream_message(websocket, message)

    else:
        print(f'Streaming api: unknown path: {path}')
        return


async def _run(host: str, port: int):
    async with serve(_handle_connection, host, port, ping_interval=None):
        await asyncio.Future()  # run forever


def _run_server(port: int, share: bool = False, tunnel_id=str):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'

    def on_start(public_url: str):
        public_url = public_url.replace('https://', 'wss://')
        print(f'Starting streaming server at public url {public_url}{PATH}')

    if share:
        try:
            try_start_cloudflared(
                port, tunnel_id, max_attempts=3, on_start=on_start)
        except Exception as e:
            print(e)
    else:
        print(f'Starting streaming server at ws://{address}:{port}{PATH}')

    asyncio.run(_run(host=address, port=port))


def start_server(port: int, share: bool = False, tunnel_id=str):
    Thread(target=_run_server, args=[
           port, share, tunnel_id], daemon=True).start()
    Thread(target=_run_server_custom, args=[
           5006, share, tunnel_id], daemon=True).start()


app = FastAPI()


@app.post("/")
async def generate_stream(request: Request):
    async def generator(body: str):
        body = json.loads(body)

        prompt = body['inputs']

        body = body["parameters"]
        body["prompt"] = prompt
        if 'stop' in body:
            body["stopping_strings"] = body.pop("stop")
        if 'truncate' in body:
            body["truncation_length"] = body.pop("truncate")
        if 'typical_p' in body and body['typical_p'] is None:
            body.pop("typical_p")
        if 'typical' in body and body['typical'] is None:
            body.pop("typical")

        async with websockets.connect("ws://localhost:5005/api/v1/stream", ping_interval=None) as websocket:
            await websocket.send(json.dumps(body))

            while True:
                incoming_data = await websocket.recv()
                incoming_data = json.loads(incoming_data)

                match incoming_data['event']:
                    case 'text_stream':
                        text = incoming_data['text']
                        yield json.dumps({
                            "token": {
                                "id": 0,
                                "text": text,
                                "logprob": 0,
                                "special": False,
                            },
                            "generated_text": None,
                            "details": None
                        })
                    case 'stream_end':
                        return

    return EventSourceResponse(generator(await request.body()))


def _run_server_custom(port: int, share: bool = False, tunnel_id=str):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'

    print(f'Starting custom streaming server at http://{address}:{port}/')

    uvicorn.run(app, host=address, port=port)
