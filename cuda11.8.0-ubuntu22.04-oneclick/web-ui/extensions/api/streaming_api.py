import asyncio
import json
from threading import Thread

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


@app.post("/")
@with_api_lock
async def _handle_stream_message(request: Request):
    def generator(body: str):
        body = json.loads(body)

        prompt = body['inputs']

        body = body["parameters"]
        if 'stop' in body:
            body["stopping_strings"] = body.pop("stop")
        if 'truncate' in body:
            body["truncation_length"] = body.pop("truncate")

        generate_params = build_parameters(body)
        stopping_strings = generate_params.pop('stopping_strings')
        generate_params['stream'] = True

        generator = generate_reply(
            prompt, generate_params, stopping_strings=stopping_strings, is_chat=False)

        # As we stream, only send the new bytes.
        skip_index = 0
        message_num = 0

        for a in generator:
            to_send = a[skip_index:]
            if to_send is None or chr(0xfffd) in to_send:  # partial unicode character, don't send it yet.
                continue

            yield json.dumps({
                "token": {
                    "id": 0,
                    "text": to_send,
                    "logprob": 0,
                    "special": False,
                },
                "generated_text": None,
                "details": None
            })

            skip_index += len(to_send)
            message_num += 1

    return EventSourceResponse(generator(await request.body()))


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


def _run_server(port: int, share: bool = False, tunnel_id=str):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'

    print(f'Starting streaming server at ws://{address}:{port}{PATH}')

    #asyncio.run(_run(host=address, port=port))
    uvicorn.run(app, host=address, port=port)


def start_server(port: int, share: bool = False, tunnel_id=str):
    Thread(target=_run_server, args=[port, share, tunnel_id], daemon=True).start()
