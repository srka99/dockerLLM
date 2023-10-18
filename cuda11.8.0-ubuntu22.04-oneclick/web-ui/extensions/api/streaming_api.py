import asyncio
import functools
import json
from threading import Thread
import threading

from extensions.api.util import (
    build_parameters,
    try_start_cloudflared
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

# We use a thread local to store the asyncio lock, so that each thread
# has its own lock.  This isn't strictly necessary, but it makes it
# such that if we can support multiple worker threads in the future,
# thus handling multiple requests in parallel.
api_tls = threading.local()


def _get_api_lock(tls) -> asyncio.Lock:
    """
    The streaming and blocking API implementations each run on their own
    thread, and multiplex requests using asyncio.  If multiple outstanding
    requests are received at once, we will try to acquire the shared lock
    shared.generation_lock multiple times in succession in the same thread,
    which will cause a deadlock.

    To avoid this, we use this wrapper function to block on an asyncio
    lock, and then try and grab the shared lock only while holding
    the asyncio lock.
    """
    if not hasattr(tls, "asyncio_lock"):
        tls.asyncio_lock = asyncio.Lock()

    return tls.asyncio_lock


def with_api_lock(func):
    """
    This decorator should be added to all streaming API methods which
    require access to the shared.generation_lock.  It ensures that the
    tls.asyncio_lock is acquired before the method is called, and
    released afterwards.
    """
    @functools.wraps(func)
    async def api_wrapper(*args, **kwargs):
        async with _get_api_lock(api_tls):
            return await func(*args, **kwargs)
    return api_wrapper


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
    Thread(target=_run_server_custom, args=[5006], daemon=True).start()


app = FastAPI()


@app.post("/")
async def generate_stream(request: Request):
    async def generator(body: str):
        async with _get_api_lock(api_tls):
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

            generate_params = build_parameters(body)
            stopping_strings = generate_params.pop('stopping_strings')
            generate_params['stream'] = True

            generator = generate_reply(
                prompt, generate_params, stopping_strings=stopping_strings, is_chat=False)

            # As we stream, only send the new bytes.
            skip_index = 0
            message_num = 0

            for a in generator:
                if await request.is_disconnected():
                    return

                to_send = a[skip_index:]
                # partial unicode character, don't send it yet.
                if to_send is None or chr(0xfffd) in to_send:
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

                #await asyncio.sleep(0)
                skip_index += len(to_send)
                message_num += 1

    return EventSourceResponse(generator(await request.body()))


def _run_server_custom(port: int):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'

    print(f'Starting custom streaming server at http://{address}:{port}/')

    uvicorn.run(app, host=address, port=port)
