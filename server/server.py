import json
import time
import tornado
from pydantic import BaseModel
from typing import List

from inference.llama import Llama
from inference.model_config import ModelConfig
from inference.interpreter import Interpreter, Operation

class TokenizeInput(BaseModel):
    text: str

class TokenizeHandler(tornado.web.RequestHandler):
    def __init__(self, *args, llama: Llama=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llama = llama

    def post(self):
        input = TokenizeInput(**json.loads(self.request.body))
        tokens = self.llama.tokenize(input.text)
        self.write(json.dumps(tokens))


class CompletionInput(BaseModel):
    operations: List[Operation]

class StreamingCompletionHandler(tornado.web.RequestHandler):
    def __init__(self, *args, llama: Llama=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llama = llama

    async def post(self):
        input = CompletionInput(**json.loads(self.request.body))
        def write_callback(x):
            self.write(x)
            self.flush()
        interpreter = Interpreter(input.operations, write_callback)
        iter = 0
        while not interpreter.is_done():
            self.llama.run_loop(interpreter)
            iter += 1
            if iter > 4096:
                break

class TokenMapHandler(tornado.web.RequestHandler):
    def __init__(self, *args, llama: Llama=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llama = llama

    def get(self):
        self.write(json.dumps(self.llama.token_map()))

def make_app(llama: Llama):
    return tornado.web.Application([
        (r"/tokenize", TokenizeHandler, {"llama": llama}),
        (r"/streaming_completion", StreamingCompletionHandler, {"llama": llama}),
        (r"/token_map", TokenMapHandler, {"llama": llama}),
        (r"/(.*)", tornado.web.StaticFileHandler, {"path": "client", "default_filename": "index.html"}),
    ])

def server_main():
    config = ModelConfig(
        name="Phi3-mini-1.0",
        model_filename="./models/Phi-3-mini-4k-instruct-q4.gguf",
        context_size=4096,
        temperature=1.0,
        batch_size=512,
        batch_max_tokens=2048,
    )
    llama = Llama(config)
    app = make_app(llama)
    app.listen(port=8888, address="0.0.0.0")
    tornado.ioloop.IOLoop.current().start()