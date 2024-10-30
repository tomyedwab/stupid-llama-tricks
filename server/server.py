import asyncio
import json
import tornado
from pydantic import BaseModel
from typing import List

from inference.llama import Llama, LlamaRequest
from inference.model_config import ModelConfig
from inference import interpreter

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
    operations: List[interpreter.Operation]

class StreamingCompletionHandler(tornado.web.RequestHandler):
    def __init__(self, *args, llama: Llama=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llama = llama

    async def post(self):
        input = CompletionInput(**json.loads(self.request.body))
        interpreter.validate(input.operations)
        result: List[interpreter.Operation] = await self.llama.do_request(LlamaRequest(interpreter.run(input.operations)))
        self.write(json.dumps([op.model_dump() for op in result]))

def make_app(llama: Llama):
    return tornado.web.Application([
        (r"/tokenize", TokenizeHandler, {"llama": llama}),
        (r"/streaming_completion", StreamingCompletionHandler, {"llama": llama}),
    ])

async def server_main():
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
    app.listen(8888)
    await asyncio.gather(*[
        llama.run(),
        asyncio.Event().wait(),
    ])