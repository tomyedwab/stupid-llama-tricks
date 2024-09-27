import asyncio
import logging
import regex

from inference.model_config import ModelConfig
from inference.llama import Llama
from inference.request import LlamaRequest

# Log to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_regex(request: LlamaRequest):
    await request.feed_text("<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nHello, how are you?<|end|>\n<|assistant|>\n")

    pattern = regex.compile(r"I feel (good|bad)")
    matched = await request.match_pattern(pattern)
    logging.info(f"Matched: {matched}")

    sentiment = matched["text"].split(" ")[-1]
    await request.feed_text(f".\n<|user|>\nWhy do you feel {sentiment}?<|end|>\n<|assistant|>\n")

    justification = await request.completion(100)

    return f"The AI feels {sentiment} because: {justification}"

async def main():
    config = ModelConfig(
        name="Phi3-mini-1.0",
        model_filename="./models/Phi-3-mini-4k-instruct-q4.gguf",
        context_size=4096,
        temperature=1.0,
        batch_size=512,
        batch_max_tokens=2048,
    )
    llama = Llama(config)
    request = llama.queue_request(LlamaRequest(test_regex))
    llama.run_in_background()
    response = await llama.await_response(request)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
