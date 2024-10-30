import asyncio
import logging
import regex

from inference.model_config import ModelConfig
from inference.llama import Llama
from inference.request import LlamaRequest, ForkArguments

# Log to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_fork(request: LlamaRequest):
    print("Starting test_fork task")
    await request.feed_text(
        "<|system|>\nYou are a helpful assistant.<|end|>\n" +
        "<|user|>\nJoey said something very hurtful to Sally. How does Sally feel?<|end|>\n" +
        "<|assistant|>\n"
    )
    logging.info("Done with prompt")

    async def get_text_likelihood(request: LlamaRequest, text: str):
        likelihood = await request.feed_text(text, calculate_likelihood=True)
        logging.info(f"Likelihood for {text}: {likelihood}")
        return {"text": text, "likelihood": likelihood}

    all_matches = await request.fork(get_text_likelihood, [
        ForkArguments(["She feels good."], {}),
        ForkArguments(["She feels bad."], {}),
    ])

    # Sort matches by logit & take the most likely one
    all_matches.sort(key=lambda x: x["likelihood"], reverse=True)
    sentiment = all_matches[0]["text"].split(" ")[-1][:-1]

    await request.feed_text(f"\n<|user|>\nWhy does she feel {sentiment}?<|end|>\n<|assistant|>\n")

    justification = await request.completion(100)

    return f"The AI predicts that Sally feels {sentiment}. The AI's reasoning is: {justification}"

async def run_requests(llama: Llama):
    result = await asyncio.gather(*[
        llama.do_request(LlamaRequest(test_fork))
        for _ in range(100)
    ])
    print(result)
    llama.request_stop()

async def fork_main():
    config = ModelConfig(
        name="Phi3-mini-1.0",
        model_filename="./models/Phi-3-mini-4k-instruct-q4.gguf",
        context_size=4096,
        temperature=1.0,
        batch_size=512,
        batch_max_tokens=2048,
    )
    llama = Llama(config)
    await asyncio.gather(*[
        llama.run(),
        run_requests(llama),
    ])
