import asyncio
import contextvars
import random
import regex
import time
from typing import Any, Callable, Awaitable

import llama_cpp

from .beam import LlamaBeam
from .actions import MatchPattern, FeedText, Done, Wait, Completion

beam_idx = contextvars.ContextVar('Beam index of the active state function')

# TODO: Support splitting beams

class LlamaRequest(object):
    def __init__(
            self,
            state_function: Callable[["LlamaRequest"], Awaitable[Any]],
        ):
        # The ID is the current timestamp and a random number
        self.id = f"{time.time()}-{random.randint(0, 1000000)}"
        self.beams = [LlamaBeam()]
        self.tokens = None
        self.num_tokens = None
        self.seq_num = None
        self.completed = False
        self.state_function = state_function
        self.state_result = None
        loop = asyncio.get_event_loop()
        loop.create_task(self.run_state_function(0))

    async def run_state_function(self, current_beam_idx: int):
        beam_idx.set(current_beam_idx)
        self.state_result = await self.state_function(self)
        self.beams[current_beam_idx].set_action(Done())

    async def wait_for_action_complete(self, current_beam_idx: int):
        while not isinstance(self.beams[current_beam_idx].current_action, Wait):
            await asyncio.sleep(0)

    async def match_pattern(self, pattern: regex.Pattern):
        current_beam_idx = beam_idx.get()
        action = MatchPattern(pattern)
        self.beams[current_beam_idx].set_action(action)
        await self.wait_for_action_complete(current_beam_idx)
        return {
            "text": action.current_match,
            "logit": action.current_match_logit,
        }

    async def feed_text(self, text: str):
        current_beam_idx = beam_idx.get()
        self.beams[current_beam_idx].set_action(FeedText(text))
        await self.wait_for_action_complete(current_beam_idx)

    async def completion(self, max_tokens: int):
        current_beam_idx = beam_idx.get()
        action = Completion(max_tokens)
        self.beams[current_beam_idx].set_action(action)
        await self.wait_for_action_complete(current_beam_idx)
        return action.response_text
