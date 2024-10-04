import asyncio
import contextvars
from dataclasses import dataclass
import logging
import random
import regex
import time
from typing import Any, Callable, Awaitable, List, Dict

from .beam import LlamaBeam
from .actions import MatchPattern, FeedText, Done, Wait, Completion

beam_idx = contextvars.ContextVar('Beam index of the active state function')

@dataclass
class ForkArguments(object):
    args: List[Any]
    kwargs: Dict[str, Any]

class LlamaRequest(object):
    def __init__(
            self,
            state_function: Callable[["LlamaRequest"], Awaitable[Any]],
        ):
        # The ID is the current timestamp and a random number
        self.id = f"{time.time()}-{random.randint(0, 1000000)}"
        self.beams = [LlamaBeam(0)]
        self.tokens = None
        self.num_tokens = None
        self.seq_num = None
        self.completed = False
        self.state_function = state_function
        loop = asyncio.get_event_loop()
        self.task = loop.create_task(self.run_state_function(0, self.state_function, ForkArguments([], {})))

    async def get_result(self):
        return await self.task

    async def run_state_function(self, current_beam_idx: int, state_function: Callable[["LlamaRequest"], Awaitable[Any]], args: ForkArguments):
        beam_idx.set(current_beam_idx)
        result = await state_function(self, *args.args, **args.kwargs)
        self.beams[current_beam_idx].set_action(Done())
        return result

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

    async def feed_text(self, text: str, calculate_likelihood: bool = False):
        current_beam_idx = beam_idx.get()
        action = FeedText(text, calculate_likelihood)
        self.beams[current_beam_idx].set_action(action)
        await self.wait_for_action_complete(current_beam_idx)
        return action.likelihood

    async def completion(self, max_tokens: int):
        current_beam_idx = beam_idx.get()
        action = Completion(max_tokens)
        self.beams[current_beam_idx].set_action(action)
        await self.wait_for_action_complete(current_beam_idx)
        return action.response_text

    async def fork(self, new_action_function: Callable[["LlamaRequest"], Awaitable[Any]], args: List[ForkArguments]):
        current_beam_idx = beam_idx.get()
        parent_beam = self.beams[current_beam_idx]
        new_beam_indices = [current_beam_idx]
        for _ in range(len(args)-1):
            new_beam_indices.append(len(self.beams))
            new_beam = LlamaBeam(len(self.beams), parent=parent_beam)
            self.beams.append(new_beam)
        logging.info(f"Forked beams: {len(new_beam_indices)}")
        forked_tasks = []
        for idx, new_beam_idx in enumerate(new_beam_indices):
            forked_tasks.append(self.run_state_function(new_beam_idx, new_action_function, args[idx]))
        results = await asyncio.gather(*forked_tasks)
        self.beams[current_beam_idx].set_action(Wait())
        return results
