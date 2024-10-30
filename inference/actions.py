import asyncio
import regex
from typing import List

class Action(object):
    def __init__(self):
        self.type = None
        self.wait_condition = asyncio.Condition()

    async def wait(self):
        async with self.wait_condition:
            await self.wait_condition.wait()

class Wait(Action):
    def __init__(self):
        super().__init__()
        self.type = "wait"

class Done(Action):
    def __init__(self):
        super().__init__()
        self.type = "done"

class MatchPattern(Action):
    def __init__(self, pattern: regex.Pattern):
        super().__init__()
        self.type = "match_pattern"
        self.pattern = pattern
        self.current_match_logit = 0
        self.current_match = ""

class FeedTokens(Action):
    def __init__(self, tokens: List[int], top_p: int = 0):
        super().__init__()
        self.type = "feed_tokens"
        self.tokens = tokens
        self.logits = None
        self.top_p = top_p
        self.token_map = {}

class Completion(Action):
    def __init__(self, max_tokens: int, top_p: int = 0):
        super().__init__()
        self.type = "completion"
        self.remaining_tokens = max_tokens
        self.response_text = ""
        self.logits = []
        self.top_p = top_p
        self.token_map = {}
