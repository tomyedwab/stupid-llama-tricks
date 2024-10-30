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

class FeedText(Action):
    def __init__(self, text: bytes, calculate_likelihood: bool = False):
        super().__init__()
        self.type = "feed_text"
        self.text = text
        self.calculate_likelihood = calculate_likelihood
        self.likelihood = None

class FeedTokens(Action):
    def __init__(self, tokens: List[int], calculate_likelihood: bool = False):
        super().__init__()
        self.type = "feed_tokens"
        self.tokens = tokens
        self.calculate_likelihood = calculate_likelihood
        self.likelihood = None

class Completion(Action):
    def __init__(self, max_tokens: int):
        super().__init__()
        self.type = "completion"
        self.remaining_tokens = max_tokens
        self.response_text = ""
