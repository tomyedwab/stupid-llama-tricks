import regex

class Action(object):
    def __init__(self):
        self.type = None

class Wait(Action):
    def __init__(self):
        self.type = "wait"

class Done(Action):
    def __init__(self):
        self.type = "done"

class MatchPattern(Action):
    def __init__(self, pattern: regex.Pattern):
        self.type = "match_pattern"
        self.pattern = pattern
        self.current_match_logit = 0
        self.current_match = ""

class FeedText(Action):
    def __init__(self, text: bytes, calculate_likelihood: bool = False):
        self.type = "feed_text"
        self.text = text
        self.calculate_likelihood = calculate_likelihood
        self.likelihood = None

class Completion(Action):
    def __init__(self, max_tokens: int):
        self.remaining_tokens = max_tokens
        self.response_text = ""
