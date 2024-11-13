import json
import uuid
from pydantic import BaseModel
from typing import List, Optional, Any, Tuple, Callable

class FeedTokensOperation(BaseModel):
    role: str
    tokens: List[int]
    top_p: int

class CompletionOperation(BaseModel):
    role: str
    max_tokens: int
    top_p: int

class BranchOperation(BaseModel):
    forks: List[List["Operation"]]

class Operation(BaseModel):
    id: str
    name: str
    result: Optional[Any] = None
    feed_tokens: Optional[FeedTokensOperation] = None
    completion: Optional[CompletionOperation] = None
    branch: Optional[BranchOperation] = None

    def get_role(self):
        if self.name == "feed_tokens":
            return self.feed_tokens.role
        elif self.name == "completion":
            return self.completion.role
        return None

    def get_top_p(self):
        if self.name == "completion":
            return self.completion.top_p
        elif self.name == "feed_tokens":
            return self.feed_tokens.top_p
        return 1

class OperationContext(object):
    def __init__(self, id: str, operation: Operation, reporting_callback: Callable[[str], None], parent_id: Optional[str] = None):
        self.id = id
        self.operation = operation
        self.parent_id = parent_id
        self.token_index = 0
        self.completed = False
        self.reporting_callback = reporting_callback

    def loop(self):
        if self.completed:
            return []
        return [self]

    def report_token(self, token: int, logits: List[Tuple[int, float]]):
        self.reporting_callback((json.dumps([
            self.operation.id,
            self.token_index,
            token,
            logits,
        ]) + "\n").encode("utf-8"))
        self.token_index += 1

    def is_completed(self) -> Optional[str]:
        if self.completed:
            return self.id
        return None

class SequenceContext(object):
    def __init__(self, id: str, operations: List[Operation], reporting_callback: Callable[[str], None], parent_id: Optional[str] = None):
        self.id = id
        self.completed = False
        self.operations = operations
        self.operation_index = 0
        self.reporting_callback = reporting_callback
        self.current_context = OperationContext(str(uuid.uuid4()), operations[0], reporting_callback, parent_id)

    def loop(self):
        if self.completed:
            return []
        next_parent = self.current_context.is_completed()
        if next_parent is not None:
            if self.operation_index < len(self.operations)-1:
                next_operation = self.operations[self.operation_index+1]
                self.operation_index += 1
                if next_operation.name == "branch":
                    self.current_context = BranchContext(str(uuid.uuid4()), next_operation.branch.forks, self.reporting_callback, next_parent)
                else:
                    self.current_context = OperationContext(str(uuid.uuid4()), next_operation, self.reporting_callback, next_parent)
            else:
                self.completed = True
                return []
        return self.current_context.loop()

    def is_completed(self) -> Optional[str]:
        if self.completed:
            return self.current_context.id
        return None

class BranchContext(object):
    def __init__(self, id: str, forks: List[List[Operation]], reporting_callback: Callable[[str], None], parent_id: str):
        self.id = id
        self.forks = [
            SequenceContext(str(uuid.uuid4()), fork, reporting_callback, parent_id)
            for fork in forks
        ]
        self.completed = False

    def loop(self):
        running_contexts = []
        all_done = True
        for fork in self.forks:
            running_contexts.extend(fork.loop())
            if fork.is_completed() is None:
                all_done = False
        if all_done:
            self.completed = True
        return running_contexts

    def is_completed(self) -> Optional[str]:
        # TODO: Return the highest probability fork
        if self.completed:
            return self.forks[0].is_completed()
        return None

class Interpreter(object):
    def __init__(self, operations: List[Operation], reporting_callback: Callable[[str], None]):
        self._validate(operations)
        self.root = SequenceContext("root", operations, reporting_callback)

    def _validate(self, operations: List[Operation]):
        for operation in operations:
            if operation.name == "feed_tokens":
                if operation.feed_tokens is None:
                    raise ValueError("feed_tokens cannot be None")
                if len(operation.feed_tokens.tokens) == 0:
                    raise ValueError("feed_tokens cannot be empty")
            elif operation.name == "completion":
                if operation.completion is None:
                    raise ValueError("completion cannot be None")
            elif operation.name == "branch":
                if operation.branch is None:
                    raise ValueError("branch cannot be None")
                for fork in operation.branch.forks:
                    self._validate(fork)

    def loop(self):
        return self.root.loop()
    
    def is_done(self):
        return self.root.is_completed() is not None
