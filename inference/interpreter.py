import asyncio
from pydantic import BaseModel
from typing import List, Optional, Callable, Awaitable, Any

from inference.request import LlamaRequest, ForkArguments

class FeedTokensOperation(BaseModel):
    tokens: List[int]
    top_p: int

class CompletionOperation(BaseModel):
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

def validate(operations: List[Operation]):
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
                validate(fork)

def run(operations: List[Operation]) -> Callable[[LlamaRequest], Awaitable[List[Operation]]]:
    async def _run_fork(request: LlamaRequest, fork_operations: List[Operation]):
        return await run(fork_operations)(request)
    
    async def _run(request: LlamaRequest):
        for operation in operations:
            if operation.name == "feed_tokens":
                operation.result = await request.feed_tokens(operation.feed_tokens.tokens, operation.feed_tokens.top_p)
            elif operation.name == "completion":
                operation.result = await request.completion(operation.completion.max_tokens, operation.completion.top_p)
            elif operation.name == "branch":
                operation.result = await request.fork(_run_fork, [
                    ForkArguments([fork_operations], {})
                    for fork_operations in operation.branch.forks
                ])
        return operations

    return _run
