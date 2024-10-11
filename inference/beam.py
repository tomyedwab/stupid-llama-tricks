import asyncio
import ctypes
import logging
import time
from typing import Any, Optional, List

import llama_cpp

from .actions import Action, Wait, Done, MatchPattern, FeedText, Completion
from .util import token_to_string

class LlamaBeam(object):
    def __init__(
            self,
            index: int,
            parent: "LlamaBeam" = None,
        ):
        self.index = index
        self.parent = parent
        if parent is not None:
            self.response = parent.response
            self.pos = parent.pos
        else:
            self.pos = 0
            self.response = ""
        self.seq_num = -1
        self.current_action = Wait()
        self.logits = None

    def is_done(self) -> bool:
        return isinstance(self.current_action, Done)

    def is_runnable(self) -> bool:
        return not isinstance(self.current_action, Wait) and not self.is_done()

    async def set_action(self, action: Action):
        previous_action = self.current_action
        self.current_action = action
        async with previous_action.wait_condition:
            previous_action.wait_condition.notify_all()

    async def wait_for_next_action(self) -> bool:
        current_action = self.current_action
        if not isinstance(current_action, Wait):
            return True
        try:
            async with current_action.wait_condition:
                await asyncio.wait_for(current_action.wait_condition.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            return False
        return True

    async def decode_next(
            self,
            ctx: llama_cpp.llama_context_p,
            model: llama_cpp.llama_model_p,
            vocab_size: int,
            temperature: float,
            candidates_p: llama_cpp.llama_token_data_array,
        ) -> Optional[llama_cpp.llama_token]:
        if self.is_done():
            return None

        if isinstance(self.current_action, MatchPattern):
            # Choose the top token that allows us to match the target
            action = self.current_action
            output = [(self.logits[i], i) for i in range(vocab_size)]
            output.sort(reverse=True)
            selected_token = None
            for logit, token_id in output:
                match_candidate = action.current_match + token_to_string(model, token_id)
                if match_candidate.strip() == action.current_match.strip():
                    continue
                match = self.current_action.pattern.match(match_candidate, partial=True)
                if match is not None:
                    action.current_match = match_candidate
                    selected_token = token_id
                    action.current_match_logit += logit
                    if not match.partial:
                        await self.set_action(Wait())
                    break
            if selected_token is None:
                # None of the available tokens allow us to match the target
                await self.set_action(Wait())

        elif isinstance(self.current_action, Completion):
            # Do normal top-p sampling
            for token_id in range(vocab_size):
                candidates_p.contents.data[token_id].id = llama_cpp.llama_token(token_id)
                candidates_p.contents.data[token_id].logit = llama_cpp.ctypes.c_float(self.logits[token_id])
            candidates_p.contents.size = vocab_size
            llama_cpp.llama_sample_temp(ctx, candidates_p, temperature)
            llama_cpp.llama_sample_top_p(ctx, candidates_p, 0.9, 1)
            selected_token = llama_cpp.llama_sample_token(ctx, candidates_p)

        else:
            return None
    
        token_string = token_to_string(model, selected_token)
        self.response += token_string

        if isinstance(self.current_action, Completion):
            action = self.current_action
            action.response_text += token_string
            action.remaining_tokens -= 1

            if llama_cpp.llama_token_is_eog(model, selected_token):
                await self.set_action(Wait())
                return None

            elif "<|end|>" in action.response_text:
                action.response_text = action.response_text.replace("<|end|>", "")
                await self.set_action(Wait())
                return None

            elif action.remaining_tokens < 1:
                await self.set_action(Wait())

        if selected_token is None:
            await self.set_action(Wait())
            return None

        return selected_token

    async def decode_tokens(
            self,
            ctx: llama_cpp.llama_context_p,
            model: llama_cpp.llama_model_p,
            batch: llama_cpp.llama_batch,
            batch_size: int,
        ) -> ctypes.Array[ctypes.c_float]:
        if self.is_done():
            return False
        
        if not isinstance(self.current_action, FeedText):
            return True

        encoded = self.current_action.text.encode('utf-8')
        tokens = (llama_cpp.llama_token * int(4096))()
        n_tokens = llama_cpp.llama_tokenize(model, encoded, len(encoded), tokens, 4096, True, False)
        if n_tokens == 0:
            await self.set_action(Wait())
            return True

        self.current_action.likelihood = 0.0

        for start in range(0, n_tokens, batch_size):
            end = min(start + batch_size, n_tokens)
            for i in range(start, end):
                batch.token[i-start] = tokens[i]
                batch.pos[i-start] = self.pos
                batch.n_seq_id[i-start] = 1
                batch.seq_id[i-start][0] = self.seq_num
                batch.logits[i-start] = self.current_action.calculate_likelihood or (i == (n_tokens - 1))
                self.pos += 1
            batch.n_tokens = end - start

            ret = llama_cpp.llama_decode(ctx, batch)
            if ret != 0:
                #raise Exception(f"Llama error {ret} with batch size {end - start} after tokenizing {start} tokens")
                print(f"Llama error {ret} with batch size {end - start} after tokenizing {start} tokens")
                return False

            if self.current_action.calculate_likelihood:
                for idx in range(end - start - 1):
                    logits = llama_cpp.llama_get_logits_ith(ctx, idx)
                    self.current_action.likelihood += logits[tokens[idx + 1]]

        self.logits = llama_cpp.llama_get_logits_ith(ctx, end - start - 1)
        self.response += self.current_action.text
        await self.set_action(Wait())

        return True