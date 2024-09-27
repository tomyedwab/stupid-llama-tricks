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
            initial_response: Optional[str] = ""
        ):
        self.seq_num = None
        self.pos = 0
        self.current_action = Wait()
        self.response = initial_response

    def is_done(self) -> bool:
        return isinstance(self.current_action, Done)

    def set_action(self, action: Action):
        self.current_action = action

    def wait_for_action(self):
        if isinstance(self.current_action, Wait):
            logging.info(f"Waiting for next action")
            while isinstance(self.current_action, Wait):
                # Wait for the next async event
                time.sleep(0.1)
            logging.info(f"Got action {type(self.current_action)}")

    def decode_next(
            self,
            ctx: llama_cpp.llama_context_p,
            model: llama_cpp.llama_model_p,
            vocab_size: int,
            temperature: float,
            logits: List[float],
            candidates_p: llama_cpp.llama_token_data_array,
        ):
        if self.is_done():
            return None

        self.wait_for_action()

        if isinstance(self.current_action, MatchPattern):
            # Choose the top token that allows us to match the target
            action = self.current_action
            output = [(logits[i], i) for i in range(vocab_size)]
            output.sort(reverse=True)
            for logit, token_id in output:
                match_candidate = action.current_match + token_to_string(model, token_id)
                match = self.current_action.pattern.match(match_candidate, partial=True)
                if match is not None:
                    action.current_match = match_candidate
                    selected_token = token_id
                    action.current_match_logit += logit
                    if not match.partial:
                        self.current_action = Wait()
                    break

        elif isinstance(self.current_action, Completion):
            # Do normal top-p sampling
            for token_id in range(vocab_size):
                candidates_p.contents.data[token_id].id = llama_cpp.llama_token(token_id)
                candidates_p.contents.data[token_id].logit = llama_cpp.ctypes.c_float(logits[token_id])
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
                self.current_action = Wait()
                return None

            elif "<|end|>" in action.response_text:
                action.response_text = action.response_text.replace("<|end|>", "")
                self.current_action = Wait()
                return None

            elif action.remaining_tokens < 1:
                self.current_action = Wait()

        if selected_token is None:
            self.current_action = Wait()
            return None

        return selected_token

    def decode_tokens(
            self,
            ctx: llama_cpp.llama_context_p,
            model: llama_cpp.llama_model_p,
            batch: llama_cpp.llama_batch,
            batch_size: int,
        ) -> ctypes.Array[ctypes.c_float]:
        if self.is_done():
            return None
        
        self.wait_for_action()

        if not isinstance(self.current_action, FeedText):
            return None

        encoded = self.current_action.text.encode('utf-8')
        tokens = (llama_cpp.llama_token * int(4096))()
        n_tokens = llama_cpp.llama_tokenize(model, encoded, len(encoded), tokens, 4096, True, False)
        if n_tokens == 0:
            self.current_action = Wait()
            return None

        for start in range(0, n_tokens, batch_size):
            end = min(start + batch_size, n_tokens)
            for i in range(start, end):
                batch.token[i-start] = tokens[i]
                batch.pos[i-start] = self.pos
                batch.n_seq_id[i-start] = 1
                batch.seq_id[i-start][0] = self.seq_num
                batch.logits[i-start] = i == (n_tokens - 1)
                self.pos += 1
            batch.n_tokens = end - start

            ret = llama_cpp.llama_decode(ctx, batch)
            if ret != 0:
                raise Exception(f"Llama error {ret} with batch size {end - start} after tokenizing {start} tokens")

        logits = llama_cpp.llama_get_logits_ith(ctx, end - start - 1)
        self.response += self.current_action.text
        self.current_action = Wait()

        return logits