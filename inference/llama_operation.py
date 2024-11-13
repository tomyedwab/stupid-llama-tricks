import ctypes
from typing import Any, Optional, List, Dict

import llama_cpp

from .interpreter import OperationContext

class LlamaOperation(object):
    def __init__(
            self,
            context: OperationContext,
            parent: "LlamaOperation" = None,
        ):
        self.context = context
        self.parent = parent
        if parent is not None:
            self.tokens = parent.tokens
            self.current_role = parent.current_role
        else:
            self.tokens = []
            self.current_role = None
        self.seq_num = -1
        self.logits = None
        if context.operation.name == "completion":
            self.remaining_tokens = context.operation.completion.max_tokens
        else:
            self.remaining_tokens = 0
        self.is_done = False

    def decode_next(
            self,
            ctx: llama_cpp.llama_context_p,
            model: llama_cpp.llama_model_p,
            vocab_size: int,
            temperature: float,
            candidates_p: llama_cpp.llama_token_data_array,
        ) -> Optional[llama_cpp.llama_token]:
        if self.is_done:
            return None

        if self.context.operation.name == "completion":
            # Do normal top-p sampling
            for token_id in range(vocab_size):
                candidates_p.contents.data[token_id].id = llama_cpp.llama_token(token_id)
                candidates_p.contents.data[token_id].logit = llama_cpp.ctypes.c_float(self.logits[token_id])
            candidates_p.contents.size = vocab_size
            candidates_p.contents.sorted = False
            llama_cpp.llama_sample_temp(ctx, candidates_p, temperature)
            llama_cpp.llama_sample_top_p(ctx, candidates_p, 0.9, self.context.operation.get_top_p())
            selected_token = llama_cpp.llama_sample_token(ctx, candidates_p)
            self.context.report_token(selected_token, [
                (tkn.id, tkn.logit)
                for tkn in candidates_p.contents.data[:self.context.operation.get_top_p()]
            ])

            self.remaining_tokens -= 1

            if llama_cpp.llama_token_is_eog(model, selected_token):
                print(f"Operation {self.context.id} completed on EOG token")
                self.is_done = True
                self.context.completed = True

            elif self.remaining_tokens < 1:
                print(f"Operation {self.context.id} completed on max tokens")
                self.is_done = True
                self.context.completed = True

            return selected_token

        else:
            return None


    def get_tokens_for_role_switch(self, tokens_for_role: Dict[str, List[int]], role: str) -> List[int]:
        if role == self.current_role:
            return []
        if self.current_role is None:
            return tokens_for_role["null_" + role]
        return tokens_for_role["end_" + role]

    def restore_tokens(
            self,
            ctx: llama_cpp.llama_context_p,
            batch: llama_cpp.llama_batch,
            batch_size: int,
            new_seq_num: int,
        ):
        for start in range(0, len(self.tokens), batch_size):
            end = min(start + batch_size, len(self.tokens))
            for i in range(start, end):
                batch.token[i-start] = self.tokens[i]
                batch.pos[i-start] = i
                batch.n_seq_id[i-start] = 1
                batch.seq_id[i-start][0] = new_seq_num
                batch.logits[i-start] = (i == end - 1)
            batch.n_tokens = end - start

            ret = llama_cpp.llama_decode(ctx, batch)
            if ret != 0:
                print(f"Llama error {ret} with batch size {end - start} after tokenizing {start} tokens")
                return False

        self.logits = llama_cpp.llama_get_logits_ith(ctx, end - start - 1)

        return True

    def decode_tokens(
            self,
            ctx: llama_cpp.llama_context_p,
            model: llama_cpp.llama_model_p,
            vocab_size: int,
            candidates_p: llama_cpp.llama_token_data_array,
            batch: llama_cpp.llama_batch,
            batch_size: int,
            tokens_for_role: Dict[str, List[int]],
        ) -> ctypes.Array[ctypes.c_float]:
        if self.is_done:
            return False
        
        tokens = []
        desired_role = self.context.operation.get_role()
        if desired_role != self.current_role:
            tokens = self.get_tokens_for_role_switch(tokens_for_role, desired_role)
            print(f"Switching role to {desired_role} with tokens {tokens}")
        if self.context.operation.name == "feed_tokens":
            tokens = tokens + self.context.operation.feed_tokens.tokens
            print(f"Feeding {len(self.context.operation.feed_tokens.tokens)} tokens")
        if len(tokens) == 0:
            return True

        self.context.report_token(tokens[0], [(tokens[0], 0.0)])

        pos = len(self.tokens)

        for start in range(0, len(tokens), batch_size):
            end = min(start + batch_size, len(tokens))
            for i in range(start, end):
                batch.token[i-start] = tokens[i]
                batch.pos[i-start] = pos + i
                batch.n_seq_id[i-start] = 1
                batch.seq_id[i-start][0] = self.seq_num
                batch.logits[i-start] = True
            batch.n_tokens = end - start

            ret = llama_cpp.llama_decode(ctx, batch)
            if ret != 0:
                #raise Exception(f"Llama error {ret} with batch size {end - start} after tokenizing {start} tokens")
                print(f"Llama error {ret} with batch size {end - start} after tokenizing {start} tokens")
                return False

            for idx in range(end - start - 1):
                logits = llama_cpp.llama_get_logits_ith(ctx, idx)
                selected = None
                # Get top p logits and store their indices
                for token_id in range(vocab_size):
                    candidates_p.contents.data[token_id].id = llama_cpp.llama_token(token_id)
                    candidates_p.contents.data[token_id].logit = llama_cpp.ctypes.c_float(logits[token_id])
                    if token_id == tokens[idx + 1 + start]:
                        selected = (token_id, logits[token_id])
                candidates_p.contents.size = vocab_size
                candidates_p.contents.sorted = False
                llama_cpp.llama_sample_top_p(ctx, candidates_p, 0.9, self.context.operation.get_top_p())
                logits = [
                    (tkn.id, tkn.logit)
                    # TODO: Should this just be the length of the candidates_p array?
                    for tkn in candidates_p.contents.data[:self.context.operation.get_top_p()]
                ]
                if selected not in logits:
                    logits.append(selected)
                self.context.report_token(tokens[idx + 1 + start], logits)
                self.tokens.append(tokens[idx + 1 + start])

        self.logits = llama_cpp.llama_get_logits_ith(ctx, end - start - 1)
        self.current_role = desired_role

        if self.context.operation.name == "feed_tokens":
            self.context.completed = True
            self.is_done = True

        return True