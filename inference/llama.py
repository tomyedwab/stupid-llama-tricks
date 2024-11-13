import llama_cpp

from typing import List

from .model_config import ModelConfig
from .interpreter import Interpreter
from .llama_operation import LlamaOperation
from .util import token_to_string

class Llama(object):
    def __init__(self, config: ModelConfig):
        llama_cpp.llama_backend_init(False) # Must be called once at the start of each program
        self.model_params = llama_cpp.llama_model_default_params()
        self.model_params.n_gpu_layers = 33
        self.params = llama_cpp.llama_context_default_params()
        self.params.n_ctx = config.context_size
        self.context_size = self.params.n_ctx
        self.temperature = config.temperature
        self.batch_size = config.batch_size
        self.batch_max_tokens = config.batch_max_tokens
        self.model_filename = config.model_filename
        self.next_seq_num = 0
        self.operations = {}

        self.model = llama_cpp.llama_load_model_from_file(self.model_filename.encode('utf-8'), self.model_params)
        self.ctx = llama_cpp.llama_new_context_with_model(self.model, self.params)
        self.vocab_size = llama_cpp.llama_n_vocab(self.model)

        _arr = (llama_cpp.llama_token_data * self.vocab_size)(
            *[
                llama_cpp.llama_token_data(token_id, 0.0, 0.0)
                for token_id in range(self.vocab_size)
            ]
        )
        self.candidates_p = llama_cpp.ctypes.pointer(llama_cpp.llama_token_data_array(_arr, len(_arr), False))

        self.batch = llama_cpp.llama_batch_init(self.batch_size, 0, 1)

        self.tokens_for_role = {
            "null_system": self.tokenize("<|system|>\n"),
            "null_user": self.tokenize("<|user|>\n"),
            "null_assistant": self.tokenize("<|assistant|>\n"),
            "end_system": self.tokenize("<|end|>\n<|system|>\n"),
            "end_user": self.tokenize("<|end|>\n<|user|>\n"),
            "end_assistant": self.tokenize("<|end|>\n<|assistant|>\n"),
        }

    def stop(self):
        llama_cpp.llama_batch_free(self.batch)
        llama_cpp.llama_kv_cache_clear(self.ctx)
        llama_cpp.llama_free(self.ctx)

    def token_map(self):
        return { token_id: token_to_string(self.model, token_id) for token_id in range(self.vocab_size) }

    def tokenize(self, text: str) -> List[int]:
        encoded = text.encode('utf-8')
        tokens = (llama_cpp.llama_token * int(4096))()
        n_tokens = llama_cpp.llama_tokenize(self.model, encoded, len(encoded), tokens, 4096, True, False)
        return list(tokens[:n_tokens])

    def _schedule_runnable_operations(self):
        # For any new operations, assign a sequence number, copying from the
        # parent operation if there is one.
        for operation in self.operations.values():
            if operation.seq_num >= 0 or operation.is_done:
                continue

            new_seq_num = self.next_seq_num

            if operation.parent is not None and operation.parent.seq_num >= 0:
                # Copy the KV cache from the parent beam to the new beam
                print(f"Copying KV cache from seq {operation.parent.seq_num} to {new_seq_num}")
                llama_cpp.llama_kv_cache_seq_cp(self.ctx, operation.parent.seq_num, new_seq_num, -1, -1)
                operation.logits = operation.parent.logits
                operation.parent = None

            elif len(operation.tokens) > 0:
                # If the operation has previously been descheduled, or the
                # parent KV cache has been cleared, generate a new cache here
                print(f"Restoring KV cache for operation {operation.context.id}")
                if not operation.restore_tokens(self.ctx, self.batch, self.batch_size, new_seq_num):
                    print(f"Cannot schedule operation {operation.context.id} due to error restoring KV cache")
                    llama_cpp.llama_kv_cache_seq_rm(self.ctx, new_seq_num, -1, -1)
                    continue

            print(f"Assigning operation {operation.context.id} seq_num {new_seq_num}")
            operation.seq_num = new_seq_num
            self.next_seq_num += 1

        # Now we are safe to clear KV cache from any operations that are done
        for operation in self.operations.values():
            if operation.seq_num >= 0 and operation.is_done:
                print(f"Descheduling operation {operation.context.id} because it is done")
                llama_cpp.llama_kv_cache_seq_rm(self.ctx, operation.seq_num, -1, -1)
                operation.seq_num = -1

    def _decode_operation_tokens(self):
        for operation in self.operations.values():
            if operation.seq_num < 0:
                continue

            if not operation.decode_tokens(self.ctx, self.model, self.vocab_size, self.candidates_p, self.batch, self.batch_size, self.tokens_for_role):
                # There was an error decoding tokens, so deschedule the beam for now
                print(f"Descheduling operation {operation.context.id} due to error")
                llama_cpp.llama_kv_cache_seq_rm(self.ctx, operation.seq_num, -1, -1)
                operation.seq_num = -1
                operation.logits = None
    
    def _decode_batch(self):
        self.batch.n_tokens = 0
        # Map each token index to the operation that produced it
        operation_result_indices = {}

        # Iterate over all operations and decode tokens
        for operation in self.operations.values():
            if operation.seq_num < 0:
                continue
            if operation.logits is None:
                continue
            next_token_id = operation.decode_next(
                self.ctx,
                    self.model,
                    self.vocab_size,
                    self.temperature,
                    self.candidates_p,
                )

            if next_token_id is not None:
                operation_result_indices[self.batch.n_tokens] = operation
                self.batch.token[self.batch.n_tokens] = next_token_id
                self.batch.pos[self.batch.n_tokens] = len(operation.tokens)
                self.batch.n_seq_id[self.batch.n_tokens] = 1
                self.batch.seq_id[self.batch.n_tokens][0] = operation.seq_num
                self.batch.logits[self.batch.n_tokens] = True
                self.batch.n_tokens += 1
                operation.tokens.append(next_token_id)

        if self.batch.n_tokens > 0:
            ret = llama_cpp.llama_decode(self.ctx, self.batch)
            if ret != 0:
                # TODO: Deschedule beams until this succeeds
                raise Exception("LLAMA ERROR " + str(ret))

            for operation in self.operations.values():
                if operation.seq_num < 0:
                    continue
                operation.logits = None

            for token_idx, operation in operation_result_indices.items():
                operation.logits = llama_cpp.llama_get_logits_ith(self.ctx, token_idx)

    def run_loop(self, interpreter: Interpreter):
        contexts = interpreter.loop()

        # Create any operations for operations that are not already created
        for context in contexts:
            if context.id not in self.operations:
                self.operations[context.id] = LlamaOperation(
                    context, self.operations.get(context.parent_id))

        # Schedule any runnable operations
        self._schedule_runnable_operations()

        # Decode fixed tokens (e.g. prompts) for any runnable operations
        self._decode_operation_tokens()

        # Decode the next batch of tokens
        self._decode_batch()