import asyncio
import logging
import threading
import time

import llama_cpp

from typing import List

from .model_config import ModelConfig
from .request import LlamaRequest

class Llama(object):
    def __init__(self, config: ModelConfig):
        llama_cpp.llama_backend_init(False) # Must be called once at the start of each program
        model_params = llama_cpp.llama_model_default_params()
        model_params.n_gpu_layers = 33
        params = llama_cpp.llama_context_default_params()
        params.n_ctx = config.context_size
        self.model = llama_cpp.llama_load_model_from_file(config.model_filename.encode('utf-8'), model_params)
        self.ctx = llama_cpp.llama_new_context_with_model(self.model, params)
        self.context_size = params.n_ctx
        self.vocab_size = llama_cpp.llama_n_vocab(self.model)
        self.temperature = config.temperature
        self.requests = []
        self.batch_size = config.batch_size
        self.batch_max_tokens = config.batch_max_tokens

        _arr = (llama_cpp.llama_token_data * self.vocab_size)(
            *[
                llama_cpp.llama_token_data(token_id, 0.0, 0.0)
                for token_id in range(self.vocab_size)
            ]
        )
        self.candidates_p = llama_cpp.ctypes.pointer(llama_cpp.llama_token_data_array(_arr, len(_arr), False))

    def queue_request(self, request: LlamaRequest) -> LlamaRequest:
        self.requests.append(request)
        return request

    def _run_batch(self, requests: List[LlamaRequest]):
        batch = llama_cpp.llama_batch_init(self.batch_size, 0, 1)
        logits = [[None] * len(requests)]
        seq_num = 0

        while True:
            # First check if any beams are waiting to be initialized
            for request_idx, request in enumerate(requests):
                if len(logits[request_idx]) < len(request.beams):
                    logits[request_idx].extend([None] * (len(request.beams) - len(logits[request_idx])))
                for beam_idx, beam in enumerate(request.beams):
                    if beam.seq_num == -1:
                        # Copy the KV cache & logits for the current beam to the
                        # new beam and assign the new sequence number
                        if beam.parent is not None:
                            llama_cpp.llama_kv_cache_seq_cp(self.ctx, beam.parent.seq_num, seq_num, -1, -1)
                            logits[request_idx][beam_idx] = logits[request_idx][beam.parent.index]
                        beam.seq_num = seq_num
                        seq_num += 1

            # Next, check if any beams are waiting for a batch of fixed tokens
            # to be decoded
            for request_idx, request in enumerate(requests):
                for beam_idx, beam in enumerate(request.beams):
                    decoded_logits = beam.decode_tokens(self.ctx, self.model, batch, self.batch_size)
                    if decoded_logits is not None:
                        logits[request_idx][beam_idx] = decoded_logits

            batch.n_tokens = 0
            beam_index_map = {}
            for request_idx, request in enumerate(requests):
                for beam_idx, beam in enumerate(request.beams):
                    if beam_idx >= len(logits[request_idx]):
                        continue
                    next_token_id = beam.decode_next(
                        self.ctx,
                        self.model,
                        self.vocab_size,
                        self.temperature,
                        logits[request_idx][beam_idx],
                        self.candidates_p,
                    )
                    if next_token_id is not None:
                        beam_index_map[(request_idx, beam_idx)] = batch.n_tokens
                        batch.token[batch.n_tokens] = next_token_id
                        batch.pos[batch.n_tokens] = beam.pos
                        batch.n_seq_id[batch.n_tokens] = 1
                        batch.seq_id[batch.n_tokens][0] = beam.seq_num
                        batch.logits[batch.n_tokens] = True
                        batch.n_tokens += 1
                        beam.pos += 1

            # TODO: Clean up KV cache for finished beams

            if batch.n_tokens == 0:
                all_done = all(beam.is_done() for request in requests for beam in request.beams)
                if all_done:
                    break
                continue

            ret = llama_cpp.llama_decode(self.ctx, batch)
            if ret != 0:
                raise Exception("LLAMA ERROR " + str(ret))

            logits = [
                [
                    llama_cpp.llama_get_logits_ith(self.ctx, beam_index_map[(request_idx, beam_idx)])
                    if (request_idx, beam_idx) in beam_index_map else None
                    for beam_idx in range(len(request.beams))
                ]
                for request_idx, request in enumerate(requests)
            ]

        llama_cpp.llama_batch_free(batch)
        llama_cpp.llama_kv_cache_clear(self.ctx)

        for request in requests:
            logging.info(f"Completed request {request.id}")
            request.completed = True

    def _process_queue(self, run_forever: bool = True):
        # TODO: Replace this with a queue that can schedule & deschedule
        # individual beams dynamically
        while True:
            pending_requests = list(filter(lambda x: not x.completed, self.requests))
            if not pending_requests:
                if not run_forever:
                    break
                logging.info("Queue: No pending requests")
                time.sleep(1.0)
                continue
            self._run_batch([pending_requests[0]])
            time.sleep(1.0)

    def run_in_background(self):
        thread = threading.Thread(target=self._process_queue, daemon=True)
        thread.start()

    def run_once(self):
        self._process_queue(run_forever=False)

    def cleanup(self):
        llama_cpp.llama_free(self.ctx)