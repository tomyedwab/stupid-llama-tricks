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
        self.model_params = llama_cpp.llama_model_default_params()
        self.model_params.n_gpu_layers = 33
        self.params = llama_cpp.llama_context_default_params()
        self.params.n_ctx = config.context_size
        self.context_size = self.params.n_ctx
        self.temperature = config.temperature
        self.batch_size = config.batch_size
        self.batch_max_tokens = config.batch_max_tokens
        self.model_filename = config.model_filename
        self.requests = []
        self.stop_requested = False
        self.next_seq_num = 0

    def _start(self):
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

    def _stop(self):
        llama_cpp.llama_batch_free(self.batch)
        llama_cpp.llama_kv_cache_clear(self.ctx)
        llama_cpp.llama_free(self.ctx)

    async def do_request(self, request: LlamaRequest) -> any:
        print(f"Queuing request {request.id}")
        self.requests.append(request)
        request.start_task(asyncio.get_event_loop())
        return await request.get_result()

    def request_stop(self):
        self.stop_requested = True

    async def _update_requests(self):
        # Wait for all beams to select their next action
        await asyncio.gather(*[
            beam.wait_for_next_action()
            for request in self.requests
            for beam in request.beams
        ])

    def _schedule_runnable_beams(self):
        # First find any beams that are scheduled that are marked "done"
        for request in self.requests:
            for beam in request.beams:
                if beam.seq_num >= 0 and beam.is_done():
                    print(f"Descheduling beam {beam.index} for request {request.id} because it is done")
                    llama_cpp.llama_kv_cache_seq_rm(self.ctx, beam.seq_num, -1, -1)
                    beam.logits = None
                    beam.seq_num = -1

        for request in self.requests:
            for beam in request.beams:
                if beam.seq_num >= 0 or not beam.is_runnable():
                    continue

                new_seq_num = self.next_seq_num

                # TODO: If a beam was previously descheduled, recover its KV cache
                # Is the beam being initialized from a parent beam?
                if beam.parent is not None:
                    if beam.parent.seq_num < 0:
                        # The parent is not itself initialized yet, so we need to wait
                        continue
                    # Copy the KV cache & logits for the current beam to the
                    # new beam and assign the new sequence number
                    llama_cpp.llama_kv_cache_seq_cp(self.ctx, beam.parent.seq_num, new_seq_num, -1, -1)
                    beam.logits = beam.parent.logits

                print(f"Assigning request {request.id} beam {beam.index} seq_num {new_seq_num}")
                beam.seq_num = new_seq_num
                self.next_seq_num += 1

    async def _decode_beam_tokens(self):
        for request in self.requests:
            for beam in request.beams:
                if beam.seq_num < 0:
                    continue

                if not await beam.decode_tokens(self.ctx, self.model, self.batch, self.batch_size):
                    # There was an error decoding tokens, so deschedule the beam for now
                    print(f"Descheduling beam {beam.index} for request {request.id} due to error")
                    llama_cpp.llama_kv_cache_seq_rm(self.ctx, beam.seq_num, -1, -1)
                    beam.seq_num = -1
                    beam.logits = None
    
    async def _decode_batch(self):
        self.batch.n_tokens = 0
        # Map each token index to the beam that produced it
        beam_result_indices = {}

        # Iterate over all beams and decode tokens
        for request in self.requests:
            for beam in request.beams:
                if beam.logits is None:
                    continue
                next_token_id = await beam.decode_next(
                    self.ctx,
                    self.model,
                    self.vocab_size,
                    self.temperature,
                    self.candidates_p,
                )

                if next_token_id is not None:
                    beam_result_indices[self.batch.n_tokens] = beam
                    self.batch.token[self.batch.n_tokens] = next_token_id
                    self.batch.pos[self.batch.n_tokens] = beam.pos
                    self.batch.n_seq_id[self.batch.n_tokens] = 1
                    self.batch.seq_id[self.batch.n_tokens][0] = beam.seq_num
                    self.batch.logits[self.batch.n_tokens] = True
                    self.batch.n_tokens += 1
                    beam.pos += 1

        if self.batch.n_tokens > 0:
            ret = llama_cpp.llama_decode(self.ctx, self.batch)
            if ret != 0:
                # TODO: Deschedule beams until this succeeds
                raise Exception("LLAMA ERROR " + str(ret))

            for request in self.requests:
                for beam in request.beams:
                    beam.logits = None

            for token_idx, beam in beam_result_indices.items():
                beam.logits = llama_cpp.llama_get_logits_ith(self.ctx, token_idx)

    async def run(self):
        self._start()
        while not self.stop_requested:
            # First, update the state of all requests
            await self._update_requests()

            # Schedule any runnable beams
            self._schedule_runnable_beams()

            # Decode fixed tokens (e.g. prompts) for any runnable beams
            await self._decode_beam_tokens()

            # Decode the next batch of tokens
            await self._decode_batch()

            # Yield control back to the event loop so that we don't starve other
            # tasks
            await asyncio.sleep(0)

        self._stop()