# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Dict, List, Set

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch


class SuffixDecodingCache:
    """
    Cache for Suffix Decoding that wraps the C++ SuffixDecodingTree.
    Manages active and cached requests and handles eviction.
    """

    def __init__(self, max_tree_depth: int, max_cached_requests: int):
        # Initialize the C++ SuffixDecodingTree via registered custom op
        self.tree = torch.ops.vllm.SuffixDecodingTree(max_tree_depth)
        self.max_cached_requests = max_cached_requests

        # Track active requests (currently being processed)
        self.active_requests: Set[str] = set()

        # Track cached requests (finished but kept for potential reuse).
        # Used as a simple FIFO for eviction.
        self.cached_requests: List[str] = []

        # Map string request IDs to integer IDs required by the C++ tree
        self.req_id_to_int: Dict[str, int] = {}
        self.next_id = 0

    def _get_id(self, req_id: str) -> int:
        if req_id not in self.req_id_to_int:
            # Check for integer overflow? C++ uses int32 for sequence ID in some places,
            # but we use int64 in bindings. SuffixTree uses int32 map.
            # We should recycle IDs if we delete them, but for now simple increment.
            # If we run for a very long time, this might overflow int32 (2 billion).
            # SuffixTree remove() frees the ID in the tree, but we need to manage mapping.
            # Given this is a cache, we could reuse IDs, but complexity is higher.
            self.req_id_to_int[req_id] = self.next_id
            self.next_id += 1
        return self.req_id_to_int[req_id]

    def start_request(self, req_id: str, prompt_token_ids: torch.Tensor):
        seq_id = self._get_id(req_id)
        if req_id in self.cached_requests:
            self.cached_requests.remove(req_id)
        self.active_requests.add(req_id)
        # Extend the tree with prompt tokens.
        # SuffixTree.extend internally handles adding to the tree.
        self.tree.extend(seq_id, prompt_token_ids)

    def add_active_response(self, req_id: str, token_ids: List[int]):
        if not token_ids:
            return
        seq_id = self._get_id(req_id)
        # Convert list to tensor efficiently
        # Use int32 as SuffixTree expects int32 mostly, but bindings handle conversion
        t = torch.tensor(token_ids, dtype=torch.int32, device="cpu")
        self.tree.extend(seq_id, t)

    def stop_request(self, req_id: str):
        if req_id in self.active_requests:
            self.active_requests.remove(req_id)
            if req_id not in self.cached_requests:
                self.cached_requests.append(req_id)

            # Evict if too many
            if self.max_cached_requests > 0:
                while len(self.cached_requests) > self.max_cached_requests:
                    evict_id = self.cached_requests.pop(0)
                    self.remove_request(evict_id)
            elif self.max_cached_requests == 0:
                # If caching is disabled, remove immediately
                self.cached_requests.remove(req_id)
                self.remove_request(req_id)

    def evict_cached_response(self, req_id: str):
        if req_id in self.cached_requests:
            self.cached_requests.remove(req_id)
        self.remove_request(req_id)

    def remove_request(self, req_id: str):
        if req_id in self.req_id_to_int:
            seq_id = self.req_id_to_int[req_id]
            self.tree.remove(seq_id)
            del self.req_id_to_int[req_id]

    def speculate(self, req_id: str, context: torch.Tensor, **kwargs):
        self._get_id(req_id)  # Ensure ID exists
        return self.tree.speculate(context, **kwargs)


class SuffixDecodingProposer:
    """
    Speculative decoding proposer for Suffix Decoding (https://arxiv.org/pdf/2411.04975).
    Uses the internal vLLM C++ SuffixDecodingTree implementation.
    """

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len

        # Initialize global cache
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_decoding_max_tree_depth,
            max_cached_requests=config.suffix_decoding_max_cached_requests,
        )

    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        """
        Propose speculative tokens for each request in the input batch.
        """
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                draft_token_ids.append([])
                continue

            # Skip requests that require sampling parameters that are not
            # supported with speculative decoding.
            req_id = input_batch.req_ids[i]
            if req_id in input_batch.spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue

            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]
            if req_id not in self.suffix_cache.active_requests:
                if req_id in self.suffix_cache.cached_requests:
                    # Reset the suffix cache for this request.
                    self.suffix_cache.evict_cached_response(req_id)
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                # Start a new request, this will build the suffix tree for that prompt.
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            # Append the newly sampled ids to the suffix cache for this request.
            self.suffix_cache.add_active_response(req_id, sampled_ids)

            # Suffix decoding only uses the most recent tokens up to max_tree_depth, so
            # we extract the pattern from the end of the input.
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]

            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(
                    self.num_speculative_tokens, self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
                max_spec_offset=0.0,  # Default per bindings
                use_tree_spec=False,  # Default or configurable?
            )

            # draft is a SuffixDecodingDraft script object
            draft_token_ids.append(draft.token_ids)

        # Stop requests that were not seen in the input batch.
        # input_batch.req_id_to_index keys are the current active requests in batch
        current_reqs = set(input_batch.req_id_to_index.keys())
        # We need to stop requests that are in suffix_cache.active_requests but NOT in current_reqs
        for req_id in list(self.suffix_cache.active_requests):
            if req_id not in current_reqs:
                self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        pass
