# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RadixKVCacheCoordinator - SGLang-style RadixCache integration for vLLM.

This coordinator uses a Radix tree for prefix matching instead of hash-based
lookups, providing better performance for workloads with shared prefixes.
"""
from collections.abc import Sequence
from typing import List, Tuple, Optional

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.radix_cache import RadixCache
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
)
from vllm.v1.request import Request


class RadixKVCacheCoordinator(KVCacheCoordinator):
    """
    KVCacheCoordinator using SGLang's RadixCache for prefix matching.

    This provides tree-based prefix matching with leaf-first eviction,
    which preserves shared prefixes (system prompts) under memory pressure.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            use_eagle=use_eagle,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )

        # Initialize RadixCache for prefix matching
        self.radix_cache = RadixCache(
            block_size=hash_block_size,
            eviction_policy="lru",
            disable=not enable_caching,
        )

        # Get block size from first KV cache group
        self.block_size = hash_block_size

        # Track which requests have been inserted into radix cache
        self._request_to_blocks: dict[str, List[KVCacheBlock]] = {}

    def find_longest_cache_hit(
        self,
        request: Request,
        block_hashes: BlockHashListWithBlockSize,
    ) -> tuple[tuple[Sequence[KVCacheBlock], ...], int]:
        """
        Find the longest prefix match using RadixCache.

        Args:
            request: The request to find cache hits for.
            block_hashes: Block hashes with block sizes.

        Returns:
            Tuple of (matched blocks per manager, hit length in tokens).
        """
        if not self.enable_caching:
            return self._get_empty_hit(), 0

        # Get token IDs from request
        token_ids = list(request.all_token_ids)

        # Use RadixCache for prefix matching
        matched_blocks, last_node = self.radix_cache.match_prefix(token_ids)

        hit_length = len(matched_blocks) * self.block_size

        # Convert to tuple format expected by vLLM
        # For single-group models, wrap in tuple
        if len(self.single_type_managers) == 1:
            return (tuple(matched_blocks),), hit_length

        # For multi-group (e.g., hybrid attention), replicate blocks
        result = []
        for manager in self.single_type_managers:
            if isinstance(manager, FullAttentionManager):
                result.append(tuple(matched_blocks))
            else:
                # Non-full attention managers don't use prefix caching
                result.append(())

        return tuple(result), hit_length

    def update_prefix_cache(
        self,
        request: Request,
        allocated_blocks: list[list[KVCacheBlock]],
    ) -> None:
        """
        Insert the request's blocks into the RadixCache.

        Args:
            request: The request.
            allocated_blocks: Blocks allocated for each KV cache group.
        """
        if not self.enable_caching:
            return

        token_ids = list(request.all_token_ids)

        # Get blocks for the first (full attention) group
        if allocated_blocks and allocated_blocks[0]:
            blocks = allocated_blocks[0]

            # Determine priority based on request type
            # System prompts/shared prefixes should have higher priority
            priority = 1 if len(token_ids) > 1000 else 0  # Simple heuristic

            # Insert into radix cache
            self.radix_cache.insert(
                token_ids=token_ids,
                blocks=blocks,
                priority=priority,
            )

            # Track for potential cleanup
            self._request_to_blocks[request.request_id] = blocks

    def _get_empty_hit(self) -> tuple[Sequence[KVCacheBlock], ...]:
        """Return empty hit result for all managers."""
        return tuple(() for _ in self.single_type_managers)

    def get_radix_cache_metrics(self) -> dict:
        """Get RadixCache metrics for monitoring."""
        return self.radix_cache.metrics
