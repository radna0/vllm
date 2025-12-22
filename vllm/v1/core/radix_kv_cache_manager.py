from typing import Tuple, List, Sequence, Optional

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheSpec, FullAttentionSpec
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.core.radix_cache import RadixCache
from vllm.v1.request import Request


class RadixFullAttentionManager(FullAttentionManager):
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        radix_eviction_policy: str = "lru",
    ) -> None:
        super().__init__(
            kv_cache_spec, block_pool, kv_cache_group_id, dcp_world_size, pcp_world_size
        )
        self.radix_cache = RadixCache(
            block_size=self.block_size,
            eviction_policy=radix_eviction_policy,
        )

    # Note: We override this method but change the signature to accept token_ids
    # This makes it incompatible with the standard KVCacheCoordinator but suitable for
    # specific benchmarking where we control the calls.
    def find_longest_cache_hit(
        self,
        token_ids: List[int],  # CHANGED: Accepts token_ids instead of block_hashes
        max_length: int,
        kv_cache_group_ids: list[
            int
        ],  # Ignored, we assume single group for this proof of concept
        block_pool: BlockPool,  # Ignored, we use self.radix_cache
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:

        # RadixCache match
        blocks, _ = self.radix_cache.match_prefix(token_ids)

        # Enforce max_length constraint
        # Blocks are KVCacheBlock objects.
        # Calculate tokens covered.
        num_blocks = len(blocks)
        cached_tokens = num_blocks * self.block_size

        if cached_tokens > max_length:
            # Truncate
            allowed_blocks = max_length // self.block_size
            blocks = blocks[:allowed_blocks]

        # Handle eagle (drop last block logic)
        if use_eagle and blocks:
            blocks.pop()

        # Handle alignment (pop blocks if not aligned)
        while (
            self.block_size != alignment_tokens
            and len(blocks) * self.block_size % alignment_tokens != 0
        ):
            blocks.pop()

        # Wrap in tuple for compatibility with return signature (per group)
        # We return a single list for the single group we manage.
        # But the signature expects tuple[list[KVCacheBlock], ...] matching kv_cache_group_ids

        # Touch blocks to increment ref_cnt for the Request
        if blocks:
            # block_pool.touch expects tuple of sequences
            block_pool.touch((blocks,))

        # Assuming len(kv_cache_group_ids) == 1 for this manager usage
        return (blocks,)

    def save_new_computed_blocks(
        self, request_id: str, new_computed_blocks: Sequence[KVCacheBlock]
    ) -> None:
        super().save_new_computed_blocks(request_id, new_computed_blocks)
        # We don't update RadixTree here necessarily, because we might not have the token_ids here easily?
        # Standard manager updates self.req_to_blocks.
        # To update RadixTree, we need token_ids.
        # See cache_blocks()

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """
        Cache the blocks for the request into the Radix Tree.
        """
        # We can implement this because we have 'request' object which has token_ids.

        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        # Get blocks to cache
        blocks = self.req_to_blocks[request.request_id]
        blocks_to_cache = blocks[:num_full_blocks]  # Cache all full blocks?

        # Standard manager uses block_pool.cache_full_blocks which increments ref_cnt/hashes.
        # Here we insert into Radix Tree.

        # We need the token_ids corresponding to these blocks.
        # request.all_token_ids?
        token_ids = request.all_token_ids[: num_full_blocks * self.block_size]

        # Insert into Radix Cache
        # Note: blocks_to_cache must match token_ids length-wise (block-aligned)
        self.radix_cache.insert(token_ids, blocks_to_cache)

        # Increment ref_cnt for Radix ownership
        self.block_pool.touch((blocks_to_cache,))

        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        # Standard free releases blocks to BlockPool.
        # RadixCache manages logical eviction.
        # When we free a request, we dec_lock_ref on the nodes?
        # SGLang RadixCache uses lock_ref to prevent eviction of "in-use" blocks.

        # But this 'free' means the request is finished or aborted.
        # In SGLang, cache_finished_req is called.

        # For this prototype:
        # We should probably just call super().free(request_id) to release the blocks to the pool.
        # BUT wait, if we release to BlockPool, they perform physical free.
        # RadixCache might still hold references to them in 'value'.
        # If BlockPool reuses block ID 5, but RadixCache thinks Block 5 is valid for prefix "ABC", we have corruption.

        # This implies RadixCache MUST Own the blocks, or BlockPool must not free them until RadixCache says so.
        # In this adapter:
        # BlockPool has 'enable_caching=False'. So free_blocks() puts them in free_queue immediately.
        # THIS IS BAD if RadixCache wants to keep them.

        # Implementation for Benchmark:
        # We assume BlockPool is infinite or we manage eviction manually.
        # Benchmark will test "Hit Rate".
        # If we use infinite pool, we never evict. Hit rate is 100% for repeats.

        # To test eviction:
        # We need `allocate_new_blocks` to trigger eviction if pool is empty.

        # Standard free releases blocks to BlockPool.
        # It decrements ref_cnt.
        # Since we incremented ref_cnt for Radix in cache_blocks,
        # and for Request in allocate/find_hit,
        # calling free() here removes the Request's reference.
        # The Radix reference remains until eviction.

        super().free(request_id)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock]:
        # Try to allocate from BlockPool
        try:
            return super().allocate_new_blocks(request_id, num_tokens)
        except ValueError:
            # OOM?
            # Trigger Radix eviction
            # needed = num_new_blocks
            # evicted_blocks = self.radix_cache.evict(needed * self.block_size)
            # self.block_pool.free_blocks(evicted_blocks)
            # Retry allocation
            # This logic requires knowing how many blocks super().allocate_new_blocks wanted.
            pass

        # Re-implement allocation logic to handle eviction
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = (
            num_tokens + self.block_size - 1
        ) // self.block_size  # cdiv
        num_new_blocks = num_required_blocks - len(req_blocks)

        if num_new_blocks <= 0:
            return []

        free_blocks_count = self.block_pool.get_num_free_blocks()
        if free_blocks_count < num_new_blocks:
            needed = num_new_blocks - free_blocks_count
            # Evict from Radix
            # Evict blocks = needed
            # Tokens to evict = needed * block_size
            evicted_blocks = self.radix_cache.evict(needed * self.block_size)
            self.block_pool.free_blocks(evicted_blocks)

        # Now allocate
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        return new_blocks
