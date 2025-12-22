import time
import random
import torch
from dataclasses import dataclass, field
from typing import List, Optional

# Monkeypatch vLLM platform detection to force CUDA
import vllm.platforms

vllm.platforms.builtin_platform_plugins = {"cuda": vllm.platforms.cuda_platform_plugin}

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock, get_block_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.core.radix_kv_cache_manager import RadixFullAttentionManager
from vllm.v1.request import Request


@dataclass
class MockRequest:
    request_id: str
    token_ids: List[int]
    block_hashes: List[BlockHash] = field(default_factory=list)
    all_token_ids: List[int] = field(
        default_factory=list
    )  # Adapter for Request interface

    def __post_init__(self):
        self.all_token_ids = self.token_ids


def generate_block_hashes(token_ids: List[int], block_size: int) -> List[BlockHash]:
    # Simulate block hashing
    # In vLLM request, this is done incrementally.
    hashes = []
    for i in range(0, len(token_ids), block_size):
        chunk = tuple(token_ids[i : i + block_size])
        if len(chunk) == block_size:
            # Simple hash for simulation
            h = hash(chunk)
            hashes.append(h)
    return hashes


def run_benchmark():
    block_size = 16
    num_blocks = 100
    cache_spec = FullAttentionSpec(block_size=block_size)

    # Setup Hash Manager
    pool_hash = BlockPool(num_blocks, enable_caching=True, hash_block_size=block_size)
    manager_hash = FullAttentionManager(cache_spec, pool_hash, kv_cache_group_id=0)

    # Setup Radix Manager
    pool_radix = BlockPool(
        num_blocks, enable_caching=False, hash_block_size=block_size
    )  # Disable BlockPool caching
    manager_radix = RadixFullAttentionManager(
        cache_spec, pool_radix, kv_cache_group_id=0, radix_eviction_policy="lru"
    )

    # Workload Simulation
    # Scenario: shared system prompt
    # Each user has multi-turn conversation.

    system_prompt = [random.randint(0, 10000) for _ in range(64)]
    users = [f"user_{i}" for i in range(5)]
    user_history = {u: list(system_prompt) for u in users}

    requests = []
    # Generate 50 requests
    for _ in range(50):
        # Pick random user
        u = random.choice(users)
        hist = user_history[u]

        # New input
        new_input = [random.randint(0, 10000) for _ in range(random.randint(10, 64))]
        full_tokens = hist + new_input

        req = MockRequest(request_id=f"{u}_{len(hist)}", token_ids=full_tokens)
        req.block_hashes = generate_block_hashes(full_tokens, block_size)

        requests.append((u, req))

        # Update history
        user_history[u] = full_tokens

    # Metric containers
    results = {
        "hash": {"hits": 0, "total_tokens": 0, "latency": 0.0, "evictions": 0},
        "radix": {"hits": 0, "total_tokens": 0, "latency": 0.0, "evictions": 0},
    }

    print(f"Running benchmark with {len(requests)} requests...")

    # Run Hash Manager
    t0 = time.time()
    for u, req in requests:
        # 1. Check Cache Hit
        start = time.perf_counter()

        # Hash Manager needs BlockHashes
        hit_blocks_list = manager_hash.find_longest_cache_hit(
            block_hashes=req.block_hashes,
            max_length=len(req.token_ids),
            kv_cache_group_ids=[0],
            block_pool=pool_hash,
            kv_cache_spec=cache_spec,
            use_eagle=False,
            alignment_tokens=block_size,
        )  # Returns tuple([blocks], ...)

        hit_blocks = hit_blocks_list[0]
        hit_len = len(hit_blocks) * block_size

        end = time.perf_counter()
        results["hash"]["latency"] += end - start
        results["hash"]["hits"] += hit_len
        results["hash"]["total_tokens"] += len(req.token_ids)

        # 2. Allocate remaining
        remaining_tokens = len(req.token_ids)
        # In simulation, we just alloc and cache

        # Alloc
        try:
            new_blocks = manager_hash.allocate_new_blocks(
                req.request_id, remaining_tokens
            )
        except ValueError:
            # Simulate freeing LRU? BlockPool implementation usually doesn't auto-evict in allocate_new_blocks
            # unless we implement a higher-level scheduler loop.
            # But wait, BlockPool.get_new_blocks DOES evict if enable_caching=True!
            # It evicts from free_block_queue which contains cached blocks (ref_cnt=0).
            # So Hash manager handles eviction automatically during alloc.
            # If fails, real OOM.
            # Reset pool for fairness? Or just skip
            print("Hash Manager OOM - skipping request")
            continue

        # Save computed
        # manager_hash.save_new_computed_blocks(req.request_id, [])
        # Actually standard flow is:
        # 1. find hit
        # 2. alloc
        # 3. compute (skip)
        # 4. cache_blocks

        # Mocking the allocation filling:
        # We need to assign hashes to the new blocks so they can be cached.
        # allocate_new_blocks returns 'new_blocks' which are empty.

        # In real vLLM, Request updates block hashes.
        # We need to manually set block hashes on the allocated blocks to simulate computation.

        cached_blocks_cnt = len(hit_blocks)
        blocks_needed = (len(req.token_ids) + block_size - 1) // block_size

        # Assign blocks to request locally (mocking Scheduler)
        manager_hash.req_to_blocks[req.request_id] = hit_blocks + new_blocks

        # Set hashes for new blocks
        all_hashes = req.block_hashes
        # Ensure new_blocks have hashes
        for i, blk in enumerate(new_blocks):
            # Corresponding hash index
            hash_idx = cached_blocks_cnt + i
            if hash_idx < len(all_hashes):
                blk.block_hash = None  # Reset first?
                # BlockPool.cache_full_blocks needs Request.block_hashes
                pass

        # Cache
        # cache_blocks uses request.block_hashes
        # Mock Request object needs block_hashes attribute
        manager_hash.cache_blocks(req, len(req.token_ids))

        # Release blocks for previous request of this user?
        # In this simulation, user history grows.
        # So previous request is inherently Freeable?
        # Actually, in vLLM, requests are freed when done.
        # But we want to Simulate CONTEXT REUSE.
        # If we free the request, the blocks become ref_cnt=0 but stay in cache (if enabled).
        # So YES, we should free the previous request of the user to allow eviction.
        # But we need to cache it first.

        manager_hash.free(req.request_id)

    time_hash = time.time() - t0

    print("\n--- Hash Manager Results ---")
    print(f"Time: {time_hash:.2f}s")
    print(
        f"Hit Rate: {results['hash']['hits'] / results['hash']['total_tokens'] * 100:.2f}%"
    )
    print(f"Latency: {results['hash']['latency']:.5f}s (Total lookup time)")

    # Run Radix Manager
    t0 = time.time()
    for u, req in requests:
        # 1. Check Cache Hit
        start = time.perf_counter()

        # Radix Manager needs token_ids
        hit_blocks_list = manager_radix.find_longest_cache_hit(
            token_ids=req.token_ids,
            max_length=len(req.token_ids),
            kv_cache_group_ids=[0],
            block_pool=pool_radix,
            kv_cache_spec=cache_spec,
            use_eagle=False,
            alignment_tokens=block_size,
        )

        hit_blocks = hit_blocks_list[0]
        hit_len = len(hit_blocks) * block_size

        end = time.perf_counter()
        results["radix"]["latency"] += end - start
        results["radix"]["hits"] += hit_len
        results["radix"]["total_tokens"] += len(req.token_ids)

        # 2. Alloc
        remaining_tokens = len(req.token_ids)
        try:
            new_blocks = manager_radix.allocate_new_blocks(
                req.request_id, remaining_tokens
            )
        except Exception as e:
            print(f"Radix Manager Alloc Error: {e}")
            continue

        # Assign blocks
        manager_radix.req_to_blocks[req.request_id] = hit_blocks + new_blocks

        # 3. Cache
        manager_radix.cache_blocks(req, len(req.token_ids))

        # 4. Free (mark as evictable)
        manager_radix.free(req.request_id)

    time_radix = time.time() - t0

    print("\n--- Radix Manager Results ---")
    print(f"Time: {time_radix:.2f}s")
    print(
        f"Hit Rate: {results['radix']['hits'] / results['radix']['total_tokens'] * 100:.2f}%"
    )
    print(f"Latency: {results['radix']['latency']:.5f}s (Total lookup time)")


if __name__ == "__main__":
    run_benchmark()
