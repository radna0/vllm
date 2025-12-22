import time
import random
import torch
from dataclasses import dataclass
from typing import List, Optional


# Local mock removed, using sys.modules mock
@dataclass
class Request:
    request_id: str
    token_ids: List[int]
    arrival_time: float


# --- Mock Dependencies to avoid loading full vLLM stack (std::bad_alloc) ---
import sys
import os
from types import ModuleType

# Add project root to sys.path explicitly
project_root = "/workspace/aimo/DEV_VERSIONS/vllm_sglang"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock vllm.v1.core.kv_cache_utils
mock_kv_utils = ModuleType("vllm.v1.core.kv_cache_utils")


class MockKVCacheBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id

    def __repr__(self):
        return f"Block({self.block_id})"


mock_kv_utils.KVCacheBlock = MockKVCacheBlock
sys.modules["vllm.v1.core.kv_cache_utils"] = mock_kv_utils

# We simply let standard import find evict_policy since it is in sys.path now.
# However, if evict_policy imports other vllm stuff, we might care.
# evict_policy imports TreeNode via TYPE_CHECKING which is fine.

# Load RadixCache
import importlib.util

RADIX_CACHE_PATH = os.path.join(project_root, "vllm/v1/core/radix_cache.py")
spec = importlib.util.spec_from_file_location(
    "vllm.v1.core.radix_cache", RADIX_CACHE_PATH
)
radix_cache_module = importlib.util.module_from_spec(spec)
sys.modules["vllm.v1.core.radix_cache"] = radix_cache_module
spec.loader.exec_module(radix_cache_module)
RadixCache = radix_cache_module.RadixCache

# Use the MockKVCacheBlock as our block class
KVCacheBlock = MockKVCacheBlock


def run_benchmark():
    # Configuration
    BLOCK_SIZE = 16
    CONTEXT_LEN = 65536
    BATCH_SIZE = 8
    SHARED_PREFIX_LEN = 64512
    UNIQUE_LEN = CONTEXT_LEN - SHARED_PREFIX_LEN

    # Init RadixCache
    # We need a meaningful eviction policy.
    # For this benchmark we assume infinite memory first to test pure throughput of matching.
    # Then we can test eviction.
    radix_cache = RadixCache(
        block_size=BLOCK_SIZE, eviction_policy="lru", disable=False
    )

    # Data Generation
    print(
        f"Generating data: Batch={BATCH_SIZE}, Context={CONTEXT_LEN}, Shared={SHARED_PREFIX_LEN}..."
    )
    shared_prefix = [random.randint(0, 32000) for _ in range(SHARED_PREFIX_LEN)]

    requests = []
    for i in range(BATCH_SIZE):
        unique_part = [random.randint(0, 32000) for _ in range(UNIQUE_LEN)]
        tokens = shared_prefix + unique_part
        requests.append(Request(f"req_{i}", tokens, time.time()))

    # Mock Block Pool
    # We need enough blocks for at least one full request + batch * unique
    # 65536 / 16 = 4096 blocks per request with NO sharing.
    # With sharing: 4096 shared blocks + 8 * (1024/16 = 64) unique blocks.
    # ~4600 blocks total needed in cache.

    total_blocks_needed = (CONTEXT_LEN // BLOCK_SIZE) + (
        BATCH_SIZE * (UNIQUE_LEN // BLOCK_SIZE)
    )
    all_blocks = [KVCacheBlock(i) for i in range(total_blocks_needed + 1000)]
    block_allocator_ptr = 0

    def allocate_blocks(num):
        nonlocal block_allocator_ptr
        res = all_blocks[block_allocator_ptr : block_allocator_ptr + num]
        block_allocator_ptr += num
        return res

    print("Starting Benchmark Loop...")
    start_time = time.perf_counter()

    # 1. Processing Request 0 (Cold Start)
    req0 = requests[0]

    # Match
    t0 = time.perf_counter()
    matched_blocks, last_node = radix_cache.match_prefix(req0.token_ids)
    t1 = time.perf_counter()
    match_time = t1 - t0

    hit_tokens = len(matched_blocks) * BLOCK_SIZE
    needed_tokens = len(req0.token_ids) - hit_tokens
    needed_blocks = (needed_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Allocate & Insert
    new_blocks = allocate_blocks(needed_blocks)

    # Update Cache
    # We purposely insert the FULL sequence (matched + new) to update the tree
    # Actually RadixCache.insert takes the full sequence usually
    t2 = time.perf_counter()
    full_blocks = matched_blocks + new_blocks
    # Ensure alignment? RadixCache insert assumes full blocks for now in our port (wrapper logic usually handles this)
    # But wait, RadixInsert might expect the NEW part or the FULL part?
    # SGLang insert() doc says: "Insert works similarly to match, but creates nodes."
    # It inserts key->value.
    radix_cache.insert(req0.token_ids, full_blocks)
    t3 = time.perf_counter()
    insert_time = t3 - t2

    print(
        f"Req 0: Hit={hit_tokens}, Needed={needed_tokens}. Match={match_time*1000:.3f}ms, Insert={insert_time*1000:.3f}ms"
    )

    # 2. Processing Requests 1..7 (Hot Start)
    for i in range(1, BATCH_SIZE):
        req = requests[i]

        t0 = time.perf_counter()
        matched_blocks, last_node = radix_cache.match_prefix(req.token_ids)
        t1 = time.perf_counter()
        match_time = t1 - t0

        hit_tokens = len(matched_blocks) * BLOCK_SIZE
        needed_tokens = len(req.token_ids) - hit_tokens

        # We expect HIGH hit rate
        print(
            f"Req {i}: Hit={hit_tokens} (Expected ~{SHARED_PREFIX_LEN}), MatchTime={match_time*1000:.3f}ms"
        )

        needed_blocks = (needed_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        new_blocks = allocate_blocks(needed_blocks)
        full_blocks = matched_blocks + new_blocks

        t2 = time.perf_counter()
        radix_cache.insert(req.token_ids, full_blocks)
        t3 = time.perf_counter()
        insert_time = t3 - t2
        # print(f"  InsertTime={insert_time*1000:.3f}ms")

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Throughput calculation
    # "Decode Throughput" usually refers to generation.
    # But here we are benchmarking PREFIX CACHING THROUGHPUT (how fast we can process new requests).
    # Total Tokens processed = BATCH_SIZE * CONTEXT_LEN
    total_tokens = BATCH_SIZE * CONTEXT_LEN
    print(f"Total Time: {total_time:.4f}s")
    print(f"Throughput (Input Processing): {total_tokens / total_time:.2f} tokens/s")

    # --- Hash Baseline Comparison ---
    print("\n--- Running Hash-Based Baseline ---")

    class HashCache:
        def __init__(self, block_size):
            self.block_size = block_size
            self.cache = {}  # hash -> block

        def match_blocks(self, token_ids):
            # Simulate vLLM's sequential block hash check
            # We assume hash computation is fast (simulated)
            matched = []

            # We must iterate block by block
            num_blocks = len(token_ids) // self.block_size

            # In a real system, we compute hash based on content + prev hash.
            # Here we simulate the effect: if we have seen this sequence/prefix before, it matches.
            # But Hash Cache is strictly block-based.

            # To simulate realistically: we use tuple of tokens as hash
            current_prefix = []

            # Optimization: vLLM doesn't re-hash everything if we have prefix.
            # But for a new request (req 1..7), we have cold start relative to the Request object,
            # but the Cache has the blocks.

            for i in range(num_blocks):
                # block_tokens = tuple(token_ids[i*self.block_size : (i+1)*self.block_size])
                # hash_key = (prev_hash, block_tokens)

                # Simplified: just use slice as key
                block_slice = tuple(
                    token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                # We need to chain it to ensure it matches the specific prefix position
                # Actually vLLM uses chain hash.
                # So we can just use the full prefix up to this block as the key?
                # No, that's expensive. vLLM uses rolling hash or chain hash.
                # Key = (prev_block_hash, current_block_tokens)

                if i == 0:
                    prev = None
                else:
                    prev = matched[-1].block_id  # Use block_id as proxy for hash

                key = (prev, block_slice)

                if key in self.cache:
                    matched.append(self.cache[key])
                else:
                    break

            return matched

        def insert_blocks(self, token_ids, blocks):
            # blocks correspond to token_ids chunks
            num_blocks = len(blocks)
            for i in range(num_blocks):
                block_slice = tuple(
                    token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                if i == 0:
                    prev = None
                else:
                    prev = blocks[i - 1].block_id

                key = (prev, block_slice)
                self.cache[key] = blocks[i]

    hash_cache = HashCache(BLOCK_SIZE)

    # Reset block allocator for hash cache
    block_allocator_ptr = 0

    # Run Hash Benchmark
    start_time_h = time.perf_counter()

    # Req 0
    req0 = requests[0]
    matched = hash_cache.match_blocks(req0.token_ids)
    needed_tokens = len(req0.token_ids) - (len(matched) * BLOCK_SIZE)
    needed_blocks = (needed_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    new_blocks = allocate_blocks(needed_blocks)
    full_blocks = matched + new_blocks
    hash_cache.insert_blocks(req0.token_ids, full_blocks)

    # Req 1..7
    for i in range(1, BATCH_SIZE):
        req = requests[i]
        matched = hash_cache.match_blocks(req.token_ids)
        needed_tokens = len(req.token_ids) - (len(matched) * BLOCK_SIZE)
        needed_blocks = (needed_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        new_blocks = allocate_blocks(needed_blocks)
        full_blocks = matched + new_blocks
        hash_cache.insert_blocks(req.token_ids, full_blocks)

    end_time_h = time.perf_counter()
    total_time_h = end_time_h - start_time_h

    print(f"Total Time (Hash): {total_time_h:.4f}s")
    print(f"Throughput (Hash): {total_tokens / total_time_h:.2f} tokens/s")

    print(f"\nSpeedup Radix vs Hash: {total_time_h / total_time:.2f}x")


if __name__ == "__main__":
    run_benchmark()
