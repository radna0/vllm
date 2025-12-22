#!/usr/bin/env python3
"""
Decode-Focused Throughput Benchmark for RadixCache vs HashCache.

Simulates:
- 65K context already cached (prefix)
- 8 parallel sequences decoding
- Measures DECODE operations ONLY (no prefill)
- Compares Radix tree-based eviction vs Hash-based LRU eviction

This benchmark focuses on the KV cache management overhead during decode phase.
"""

import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sys
import os
from types import ModuleType

# --- Setup sys.path and Mock Dependencies ---
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
KVCacheBlock = MockKVCacheBlock


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class BenchmarkConfig:
    # Context configuration
    CONTEXT_LEN: int = 65536  # 65K context length
    BLOCK_SIZE: int = 16  # vLLM block size

    # Batch configuration
    BATCH_SIZE: int = 8  # 8 parallel sequences
    SHARED_PREFIX_LEN: int = 64512  # ~98% shared (system prompt)

    # Decode simulation
    NUM_DECODE_STEPS: int = 1024  # Number of decode tokens per sequence

    # Pool configuration
    NUM_BLOCKS: int = 8192  # Total blocks in pool

    # Eviction simulation
    MEMORY_PRESSURE_RATIO: float = 0.9  # Fill pool to 90% before stress test


# =============================================================================
# HASH-BASED CACHE (TensorRT-LLM style)
# =============================================================================
class HashLinkedCache:
    """
    Simulates TensorRT-LLM's NextBlockMap approach with hash-chained lookups.
    Uses LRU for eviction.
    """

    def __init__(self, block_size: int, max_blocks: int):
        self.block_size = block_size
        self.max_blocks = max_blocks

        # BlockKey -> Block mapping (hash table)
        self.block_map: Dict[Tuple, KVCacheBlock] = {}

        # LRU tracking: list of block_ids, most recently used at end
        self.lru_list: List[int] = []

        # Block pool
        self.free_blocks: List[KVCacheBlock] = [
            KVCacheBlock(i) for i in range(max_blocks)
        ]
        self.used_blocks: Dict[int, KVCacheBlock] = {}

        # Metrics
        self.metrics = {
            "hit_count": 0,
            "miss_count": 0,
            "evict_count": 0,
        }

    def _compute_block_key(
        self, token_ids: List[int], block_idx: int, prev_block_id: Optional[int]
    ) -> Tuple:
        """Compute hash key for a block (chain hash like TensorRT-LLM)."""
        block_tokens = tuple(
            token_ids[block_idx * self.block_size : (block_idx + 1) * self.block_size]
        )
        return (prev_block_id, block_tokens)

    def _touch_lru(self, block_id: int):
        """Move block to end of LRU list (most recently used)."""
        if block_id in self.lru_list:
            self.lru_list.remove(block_id)
        self.lru_list.append(block_id)

    def _evict_one(self) -> KVCacheBlock:
        """Evict least recently used block."""
        if not self.lru_list:
            raise RuntimeError("No blocks to evict!")

        lru_block_id = self.lru_list.pop(0)  # Front = LRU
        block = self.used_blocks.pop(lru_block_id)

        # Remove from block_map (expensive O(n) scan, but simulating correctness)
        keys_to_remove = [
            k for k, v in self.block_map.items() if v.block_id == lru_block_id
        ]
        for k in keys_to_remove:
            del self.block_map[k]

        self.metrics["evict_count"] += 1
        return block

    def allocate_block(self) -> KVCacheBlock:
        """Allocate a new block, evicting if necessary."""
        if not self.free_blocks:
            block = self._evict_one()
        else:
            block = self.free_blocks.pop()

        self.used_blocks[block.block_id] = block
        return block

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[KVCacheBlock], int]:
        """Find cached prefix using hash chain lookup."""
        matched_blocks = []
        num_blocks = len(token_ids) // self.block_size
        prev_block_id = None

        for block_idx in range(num_blocks):
            key = self._compute_block_key(token_ids, block_idx, prev_block_id)

            if key in self.block_map:
                block = self.block_map[key]
                matched_blocks.append(block)
                self._touch_lru(block.block_id)
                prev_block_id = block.block_id
                self.metrics["hit_count"] += 1
            else:
                self.metrics["miss_count"] += 1
                break

        return matched_blocks, len(matched_blocks) * self.block_size

    def insert_blocks(self, token_ids: List[int], blocks: List[KVCacheBlock]):
        """Insert blocks into cache with hash keys."""
        prev_block_id = None
        num_blocks = len(blocks)

        for block_idx in range(num_blocks):
            key = self._compute_block_key(token_ids, block_idx, prev_block_id)
            self.block_map[key] = blocks[block_idx]
            self._touch_lru(blocks[block_idx].block_id)
            prev_block_id = blocks[block_idx].block_id


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================
def run_decode_benchmark(config: BenchmarkConfig):
    """
    Run decode-focused benchmark comparing RadixCache vs HashCache.

    Workflow:
    1. PREFILL phase: Insert 65K context for all 8 sequences (shared prefix)
    2. DECODE phase: Simulate 1024 decode steps per sequence
       - Each step: append 1 token, get new block if needed, evict if full
    3. Measure decode operations throughput
    """

    print("=" * 70)
    print("DECODE-FOCUSED THROUGHPUT BENCHMARK")
    print("=" * 70)
    print(f"Context Length: {config.CONTEXT_LEN:,} tokens")
    print(
        f"Shared Prefix: {config.SHARED_PREFIX_LEN:,} tokens ({100*config.SHARED_PREFIX_LEN/config.CONTEXT_LEN:.1f}%)"
    )
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Decode Steps: {config.NUM_DECODE_STEPS}")
    print(f"Block Size: {config.BLOCK_SIZE}")
    print(f"Total Blocks: {config.NUM_BLOCKS:,}")
    print()

    # Generate shared prefix and unique suffixes
    shared_prefix = list(range(1, config.SHARED_PREFIX_LEN + 1))
    unique_len = config.CONTEXT_LEN - config.SHARED_PREFIX_LEN

    sequences = []
    for i in range(config.BATCH_SIZE):
        unique_suffix = list(
            range(
                config.SHARED_PREFIX_LEN + i * 10000 + 1,
                config.SHARED_PREFIX_LEN + i * 10000 + unique_len + 1,
            )
        )
        sequences.append(shared_prefix + unique_suffix)

    # Block allocator for Radix
    radix_block_ptr = [0]

    def allocate_radix_block():
        block = KVCacheBlock(radix_block_ptr[0])
        radix_block_ptr[0] += 1
        return block

    # ==========================================================================
    # RADIX CACHE BENCHMARK
    # ==========================================================================
    print("-" * 70)
    print("RADIX CACHE (Tree-based, Leaf-first Eviction)")
    print("-" * 70)

    radix_cache = RadixCache(
        block_size=config.BLOCK_SIZE, eviction_policy="lru", disable=False
    )

    # PREFILL: Insert initial contexts
    prefill_start = time.perf_counter()
    radix_seq_blocks = []
    for seq_idx, token_ids in enumerate(sequences):
        # Match prefix
        matched_blocks, last_node = radix_cache.match_prefix(token_ids)

        # Allocate new blocks for unmatched portion
        num_matched_tokens = len(matched_blocks) * config.BLOCK_SIZE
        remaining_tokens = len(token_ids) - num_matched_tokens
        num_new_blocks = (remaining_tokens + config.BLOCK_SIZE - 1) // config.BLOCK_SIZE

        new_blocks = [allocate_radix_block() for _ in range(num_new_blocks)]

        # Insert into cache
        radix_cache.insert(token_ids, matched_blocks + new_blocks, priority=1)
        radix_seq_blocks.append(matched_blocks + new_blocks)

    prefill_end = time.perf_counter()
    radix_prefill_time = prefill_end - prefill_start
    print(f"Prefill Time: {radix_prefill_time*1000:.2f}ms")
    print(
        f"Prefill Throughput: {config.BATCH_SIZE * config.CONTEXT_LEN / radix_prefill_time / 1e6:.2f}M tok/s"
    )

    # DECODE: Simulate token-by-token decoding
    decode_start = time.perf_counter()
    decode_ops = 0

    for step in range(config.NUM_DECODE_STEPS):
        for seq_idx in range(config.BATCH_SIZE):
            # Append new token
            new_token = 50000 + step * config.BATCH_SIZE + seq_idx
            sequences[seq_idx].append(new_token)

            # Check if we need a new block
            current_len = len(sequences[seq_idx])
            if current_len % config.BLOCK_SIZE == 1:  # New block started
                new_block = allocate_radix_block()
                radix_seq_blocks[seq_idx].append(new_block)
                # Insert updated sequence
                radix_cache.insert(
                    sequences[seq_idx], radix_seq_blocks[seq_idx], priority=0
                )

            decode_ops += 1

    decode_end = time.perf_counter()
    radix_decode_time = decode_end - decode_start
    radix_decode_throughput = decode_ops / radix_decode_time

    print(f"Decode Time: {radix_decode_time*1000:.2f}ms")
    print(f"Decode Ops: {decode_ops:,}")
    print(f"Decode Throughput: {radix_decode_throughput:.2f} ops/s")
    print(f"Tokens Generated: {config.BATCH_SIZE * config.NUM_DECODE_STEPS:,}")
    print(
        f"Avg Latency per Token: {radix_decode_time / (config.BATCH_SIZE * config.NUM_DECODE_STEPS) * 1000:.4f}ms"
    )

    # Report metrics
    print(f"RadixCache Metrics: {radix_cache.metrics}")

    # ==========================================================================
    # HASH CACHE BENCHMARK
    # ==========================================================================
    print()
    print("-" * 70)
    print("HASH CACHE (TensorRT-LLM style, Block LRU Eviction)")
    print("-" * 70)

    # Reset sequences for fair comparison
    sequences = []
    for i in range(config.BATCH_SIZE):
        unique_suffix = list(
            range(
                config.SHARED_PREFIX_LEN + i * 10000 + 1,
                config.SHARED_PREFIX_LEN + i * 10000 + unique_len + 1,
            )
        )
        sequences.append(shared_prefix + unique_suffix)

    hash_cache = HashLinkedCache(
        block_size=config.BLOCK_SIZE, max_blocks=config.NUM_BLOCKS
    )

    # PREFILL
    prefill_start = time.perf_counter()
    hash_seq_blocks = []
    for seq_idx, token_ids in enumerate(sequences):
        matched_blocks, matched_tokens = hash_cache.match_prefix(token_ids)

        remaining_tokens = len(token_ids) - matched_tokens
        num_new_blocks = (remaining_tokens + config.BLOCK_SIZE - 1) // config.BLOCK_SIZE

        new_blocks = [hash_cache.allocate_block() for _ in range(num_new_blocks)]
        full_blocks = matched_blocks + new_blocks

        hash_cache.insert_blocks(token_ids, full_blocks)
        hash_seq_blocks.append(full_blocks)

    prefill_end = time.perf_counter()
    hash_prefill_time = prefill_end - prefill_start
    print(f"Prefill Time: {hash_prefill_time*1000:.2f}ms")
    print(
        f"Prefill Throughput: {config.BATCH_SIZE * config.CONTEXT_LEN / hash_prefill_time / 1e6:.2f}M tok/s"
    )

    # DECODE
    decode_start = time.perf_counter()
    decode_ops = 0

    for step in range(config.NUM_DECODE_STEPS):
        for seq_idx in range(config.BATCH_SIZE):
            new_token = 50000 + step * config.BATCH_SIZE + seq_idx
            sequences[seq_idx].append(new_token)

            current_len = len(sequences[seq_idx])
            if current_len % config.BLOCK_SIZE == 1:
                new_block = hash_cache.allocate_block()
                hash_seq_blocks[seq_idx].append(new_block)
                hash_cache.insert_blocks(sequences[seq_idx], hash_seq_blocks[seq_idx])

            decode_ops += 1

    decode_end = time.perf_counter()
    hash_decode_time = decode_end - decode_start
    hash_decode_throughput = decode_ops / hash_decode_time

    print(f"Decode Time: {hash_decode_time*1000:.2f}ms")
    print(f"Decode Ops: {decode_ops:,}")
    print(f"Decode Throughput: {hash_decode_throughput:.2f} ops/s")
    print(f"Tokens Generated: {config.BATCH_SIZE * config.NUM_DECODE_STEPS:,}")
    print(
        f"Avg Latency per Token: {hash_decode_time / (config.BATCH_SIZE * config.NUM_DECODE_STEPS) * 1000:.4f}ms"
    )

    print(f"HashCache Metrics: {hash_cache.metrics}")

    # ==========================================================================
    # COMPARISON SUMMARY
    # ==========================================================================
    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"{'Metric':<30} {'RadixCache':<20} {'HashCache':<20} {'Winner'}")
    print("-" * 70)

    # Prefill
    prefill_winner = "Radix" if radix_prefill_time < hash_prefill_time else "Hash"
    print(
        f"{'Prefill Time (ms)':<30} {radix_prefill_time*1000:<20.2f} {hash_prefill_time*1000:<20.2f} {prefill_winner}"
    )

    # Decode throughput
    decode_winner = (
        "Radix" if radix_decode_throughput > hash_decode_throughput else "Hash"
    )
    print(
        f"{'Decode Throughput (ops/s)':<30} {radix_decode_throughput:<20.2f} {hash_decode_throughput:<20.2f} {decode_winner}"
    )

    # Decode latency
    radix_latency = (
        radix_decode_time / (config.BATCH_SIZE * config.NUM_DECODE_STEPS) * 1000
    )
    hash_latency = (
        hash_decode_time / (config.BATCH_SIZE * config.NUM_DECODE_STEPS) * 1000
    )
    latency_winner = "Radix" if radix_latency < hash_latency else "Hash"
    print(
        f"{'Avg Decode Latency (ms/tok)':<30} {radix_latency:<20.4f} {hash_latency:<20.4f} {latency_winner}"
    )

    # Speedup
    speedup = hash_decode_time / radix_decode_time
    print()
    print(f"RadixCache Decode Speedup: {speedup:.2f}x")

    return {
        "radix": {
            "prefill_time_ms": radix_prefill_time * 1000,
            "decode_time_ms": radix_decode_time * 1000,
            "decode_throughput": radix_decode_throughput,
            "latency_ms_per_token": radix_latency,
            "metrics": radix_cache.metrics,
        },
        "hash": {
            "prefill_time_ms": hash_prefill_time * 1000,
            "decode_time_ms": hash_decode_time * 1000,
            "decode_throughput": hash_decode_throughput,
            "latency_ms_per_token": hash_latency,
            "metrics": hash_cache.metrics,
        },
        "speedup": speedup,
    }


if __name__ == "__main__":
    config = BenchmarkConfig()
    results = run_decode_benchmark(config)
