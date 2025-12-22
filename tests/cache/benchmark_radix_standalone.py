import sys
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Any, Type
from types import ModuleType

# --- Mocks Setup ---


# Define KVCacheBlock Mock
@dataclass
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    block_hash: Any = None
    is_null: bool = False

    def __repr__(self):
        return f"KVCacheBlock(id={self.block_id}, ref={self.ref_cnt})"


# Define Eviction Strategies Mock (Since evict_policy.py is simple, we could import it,
# but mocking guarantees no side deps)
from abc import ABC, abstractmethod


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node) -> Any:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node) -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node) -> Any:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node) -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    def get_priority(self, node) -> Any:
        return (node.priority, node.last_access_time)


# Inject Mocks into sys.modules
# We need to act BEFORE importing RadixCache
vllm = ModuleType("vllm")
sys.modules["vllm"] = vllm

vllm_v1 = ModuleType("vllm.v1")
sys.modules["vllm.v1"] = vllm_v1

vllm_v1_core = ModuleType("vllm.v1.core")
sys.modules["vllm.v1.core"] = vllm_v1_core

kv_cache_utils = ModuleType("vllm.v1.core.kv_cache_utils")
kv_cache_utils.KVCacheBlock = KVCacheBlock
sys.modules["vllm.v1.core.kv_cache_utils"] = kv_cache_utils

evict_policy = ModuleType("vllm.v1.core.evict_policy")
evict_policy.EvictionStrategy = EvictionStrategy
evict_policy.LRUStrategy = LRUStrategy
evict_policy.LFUStrategy = LFUStrategy
evict_policy.FIFOStrategy = FIFOStrategy
evict_policy.MRUStrategy = MRUStrategy
evict_policy.FILOStrategy = FILOStrategy
evict_policy.PriorityStrategy = PriorityStrategy
sys.modules["vllm.v1.core.evict_policy"] = evict_policy

# Now we can import RadixCache from the file on disk
# Since the file is at vllm/v1/core/radix_cache.py, and PYTHONPATH should include root
# We need to make sure python can find the FILE vllm/v1/core/radix_cache.py
# If we are in /workspace/aimo/DEV_VERSIONS/vllm_sglang/, 'vllm' is a package.
# But we MOCKED 'vllm'.
# If 'vllm' is in sys.modules, normal import mechanism stops looking at disk?
# Yes.
# So we cannot 'from vllm.v1.core.radix_cache import RadixCache' because vllm.v1.core is a Mock with no attributes.
# We must manually load the code from the file.

import importlib.util
import os

RADIX_CACHE_PATH = (
    "/workspace/aimo/DEV_VERSIONS/vllm_sglang/vllm/v1/core/radix_cache.py"
)
spec = importlib.util.spec_from_file_location(
    "vllm.v1.core.radix_cache", RADIX_CACHE_PATH
)
radix_cache_module = importlib.util.module_from_spec(spec)
sys.modules["vllm.v1.core.radix_cache"] = radix_cache_module
spec.loader.exec_module(radix_cache_module)


RadixCache = radix_cache_module.RadixCache

# --- Implementation of Managers using Mocks ---


class MockBlockPool:
    def __init__(self, num_blocks: int, enable_caching: bool):
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.enable_caching = enable_caching
        self.free_blocks_list = list(reversed(self.blocks))  # Stack behaviour
        self.cached_block_hash_to_block = {}

    def get_num_free_blocks(self):
        return len(self.free_blocks_list)

    def get_new_blocks(self, num_blocks: int) -> List[KVCacheBlock]:
        if num_blocks > len(self.free_blocks_list):
            raise ValueError("OOM")

        ret = []
        for _ in range(num_blocks):
            b = self.free_blocks_list.pop()
            b.ref_cnt += 1
            ret.append(b)
        return ret

    def free_blocks(self, blocks: List[KVCacheBlock]):
        for b in blocks:
            b.ref_cnt -= 1
            if b.ref_cnt <= 0:
                b.ref_cnt = 0
                if self.enable_caching and b.block_hash is not None:
                    # It is cached, so do NOT put in free list immediately?
                    # In vLLM, if enable_caching, free_queue contains cached blocks.
                    # So YES put in free list.
                    if b not in self.free_blocks_list:
                        self.free_blocks_list.append(b)
                elif not self.enable_caching:
                    # Always free
                    if b not in self.free_blocks_list:
                        self.free_blocks_list.append(b)
            # If ref_cnt > 0, it stays used.

    def touch(self, blocks_tuple: Tuple[List[KVCacheBlock]]):
        for blocks in blocks_tuple:
            for b in blocks:
                # If cached and in free list, resurrect it
                if b.ref_cnt == 0 and b in self.free_blocks_list:
                    self.free_blocks_list.remove(b)
                b.ref_cnt += 1

    def get_cached_block(self, block_hash) -> Optional[KVCacheBlock]:
        if not self.enable_caching:
            return None
        b = self.cached_block_hash_to_block.get(block_hash)
        if b and b.ref_cnt == 0:
            # It should be in free list. Resurrecting is done by caller via touch?
            # Or here? vLLM FullAttentionManager calls get_cached_block then touch.
            pass
        return b

    def cache_block(self, block_hash, block):
        self.cached_block_hash_to_block[block_hash] = block
        block.block_hash = block_hash


class MockFullAttentionManager:
    def __init__(self, block_size, block_pool):
        self.block_size = block_size
        self.block_pool = block_pool
        self.req_to_blocks = {}

    def find_longest_cache_hit(self, block_hashes) -> Tuple[List[KVCacheBlock]]:
        hit_blocks = []
        for h in block_hashes:
            b = self.block_pool.get_cached_block(h)
            if b:
                hit_blocks.append(b)
            else:
                break

        # Touch hits
        if hit_blocks:
            self.block_pool.touch((hit_blocks,))

        return (hit_blocks,)

    def allocate_new_blocks(self, request_id, num_tokens):
        num_needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        current_blocks = self.req_to_blocks.get(request_id, [])
        new_needed = num_needed_blocks - len(current_blocks)
        if new_needed <= 0:
            return []

        new_blocks = self.block_pool.get_new_blocks(new_needed)
        if request_id not in self.req_to_blocks:
            self.req_to_blocks[request_id] = []
        self.req_to_blocks[request_id].extend(new_blocks)
        return new_blocks

    def cache_blocks(self, req, num_tokens):
        # We assume req has block_hashes updated
        hashes = req.block_hashes
        blocks = self.req_to_blocks.get(req.request_id, [])

        # Cache full blocks
        num_full = num_tokens // self.block_size
        for i in range(num_full):
            if i < len(hashes) and i < len(blocks):
                self.block_pool.cache_block(hashes[i], blocks[i])

    def free(self, request_id):
        blocks = self.req_to_blocks.pop(request_id, [])
        self.block_pool.free_blocks(blocks)


class MockRadixFullAttentionManager(MockFullAttentionManager):
    def __init__(self, block_size, block_pool, eviction_policy="lru"):
        super().__init__(block_size, block_pool)
        self.radix_cache = RadixCache(
            block_size=block_size, eviction_policy=eviction_policy
        )
        self.num_cached_block = {}

    def find_longest_cache_hit(self, token_ids) -> Tuple[List[KVCacheBlock]]:
        blocks, _ = self.radix_cache.match_prefix(token_ids)
        if blocks:
            self.block_pool.touch((blocks,))
        return (blocks,)

    def cache_blocks(self, req, num_tokens):
        num_cached = self.num_cached_block.get(req.request_id, 0)
        num_full = num_tokens // self.block_size

        if num_cached >= num_full:
            return

        blocks = self.req_to_blocks[req.request_id]
        blocks_to_cache = blocks[:num_full]
        token_ids = req.token_ids[: num_full * self.block_size]

        self.radix_cache.insert(token_ids, blocks_to_cache)
        # Radix ownership
        self.block_pool.touch((blocks_to_cache,))

        self.num_cached_block[req.request_id] = num_full

    def free(self, request_id):
        # Remove request
        blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_block.pop(request_id, None)

        self.block_pool.free_blocks(blocks)

    def allocate_new_blocks(self, request_id, num_tokens):
        try:
            return super().allocate_new_blocks(request_id, num_tokens)
        except ValueError:
            # Evict
            needed = ((num_tokens + self.block_size - 1) // self.block_size) - len(
                self.req_to_blocks.get(request_id, [])
            )
            if needed > 0:
                evicted = self.radix_cache.evict(needed * self.block_size)
                self.block_pool.free_blocks(evicted)  # Drop Radix ref
                return super().allocate_new_blocks(request_id, num_tokens)
        return []


# --- Benchmark Driver ---


@dataclass
class MockRequest:
    request_id: str
    token_ids: List[int]
    block_hashes: List[Any] = field(default_factory=list)


def generate_hashes(tokens, bs):
    hashes = []
    for i in range(0, len(tokens), bs):
        chunk = tuple(tokens[i : i + bs])
        if len(chunk) == bs:
            hashes.append(hash(chunk))
    return hashes


def run():
    print("Running Standalone Benchmark...")
    block_size = 16
    num_blocks = 100

    # Hash
    pool_hash = MockBlockPool(num_blocks, enable_caching=True)
    mgr_hash = MockFullAttentionManager(block_size, pool_hash)

    # Radix
    pool_radix = MockBlockPool(num_blocks, enable_caching=False)
    mgr_radix = MockRadixFullAttentionManager(block_size, pool_radix)

    # Trace
    # System prompt
    sys_prompt = [random.randint(0, 1000) for _ in range(64)]

    requests_data = []
    users = [f"u{i}" for i in range(5)]
    history = {u: list(sys_prompt) for u in users}

    for _ in range(200):
        u = random.choice(users)
        h = history[u]
        new_in = [random.randint(0, 1000) for _ in range(16)]
        full = h + new_in
        req = MockRequest(f"{u}_{len(h)}", full, generate_hashes(full, block_size))
        requests_data.append(req)
        history[u] = full

    # Run Hash
    t0 = time.time()
    hits = 0
    tot = 0
    for req in requests_data:
        hit_tup = mgr_hash.find_longest_cache_hit(req.block_hashes)
        hit_blocks = hit_tup[0]
        hits += len(hit_blocks) * block_size
        tot += len(req.token_ids)

        try:
            mgr_hash.allocate_new_blocks(req.request_id, len(req.token_ids))
        except ValueError:
            continue  # OOM

        mgr_hash.cache_blocks(req, len(req.token_ids))
        mgr_hash.free(req.request_id)
    print(f"Hash: {time.time()-t0:.4f}s, HitRate: {hits/tot:.2%}")

    # Run Radix
    t0 = time.time()
    hits = 0
    tot = 0
    evictions = 0
    for req in requests_data:
        # print(f"DEBUG: Req {req.request_id} TokenLen {len(req.token_ids)}")
        hit_tup = mgr_radix.find_longest_cache_hit(req.token_ids)
        hit_blocks = hit_tup[0]
        hits += len(hit_blocks) * block_size
        tot += len(req.token_ids)

        # if len(hit_blocks) > 0:
        #     print(f"DEBUG: Hit {len(hit_blocks)} blocks for {req.request_id}")
        # else:
        #     # print(f"DEBUG: Miss for {req.request_id}")
        #     pass

        try:
            new = mgr_radix.allocate_new_blocks(req.request_id, len(req.token_ids))
        except ValueError:
            continue

        mgr_radix.cache_blocks(req, len(req.token_ids))
        mgr_radix.free(req.request_id)
    print(f"Radix: {time.time()-t0:.4f}s, HitRate: {hits/tot:.2%}")


if __name__ == "__main__":
    run()
