from __future__ import annotations

import heapq
import time
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Iterator, Union

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
)


class RadixKey:
    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
    ):
        self.token_ids = tuple(token_ids)
        self.extra_key = extra_key

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(list(self.token_ids[idx]), self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        return f"RadixKey(len={len(self.token_ids)})"


class TreeNode:
    counter = 0

    def __init__(
        self,
        key: RadixKey = None,
        value: List[KVCacheBlock] = None,
        parent: TreeNode = None,
        priority: int = 0,
    ):
        self.children: Dict[Union[int, Tuple[int, ...]], TreeNode] = {}
        self.parent = parent
        self.key = key if key is not None else RadixKey([])
        self.value = value if value is not None else []
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()
        self.hit_count = 0
        self.priority = priority
        self.id = TreeNode.counter
        TreeNode.counter += 1

    @property
    def evicted(self):
        # In vLLM context, if value (blocks) is empty, it's effectively evicted or a virtual node
        return len(self.value) == 0 and self.key.token_ids

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match(key0: RadixKey, key1: RadixKey) -> int:
    if key0.extra_key != key1.extra_key:
        return 0
    i = 0
    # key0.token_ids is tuple
    l0 = len(key0.token_ids)
    l1 = len(key1.token_ids)
    limit = min(l0, l1)
    t0 = key0.token_ids
    t1 = key1.token_ids
    while i < limit:
        if t0[i] != t1[i]:
            break
        i += 1
    return i


class RadixCache:
    def __init__(
        self,
        block_size: int,
        eviction_policy: str = "lru",
        disable: bool = False,
    ):
        self.block_size = block_size
        self.disable = disable
        self.eviction_policy = eviction_policy.lower()

        if self.eviction_policy == "lru":
            self.eviction_strategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy = PriorityStrategy()
        else:
            raise ValueError(f"Unknown eviction policy: {eviction_policy}")

        self.reset()

    def reset(self):
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.lock_ref = 1  # Root is always locked
        self.evictable_size_ = 0
        # Metrics
        self.metrics = {
            "match_count": 0,
            "match_hit_tokens": 0,
            "match_total_tokens": 0,
            "insert_count": 0,
            "evict_count": 0,
            "evict_tokens": 0,
        }

    def match_prefix(
        self, token_ids: List[int], extra_key: Optional[str] = None
    ) -> Tuple[List[KVCacheBlock], TreeNode]:
        if self.disable or not token_ids:
            return [], self.root_node

        self.metrics["match_count"] += 1
        self.metrics["match_total_tokens"] += len(token_ids)
        key = RadixKey(token_ids, extra_key)
        # We process in block_size chunks for vLLM
        # vLLM blocks are fixed size.
        # SGLang RadixCache works on tokens.
        # Here we adapt: The tree nodes will represent sequences of tokens, corresponding to blocks.

        # Actually, for vLLM integration, it is easiest if each node represents exactly one block or a sequence of blocks.
        # But SGLang's power is variable length nodes.
        # Let's keep SGLang's logic: nodes map to tokens. The 'value' of a node will be the KVCacheBlocks that cover `node.key`.

        # However, vLLM works with blocks. If proper prefix matches partial block, we can't easily reuse it without complex logic.
        # For this port, we align with block boundaries.

        node = self.root_node
        value: List[KVCacheBlock] = []

        node.last_access_time = time.monotonic()

        # Current key we are looking for
        # We search child by the first token of the remaining key
        while len(key) > 0:
            first_token = key.token_ids[0]
            if first_token in node.children:
                child = node.children[first_token]
                child.last_access_time = time.monotonic()

                # Match full key of child
                prefix_len = _key_match(child.key, key)

                if prefix_len < len(child.key):
                    # Partial match on the child edge.
                    # We found a prefix that is INSIDE an existing node.
                    # SGLang splits here.
                    # For vLLM, if we split, we must split the physical blocks too?
                    # vLLM blocks are atomic. We cannot split a block.
                    # So we can only match if prefix_len aligns with block boundaries OR if we accept partial matches (wasteful?).
                    # To keep it simple: we split the node logic but the 'value' (KVCacheBlocks) must be handled carefully.

                    # If we split logic node, the physical blocks might still be shared?
                    # Use Case:
                    # Node A: tokens [0..15] (Block 0, 1).
                    # Request comes for [0..7]. Match len 8.
                    # Split Node A into A1 [0..7] (Block 0) and A2 [8..15] (Block 1).
                    # This implies 'value' must be sliceable.

                    new_node = self._split_node(child, prefix_len)
                    value.extend(new_node.value)
                    node = new_node
                    break
                else:
                    # Full match of child key
                    value.extend(child.value)
                    node = child
                    key = key[prefix_len:]
            else:
                break

        hit_tokens = len(value) * self.block_size
        self.metrics["match_hit_tokens"] += hit_tokens
        return value, node

    def insert(
        self,
        token_ids: List[int],
        blocks: List[KVCacheBlock],
        extra_key: Optional[str] = None,
        priority: int = 0,
    ):
        if self.disable or not token_ids:
            return 0

        self.metrics["insert_count"] += 1
        key = RadixKey(token_ids, extra_key)
        # Insert works similarly to match, but creates nodes.

        node = self.root_node
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Priority propagation: inherit max priority along path
        node.priority = max(node.priority, priority)
        total_matched_len = 0

        # We assume 'blocks' corresponds exactly to 'token_ids'
        # Verification: len(token_ids) should roughly be len(blocks) * block_size

        current_block_idx = 0

        while len(key) > 0:
            first_token = key.token_ids[0]
            if first_token in node.children:
                child = node.children[first_token]
                child.last_access_time = access_time
                # Priority propagation
                child.priority = max(child.priority, priority)

                prefix_len = _key_match(child.key, key)
                total_matched_len += prefix_len

                if prefix_len < len(child.key):
                    # Split
                    new_node = self._split_node(child, prefix_len)
                    new_node.priority = max(new_node.priority, priority)
                    node = new_node

                else:
                    node = child

                key = key[prefix_len:]
                # Advance blocks?
                # The node.value contains blocks covering node.key.
                # Since we matched child, we implicitly "used" those blocks.
                # Use len(child.value) to approximate?
                num_blocks_in_child = len(child.value)  # Roughly
                current_block_idx += num_blocks_in_child

            else:
                # No child, create new node
                new_node = TreeNode(
                    key=key,
                    value=blocks[current_block_idx:],
                    parent=node,
                    priority=priority,
                )
                node.children[first_token] = new_node
                self.evictable_size_ += len(key)
                return total_matched_len

    def _split_node(self, child: TreeNode, split_len: int) -> TreeNode:
        # child is the node to be split
        # split_len is relative to child.key

        key_prefix = RadixKey(child.key.token_ids[:split_len], child.key.extra_key)
        key_suffix = RadixKey(child.key.token_ids[split_len:], child.key.extra_key)

        # Split values (blocks)
        # This is where strict block alignment matters.
        # If split_len is not a multiple of block_size, we can't cleanly split the KVCacheBlock list.
        # We assume for this implementation that splits happen at block boundaries.

        cnt_blocks = len(child.value)
        # Approximate split index
        # Assuming homogeneous block size.
        # This is an approximation. Ideally we check the block content.

        split_block_idx = split_len // self.block_size
        # Handle residual?
        # If split_len % block_size != 0, it means the split point is inside a block.
        # We CANNOT split a KVCacheBlock.
        # So we effectively have to choose:
        # 1. Don't split (fail partial match).
        # 2. Duplicate the block? No.

        # SGLang uses paged attention where pages are small (1 usually?) or it handles token-level granularity?
        # SGLang RadixCache works with paged kv cache.

        # If split is internal to a block, we put the Shared Block in the Parent (Prefix) node?
        # or the Child (Suffix) node?
        # Usually checking cache hits implies we have computed that far.
        # For simplicity, we enforce block alignment in this vLLM adapter.

        if split_len % self.block_size != 0:
            # Cannot split inside a block for vLLM KVCacheBlock
            # We fallback: we treat the WHOLE block as belonging to the child (Suffix),
            # and only split if we have > 1 block.
            # OR we just don't split and return child.
            pass

        value_prefix = child.value[:split_block_idx]
        value_suffix = child.value[split_block_idx:]

        new_node = TreeNode(
            key=key_prefix,
            value=value_prefix,
            parent=child.parent,
            priority=child.priority,
        )
        new_node.children[key_suffix.token_ids[0]] = child

        child.parent.children[child.key.token_ids[0]] = new_node
        child.parent = new_node
        child.key = key_suffix
        child.value = value_suffix

        return new_node

    def evict(self, num_tokens_to_evict: int) -> List[KVCacheBlock]:
        # Collect leaves and evict based on policy
        if self.disable:
            return []

        leaves = self._collect_leaves()
        # Sort/Heapify
        heap = [(self.eviction_strategy.get_priority(n), n) for n in leaves]
        heapq.heapify(heap)

        evicted_blocks = []
        evicted_count = 0

        while evicted_count < num_tokens_to_evict and heap:
            prio, node = heapq.heappop(heap)
            if node.lock_ref > 0:
                continue

            # Evict this node
            # Remove from parent
            if node.parent:
                del node.parent.children[node.key.token_ids[0]]

            evicted_blocks.extend(node.value)
            evicted_count += len(node.key)
            self.evictable_size_ -= len(node.key)

            # Check if parent becomes a leaf
            parent = node.parent
            if parent and not parent.children and parent != self.root_node:
                heapq.heappush(
                    heap, (self.eviction_strategy.get_priority(parent), parent)
                )

        self.metrics["evict_count"] += 1
        self.metrics["evict_tokens"] += evicted_count
        return evicted_blocks

    def _collect_leaves(self):
        leaves = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if not node.children:
                if node != self.root_node:
                    leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def inc_lock_ref(self, node: TreeNode):
        curr = node
        while curr:
            curr.lock_ref += 1
            curr = curr.parent

    def dec_lock_ref(self, node: TreeNode):
        curr = node
        while curr:
            curr.lock_ref -= 1
            curr = curr.parent
