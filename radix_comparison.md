# RadixCache Comparison: SGLang vs vLLM Port

## Overview
This document outlines the differences between the original SGLang `RadixCache` and the vLLM port, identifying missing optimizations to be implemented.

## Key Differences

| Feature | SGLang Implementation | vLLM Port (Current) | Impact |
| :--- | :--- | :--- | :--- |
| **Granularity** | Token-level (with page awareness) | Block-level (16 tokens) | Block-level is coarser but aligns with vLLM's memory model. SGLang handles token-level splits more gracefully. |
| **Eviction** | Lazy (via `evict` method) + Priority Heap | Lazy (via `evict` method) + Priority Heap | Similar, but SGLang tracks `evictable_size` more granularly. |
| **Node Merging** | No explicit node merging observed in `RadixCache.py` (only split) | No explicit node merging | Potential fragmentation if many small insertions occur. |
| **Events** | `enable_kv_cache_events` for lazy hash computation & storage handling | None | SGLang lazy hash computation might save CPU cycles. vLLM currently ignores this. |
| **Bigram Keys** | Supports Bigram keys (for Eagle) | No explicit Bigram support | Critical for Eagle speculative decoding accuracy. |
| **Locking** | `inc_lock_ref` / `dec_lock_ref` propagates to root | Same | Correctness verified. |
| **Memory Pool** | Directly interacts with `token_to_kv_pool_allocator` | Abstracts via `BlockPool` | vLLM's `BlockPool` is more rigid. SGLang manually frees indices. |

## Missed Optimizations / TODOs

1. **Bigram Key Support**:
   - SGLang uses `convert_to_bigram_key` for Eagle. vLLM port treats all keys as standard token sequences.
   - **Action**: Add `maybe_bigram_convert` logic to vLLM port if Eagle support is required (User objective mentions Eagle).

2. **Lazy Hash Computation**:
   - SGLang computes leaf hashes lazily only when recording store events.
   - **Action**: In vLLM, `BlockHashList` is computed eagerly in the Model Runner. We might not need this for RadixCache itself, but it helps with "Storage" events if we move to a disaggregated setup. For now, low priority.

3. **Leaf Collection Optimization**:
   - SGLang's `_collect_leaves` iterates recursively. vLLM port uses a stack-based iterative approach (good).

4. **Split Logic & Block Alignment**:
   - SGLang splits `kv_indices` at exact token boundaries.
   - vLLM port currently assumes splits happen at block boundaries or fails.
   - **Critical Problem**: If a prefix match ends in the middle of a block, vLLM port either:
     - Fails the match (miss).
     - Or needs to duplicate the block? (Can't split `KVCacheBlock`).
   - **Optimization**: "Unaligned Split Handling". If a split is needed in the middle of a block:
     - Treat the block as shared? (Ref count ++).
     - But vLLM `KVCacheBlock` doesn't support sub-block addressing easily in the manager.
     - **Resolution**: Strict Block Alignment is likely necessary for vLLM's PagedAttention unless we modify PagedAttention to accept "start_offset" in blocks (complex).
     - **Mitigation**: Ensure `insert` always inserts full blocks. `match_prefix` will only match full blocks.

5. **Throughput Optimization**:
   - `match_prefix` in SGLang returns `device_indices` (Tensor). vLLM port returns `List[KVCacheBlock]`.
   - Creating distinct lists of blocks for every request might be slow for 65k context.
   - **Optimization**: Use `tuple` or lightweight structures for cache hits? Or keep `List[KVCacheBlock]` but ensure `block_pool.touch` is batched.

## Proposed benchmarking Strategy
- **Scenario**: 8 concurrent requests.
- **Context**: 65k tokens each.
- **Overlap**: All 8 requests share the first 64k tokens (System Prompt + Few Shot). only last 1k differs.
- **Expectation**:
  - Request 1: Full Prefill (65k).
  - Request 2-8: 64k Hit (Radix), 1k Prefill.
  - **throughput**: Should be massive (approximating 8x speedup for prefill part).

## Action Plan
1. Updates `RadixCache` to support Bigram keys (for Eagle future-proofing).
2. Optimize `match_prefix` to be faster (avoid unnecessary python object creation if possible).
3. Implement `benchmark_radix_throughput.py`.
