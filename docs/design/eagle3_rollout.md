# EAGLE3 Rollout Plan (DP=1, Single GPU)

1. 1-model EAGLE3 (single engine) on batch=1 and batch>1 with padded batches.
   - Verify greedy equivalence against baseline for fixed seeds.
   - Track acceptance rate, draft/verify latency, and tokens/sec.

2. 2-model EAGLE3 (separate draft engine) with identical batch shapes.
   - Confirm acceptance rate stays stable vs 1-model path.
   - Validate memory overhead and draft/verify overlap behavior.

3. Batch>1 parity sweep (still single GPU, DP=1).
   - Sweep batch sizes and max_num_batched_tokens.
   - Ensure stable cudagraph capture and no shape recompile churn.

4. Pipeline/PP expansion after single-GPU parity.
   - Add PP support once acceptance/throughput matches single-GPU numbers.
