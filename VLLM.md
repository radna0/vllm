# vLLM Local Notes

## EAGLE3 speculative decoding

- **Primary runtime path:** `vllm/v1/worker/gpu_model_runner.py` with
  `vllm/v1/spec_decode/eagle.py` (`EagleProposer`).
- **Legacy path:** `vllm/v1/worker/gpu/model_runner.py` with
  `vllm/v1/worker/gpu/spec_decode/eagle.py` (`EagleSpeculator`).
- **Aux hidden states:** legacy path now supports Eagle3 aux-layer outputs.
- **Tree decode:** Tree attention runs in `EagleProposer.propose_tree`; M-RoPE is
  still unsupported for tree drafting.
- **Dynamic tree (TRTLLM):** when `VLLM_EAGLE_DYNAMIC_TREE=1` and TRTLLM
  attention is used, drafting only runs the selected dynamic nodes and supplies
  per-step packed masks/position offsets to TRTLLM.
- **Tree slot mapping:** Draft slot mapping is built with
  `_eagle_tree_slot_mapping_kernel` into a reusable buffer.
- **SpecTreeManager:** static tree precompute lives in
  `vllm/v1/spec_decode/spec_tree_manager.py` (parent/child indices, packed mask).
- **Tree attn mask:** `TreeAttentionMetadata` now carries packed mask + position
  offsets (draft slices for tree levels are precomputed).

### Local env toggles

- `VLLM_EAGLE_AUTO_PADDED=1` auto-selects padded vs unpadded draft batches.
- `VLLM_EAGLE_CUDAGRAPH_ALLOW_RANDOM=1` allows non-greedy draft sampling with
  CUDA graphs (if shapes are capture-safe).
- `VLLM_EAGLE_FAST_LOGITS=1` enables fast-logits sampling (Gumbel, no softmax);
  now supports top-k/top-p by masking logits before sampling.
- `VLLM_EAGLE_GPU_VERIFY=1` enables GPU verification path for unpadded batches.
  Unpadded GPU verify now also rewinds slot mappings for rejected tokens.
  Rejection sampling now has a greedy fast path that skips target softmax.
- `VLLM_EAGLE_CUDA_INDICES=1` builds spec-decode indices on GPU via C++ kernel
  (replaces NumPy index construction in `_calc_spec_decode_metadata`).
- `VLLM_EAGLE_CUDA_REWIND=1` uses a CUDA kernel for slot-mapping rewind
  (replaces the Triton `_eagle_rewind_slot_mapping_kernel`).
- `VLLM_EAGLE_CUDA_DRAFT=1` uses CUDA kernels for draft slot mapping and
  draft-state updates (replaces Triton draft kernels in `EagleProposer`).
- `VLLM_EAGLE_CUDA_SAMPLE=1` routes draft sampling through CUDA ops
  (top‑k/top‑p + Gumbel) and enables capture‑safe draft loops.
- `VLLM_EAGLE_CUDA_KV_REWIND=1` zeros rejected draft KV slots on GPU before
  slot-mapping rewind (keeps KV cleanup off the CPU path).
- `VLLM_EAGLE_CUDA_KV_COMPACT=1` compacts accepted tree draft KV slots on GPU
  and rewrites slot mappings (required for non-linear tree decode).
- `VLLM_EAGLE_OPT_MODE=baseline|optimized|goal` switches between legacy and
  optimized feature sets for A/B comparisons.
- `VLLM_EAGLE_DYNAMIC_TREE_KERNELS=1` enables the CUDA dynamic-tree kernels
  (second‑top‑k path extraction) for TRT‑LLM parity; set `0` to keep the
  legacy Torch top‑k path.
- Dynamic tree CUDA helpers now include a packed mask builder and small top‑k
  selector (enabled when `VLLM_EAGLE_OPT_MODE` is not `baseline`).
- Large‑vocab top‑k can be routed through the `eagle_topk_logits_custom` CUDA op
  when `VLLM_EAGLE_OPT_MODE` is not `baseline` (custom kernel for top‑k + logsumexp).
- Rebuild `vllm._C` to enable the CUDA indices/rewind/draft/KV ops.
- Prompt tokens (and accepted outputs) are mirrored on GPU for spec‑decode
  batches to avoid CPU fallback when computing backup tokens.
- Spec‑decode accepted tokens are now GPU‑packed (flat buffer + cu offsets)
  to reduce CPU parse overhead in sync mode; CUDA op used when available,
  Triton fallback otherwise.
  Rebuild `vllm._C` to enable the CUDA packing op.

## Code paths (DP=1, single GPU)

- **Primary v1 runtime:** `vllm/v1/worker/gpu_model_runner.py` →
  `propose_draft_token_ids()` → `EagleProposer.prepare_inputs*()` →
  `EagleProposer.propose()` → `EagleProposer.propose_tree()` (tree attn).
- **Legacy runtime:** `vllm/v1/worker/gpu/model_runner.py` →
  `vllm/v1/worker/gpu/spec_decode/eagle.py` (EagleSpeculator).
- **Tree attention builder:** `vllm/v1/attention/backends/tree_attn.py`
  (`TreeAttentionMetadataBuilder.build_for_drafting`).

## Required tests (DP=1, single GPU)

- **Syntax/compile:**  
  `python -m py_compile vllm/v1/spec_decode/eagle.py vllm/v1/spec_decode/spec_tree_manager.py vllm/v1/attention/backends/tree_attn.py vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/gpu/model_runner.py vllm/v1/worker/gpu/spec_decode/eagle.py vllm/envs.py`
- **Tree precompute sanity:** run a short script that instantiates
  `SpecTreeManager` with the active `speculative_token_tree` and verifies
  `spec_dec_mask_matrix` shape and `draft_packed_masks` sizes.
- **SpecTreeManager test:** `pytest -q tests/v1/spec_decode/test_spec_tree_manager.py`
- **Spec decode indices kernel:** `pytest -q tests/v1/spec_decode/test_spec_decode_indices.py`
- **EAGLE3 tree parity (gated):** `VLLM_TEST_EAGLE_TREE=1 pytest -q tests/v1/spec_decode/test_eagle_tree_decode.py`
- **Spec decode benchmark:** run the local GPT‑OSS harness in
  `/workspace/aimo/jupyter` and compare baseline vs EAGLE3 acceptance + tokens/sec.
- **Tree decode microbench:** `python benchmarks/benchmark_eagle_tree_decode.py`
  with `--attention-backend TREE_ATTN` and a non-linear `speculative_token_tree`.
- **Greedy equivalence:** same prompts/seeds with spec decode on vs off (batch=1)
  and compare token outputs for equality.
- **Tree decode acceptance:** enable tree attention and confirm acceptance rate
  (no regressions vs non‑tree) on a fixed prompt set.
- **CUDAGraph parity:** run with and without `VLLM_EAGLE_AUTO_PADDED=1` and verify
  outputs and acceptance do not change.
- **GPU verify parity:** run with `VLLM_EAGLE_GPU_VERIFY=1` (unpadded) and confirm
  outputs + acceptance match the CPU path.
- **CUDA draft kernels parity:** run with `VLLM_EAGLE_CUDA_DRAFT=1` and confirm
  outputs + acceptance match the Triton path.
- **CUDA KV rewind parity:** run with `VLLM_EAGLE_CUDA_KV_REWIND=1` and confirm
  outputs + acceptance match the non-KV-rewind path.
- **CUDA KV compaction parity:** run with `VLLM_EAGLE_CUDA_KV_COMPACT=1` on a
  non-linear tree and confirm outputs + acceptance match the non-compacted path.

## TRTLLM attention + MoE validation (SM100/SM120, DP=1)

- **Attention kernels:** `python -m pytest tests/kernels/attention/test_flashinfer_trtllm_attention.py`
- **TRTLLM decode/prefill TPS:** `python benchmarks/kernels/benchmark_trtllm_decode_attention.py` and
  `python benchmarks/kernels/benchmark_trtllm_prefill_attention.py` (record TPS + KV bytes/token).
- **Attention sinks + sliding window:** run GPT-OSS with TRTLLM attention and verify
  outputs match a BF16 baseline on a fixed prompt set (batch=1 and batch>1).
- **SM120 TRTLLM-GEN enablement (FlashInfer):** set `FLASHINFER_CUBIN_DIR` (or
  `FLASHINFER_REPOSITORY`) to a cubin repo that contains SM120 TRTLLM-GEN kernels
  and export `VLLM_FLASHINFER_TRTLLM_SM120_ALLOW=1` if the meta headers do not
  advertise SM120; run GPT-OSS with sinks on SM120 and `--attention-backend auto`,
  confirm decode picks TRTLLM-GEN without `TllmGenFmhaRunner` errors.
- **SM120 FlashInfer MoE autotune:** clear
  `~/.cache/flashinfer/*/cached_ops/fused_moe_120` after header changes and run a
  short GPT-OSS pass; confirm no `Unsupported tile` warnings during autotune.
- **TRTLLM bindings auto-detect:** with a local TensorRT-LLM repo, verify
  `vllm.utils.trtllm.get_trtllm_thop()` returns a module and TRTLLM backend
  becomes eligible (no "TRTLLM bindings not available" warning).
- **TRTLLM spec-decoding tree masks:** run EAGLE3 with TRTLLM attention (SM120)
  and compare outputs vs FlashAttention/tree-attn baselines; verify tree mask
  shapes match `SpecTreeManager` (packed mask + position offsets).
- **TRTLLM dynamic tree masks:** run `VLLM_EAGLE_DYNAMIC_TREE=1` with TRTLLM
  attention; confirm packed mask/offsets are updated per step and outputs match
  static tree decode (acceptance-only difference expected).
- **Custom spec-decoding masks:** if supplying dynamic masks, ensure
  `spec_decoding_position_offsets`/`spec_decoding_packed_mask` are int32 GPU
  tensors with batch-aligned shapes, and validate TRTLLM outputs vs BF16.
- **MoE custom ops (FP8/NVFP4):** run a small forward pass with
  `VLLM_USE_TRTLLM_MOE=1` and compare outputs against the FlashInfer path
  (same inputs, tolerance vs BF16).
- **EPLB routing:** enable EPLB and confirm topk routing matches non-EPLB outputs
  within tolerance for the same inputs (no all2all).

## NVFP4 KV cache (GPT-OSS, DP=1)

- **Backend selection (TRTLLM preferred):** `pytest -q tests/v1/attention/test_gpt_oss_nvfp4_backend_selection.py`
- **TRTLLM NVFP4 guard (TODO):** add a GPT‑OSS test that forces TRTLLM NVFP4
  (full + sliding/chunked) and asserts `attention_supports_nvfp4_output` is
  True; expect a NotImplementedError if TRTLLM reports no NVFP4 kernels.
- **Fused RoPE + cache (TRITON):** `pytest -q tests/v1/attention/test_nvfp4_fused_rope_cache.py`
- **Chunked prefill + sliding window (TODO):** add a GPT-OSS test that runs
  chunked prefill with sliding window layers enabled and asserts NVFP4 fused
  cache path + NVFP4 read kernel are selected (no dequant fallback).
- **KV cache scale ingest (TODO):** add a GPT-OSS test that loads a checkpoint
  with `kv_cache_scaling_factor` or `{k,v}_cache_scaling_factor` and verifies
  those scales are consumed and runtime calibration is disabled.
- **Slot-mapping padding (TODO):** add a unit test that pads slot_mapping while
  keeping positions shorter and verifies fused NVFP4 cache write succeeds.
- **Nibble ordering:** `pytest -q tests/kernels/quantization/test_nvfp4_nibble_order.py`
- **KV cache pack/unpack order:** `pytest -q tests/kernels/quantization/test_nvfp4_kv_cache_unpack.py`
- **TRTLLM attention kernel parity:** `pytest -q tests/kernels/attention/test_flashinfer_trtllm_attention.py`

## Remaining TRT‑LLM parity tasks (100% target)

1. **GPU‑only verification + KV rewind:**  
   wire rejection sampling outputs on GPU into the EAGLE3 acceptance flow and
   apply slot‑mapping rewind for rejected tokens (CUDA op available); optional
   KV cache rewind now runs via `VLLM_EAGLE_CUDA_KV_REWIND`.  
   Files: `vllm/v1/worker/gpu/spec_decode/rejection_sample.py`,
   `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/spec_decode/eagle.py`.
2. **Fast‑logits / logit‑free draft path:**  
   avoid draft‑logits transfers unless required (TRT‑LLM fast‑logits parity).  
   Status: draft sampling path now supports top‑k/top‑p without softmax; still
   need to wire optional draft‑prob materialization (if we decide to support it).  
   Files: `vllm/v1/spec_decode/eagle.py`, `vllm/v1/worker/gpu_model_runner.py`.
3. **Tree‑decode microbench:**  
   added `benchmarks/benchmark_eagle_tree_decode.py` (baseline vs tree).  
   Files: `benchmarks/benchmark_eagle_tree_decode.py`, `VLLM.md`.
4. **Correctness tests:**  
   added greedy‑equivalence (gated) and spec‑decode indices kernel tests.  
   Files: `tests/v1/spec_decode/test_eagle_tree_decode.py`,
   `tests/v1/spec_decode/test_spec_decode_indices.py`.
5. **End‑to‑end perf harness:**  
   integrate a spec‑decode benchmark runner that logs tokens/sec, acceptance
   rate, draft/verify time, and total latency.  
   Files: `benchmarks/` (new) or reuse `/workspace/aimo/jupyter` script.


---

Merged from /workspace/aimo/VLLM.md:

# DriftEngine (vLLM) SM120 Plan

This doc outlines the SM120-first (RTX Pro 6000 Blackwell) bring-up for GPT-OSS-120B + Eagle3 in vLLM.
Target: maximum throughput and minimal load time on a single GPU (DP=1).

## Scope and Constraints
- Hardware: RTX Pro 6000 Blackwell (SM120), 96GB VRAM
- DP: 1 (single device only)
- Model: GPT-OSS-120B + Eagle3 draft (spec decoding)
- Max context: 65536
- Target batch: 8
- Model artifacts: local `models/` folder
- Format: safetensors-compatible, extra metadata allowed

## Inputs (Local Artifacts)
- Base HF shards: `models/gpt-oss-120b`
- Eagle3 draft: `models/gpt-oss-120b-eagle3`
- NVFP4 kv-cache variant: `models/gpt-oss-120b_kv_nvfp4`
- TurboMind format: `models/gpt-oss-120b-turbomind`

## Checkpoint Inspection (Quantized Weights)
### MXFP4 (HF: `models/gpt-oss-120b`)
- MoE weights are packed MXFP4 (uint8 blocks).
  - `gate_up_proj_blocks`: `[E=128, 2I=5760, H/32=90, 16]` uint8
  - `gate_up_proj_scales`: `[128, 5760, 90]` uint8
  - `down_proj_blocks`: `[128, H=2880, I/32=90, 16]` uint8
  - `down_proj_scales`: `[128, 2880, 90]` uint8
  - Biases are bf16 (`[128, 5760]`, `[128, 2880]`)
- Attention / router / norms / embeddings / lm_head are bf16.
- Sinks in HF are bf16 (`model.layers.N.self_attn.sinks`).

### NVFP4 (ModelOpt: `models/gpt-oss-120b_kv_nvfp4`)
- Quant method: `modelopt` with `kv_cache_scheme: NVFP4`.
- MoE weights stored as `uint8` + `float8_e4m3fn` scales:
  - `gate_up_proj`: `[128, 1440, 5760]` uint8
  - `gate_up_proj_weight_scale`: `[128, 180, 5760]` float8_e4m3fn
  - `down_proj`: `[128, 1440, 2880]` uint8
  - `down_proj_weight_scale`: `[128, 180, 2880]` float8_e4m3fn
  - `*_weight_scale_2`: scalar float32
- Attention / router / norms / embeddings / lm_head are bf16.

## Performance Goals
- Load time: 1-2 minutes (avoid runtime transpose/repack)
- Highest possible decode throughput at batch 8
- Correct outputs (sinks + sliding window) for GPT-OSS
- Stable at 64k context

## Kernel Stack (SM120, Initial Decisions)
### Attention (GPT-OSS requires sinks + sliding window)
- TRTLLM attention now accepts per-layer sliding window and sinks in vLLM.
- FlashInfer sinks path is still TRTLLM-only; we route prefill/decode through TRTLLM on SM120 when sinks are enabled.
- Short-term: FlashInfer backend with TRTLLM prefill+decode (sinks + window per layer).
- Fallback: FlashAttention backend if TRTLLM prefill proves unstable.
- SM120 TRTLLM-GEN requires local SM120 cubins; set `FLASHINFER_CUBIN_DIR` (or `FLASHINFER_CUBINS_REPOSITORY`)
  and `VLLM_FLASHINFER_TRTLLM_SM120_ALLOW=1` once cubins/meta are available.

### MoE (MXFP4 / NVFP4)
- GPT-OSS TurboMind uses MXFP4 with per-expert scales.
- vLLM supports MXFP4 MoE with FlashInfer backends on SM120.
- Backend selection gates on FlashInfer CUTLASS/TRTLLM fused MoE availability.
- Default on SM120: CUTLASS if available, else TRTLLM BF16; TRTLLM MXFP8 via env.
- GPT-OSS MXFP4 loader validates weight + scale layouts early.
- GPT-OSS MXFP4 block layout: w13 `(E, 2I, H/32, 16)` and w2 `(E, H, I/32, 16)` (uint8).

### KV Cache
- Use NVFP4 kv-cache where possible (head_dim=64 is compatible).
- Ensure kv_cache scales are materialized and shared for TRTLLM/FlashInfer paths.
- Fused RoPE+cache NVFP4 path is currently Triton-only for GPT-OSS.
- GPT-OSS NVFP4 path forces q/k/v to contiguous to enable fused RoPE+cache.

### GEMM
- For fp8 paths, prefer DeepGEMM or FlashInfer TRTLLM as available.
- For bf16 or mxfp4 paths, use FlashInfer cutlass where possible.

### Speculative Decoding (Eagle3)
- Use vLLM Eagle3 path with correct aux layers for GPT-OSS.
- Ensure attention backend and kv-cache choices are compatible with spec decode.

### EAGLE3 GPU Spec Decode (Parity Work)
- `VLLM_EAGLE_OPT_MODE=baseline|optimized|goal`: preset switch for comparing
  baseline vLLM vs optimized kernel path vs "goal" mode. When set, it overrides
  the EAGLE CUDA feature flags listed below.
- `VLLM_EAGLE_CUDA_DRAFT=1`: use fused CUDA draft-loop update (updates state + stores draft tokens).
- Draft sampling uses logit-only Gumbel path (no softmax/prob tensor) since draft_probs are unused; `VLLM_EAGLE_FAST_LOGITS=1` keeps this path even if draft_probs are requested later.
- `VLLM_EAGLE_CUDA_SAMPLE=1`: use CUDA sampling for draft steps (argmax for greedy; top‑k/top‑p + gumbel for non‑greedy).
- `VLLM_EAGLE_CUDA_TREE_COPY=1`: use CUDA kernels for tree drafting (copy level buffers + gather parent hidden states + select child tokens).
- `VLLM_EAGLE_CUDA_TREE_DRAFT=1`: use CUDA tree draft extraction (`eagle_extract_real_draft_tokens`) and preallocated buffers.
- Mixed draft lengths are supported in tree compaction via `eagle_expand_draft_tokens`.
- KV tree compaction path uses `VLLM_EAGLE_CUDA_KV_COMPACT=1`.
- Dynamic tree kernels (ported from TRT-LLM) are available via `torch.ops.vllm`:
  - `eagle_update_scores`, `eagle_update_path`, `eagle_update_draft_tokens_and_scores`.
  - `eagle_set_topks_from_dynamic_tree`, `eagle_assemble_second_topk_inputs`,
    `eagle_extract_scores_and_real_draft_tokens`, `eagle_assemble_third_topk_inputs`,
    `eagle_reconstruct_final_path`.
  - Wired to build dynamic drafts and cache per-request masks/paths when `VLLM_EAGLE_DYNAMIC_TREE=1`.
  - Dynamic tree verification uses cached masks/paths + GPU pack/compaction; target verification needs TRTLLM attention to consume custom masks.
  - `VLLM_EAGLE_DYNAMIC_TOPK=<int>` sets the dynamic-tree top‑k (defaults to root children).

#### EAGLE3 Mode Comparisons
- Baseline: `VLLM_EAGLE_OPT_MODE=baseline`
- Optimized: `VLLM_EAGLE_OPT_MODE=optimized`
- Goal: `VLLM_EAGLE_OPT_MODE=goal` (currently equivalent to optimized until the dynamic tree path is wired)

#### EAGLE3 Tests / Benches to Run
- Build kernels: `cmake --build build --target _C` and `cmake --install build --component _C`
- Unit tests: `pytest tests/v1/spec_decode/test_eagle.py`
- Tree path: `pytest tests/v1/spec_decode/test_eagle_tree_decode.py` and `pytest tests/v1/spec_decode/test_tree_attention.py`
- Rejection sampler: `pytest tests/v1/sample/test_rejection_sampler.py`
- E2E spec decode: `pytest tests/v1/e2e/test_spec_decode.py` and `pytest tests/v1/e2e/test_async_spec_decode.py`
- Bench (tree): `python benchmarks/benchmark_eagle_tree_decode.py --model <path>`
- Dynamic tree (manual): `VLLM_EAGLE_DYNAMIC_TREE=1 VLLM_EAGLE_DYNAMIC_TOPK=4 python benchmarks/benchmark_eagle_tree_decode.py --model <path>`
- Tree draft kernels (manual): `VLLM_EAGLE_CUDA_TREE_DRAFT=1 python benchmarks/benchmark_eagle_tree_decode.py --model <path>`
- New tests to add/run:
  - Dynamic tree acceptance equivalence: `pytest tests/v1/spec_decode/test_eagle_dynamic_tree.py`
  - Dynamic tree mask correctness vs TRT‑LLM: `pytest tests/v1/spec_decode/test_eagle_dynamic_tree_masks.py`

#### SM120 FlashInfer/TRTLLM Tests
- TRTLLM-GEN availability probe:
  `python - <<'PY'\nfrom vllm.utils.flashinfer import supports_trtllm_attention\nprint("supports_trtllm_attention:", supports_trtllm_attention())\nPY`
- FlashInfer TRTLLM-GEN decode sanity (requires SM120 cubins):
  `VLLM_FLASHINFER_TRTLLM_SM120_ALLOW=1 python -m vllm.entrypoints.openai.api_server --model /workspace/aimo/models/gpt-oss-120b --attention-backend FLASHINFER`
- FlashInfer CUTLASS MoE rebuild after kernel patches:
  `rm -rf ~/.cache/flashinfer/*/120a/cached_ops/fused_moe_120`

#### EAGLE3 Parity TODOs (TRT-LLM)
- Draft compute: replace static tree compute with dynamic-tree parent selection + reduced query_len per level.
- Tree kernels: port TRT‑LLM dynamic tree kernels (path update, score update, expand, final path reconstruction, packed-token move).
- GPU-only verification: eliminate CPU fallback in acceptance/packing; keep all acceptance indices on GPU.
- KV compaction: port TRT‑LLM packed-token move kernel for tree acceptance path.
- Attention backend: TRTLLM-only dynamic tree masks; tree attention backend still static.
- Spec metadata reuse: avoid per-step rebuild of packed masks; reuse cached buffers for CUDA graph.

## Model Format Strategy (Safetensors + Metadata)
Goal: keep safetensors but include metadata to skip runtime repacking.

### Proposed Metadata
- `drift.format`: `gpt-oss-packed`
- `drift.layout`: `qkv_fused`, `moe_mxfp4`, `kv_nvfp4`
- `drift.scales`: per-layer scaling tensors for mxfp4/nvfp4
- `drift.sinks`: per-layer sink tensor (float32)
- `drift.sliding_window`: per-layer window sizes
- `drift.permute`: any pre-applied tensor permutations

### Sources and Conversion
- HF shards: base weights (bf16)
- TurboMind: convert per-file expert weights + scales to packed safetensors
- ModelOpt NVFP4: reuse kv-cache quant config metadata

#### TurboMind -> Packed Safetensors Converter
Converter script: `scripts/turbomind_to_packed_safetensors.py`

Example (full convert + optional verify):
```bash
python scripts/turbomind_to_packed_safetensors.py \
  --tm-dir /workspace/aimo/models/gpt-oss-120b-turbomind \
  --out-dir /workspace/aimo/models/gpt-oss-120b-packed \
  --hf-dir /workspace/aimo/models/gpt-oss-120b \
  --verify-layer 0 \
  --force
```

Outputs:
- `model-layer-XX.safetensors` per layer
- `model-global.safetensors` for embeddings/norm/lm_head
- `model.safetensors.index.json`
- `drift_meta.json` (includes vocab_size, embedding_size, mxfp4 layout, sinks/window info)

#### Packed Loader Path (Drift Meta)
- If `drift_meta.json` is present in the model folder, vLLM reads it and marks
  weights as packed before loading.
- GPT-OSS loader skips runtime weight permutes (non-MXFP4) and can accept prepacked
  layouts without extra transpose work.

#### HF -> Packed Meta (No Weight Repack)
For HF MXFP4 checkpoints, weights are already in packed block layout. To
enable packed loading, write `drift_meta.json` in-place:
```bash
python scripts/hf_to_packed_meta.py \
  --model-dir /workspace/aimo/models/gpt-oss-120b \
  --verify-layer 0 \
  --force
```

#### Drift Format (vLLM-Native, Non-HF Naming)
Drift uses safetensors but stores vLLM parameter names (no HF mapping) and
ships `drift_meta.json` to mark packed layouts.

Convert HF -> Drift:
```bash
python scripts/hf_to_drift_safetensors.py \
  --model-dir /workspace/aimo/models/gpt-oss-120b \
  --out-dir /workspace/aimo/models/gpt-oss-120b-drift \
  --force
```

Load with vLLM:
```bash
vllm serve /workspace/aimo/models/gpt-oss-120b-drift \
  --load-format drift
```

#### Offline Prepack (FlashInfer CUTLASS MXFP4)
Precompute the CUTLASS layout so vLLM can skip runtime repack:
```bash
python scripts/prepack_drift_mxfp4_cutlass.py \
  --in-dir /workspace/aimo/models/gpt-oss-120b-drift \
  --out-dir /workspace/aimo/models/gpt-oss-120b-drift-cutlass \
  --force
```

Then load:
```bash
vllm serve /workspace/aimo/models/gpt-oss-120b-drift-cutlass \
  --load-format drift
```

#### Offline Prepack (FlashInfer TRTLLM MXFP4)
Use TRTLLM kernel layout (requires FlashInfer TRTLLM):
```bash
python scripts/prepack_drift_mxfp4_trtllm.py \
  --in-dir /workspace/aimo/models/gpt-oss-120b-drift \
  --out-dir /workspace/aimo/models/gpt-oss-120b-drift-trtllm \
  --force
```

#### Shard Consolidation (I/O Throughput)
Consolidate into fewer large shards for faster sequential reads:
```bash
python scripts/consolidate_drift_shards.py \
  --in-dir /workspace/aimo/models/gpt-oss-120b-drift-cutlass \
  --out-dir /workspace/aimo/models/gpt-oss-120b-drift-cutlass-consolidated \
  --max-shard-size-gb 32 \
  --force
```

## TurboMind Format Decision
- TurboMind format is optimized for TurboMind/TensorRT-style loaders, not vLLM.
- Best for vLLM: convert TurboMind → packed safetensors + metadata so we skip runtime repacks/transposes.
- Keep TurboMind as input only; packed safetensors is the runtime format.

## vLLM Integration Points (SM120)
- Model: `LM/vllm/vllm/model_executor/models/gpt_oss.py`
- Attention backends: `LM/vllm/vllm/v1/attention/backends/*`
- NVFP4 KV cache hooks: `LM/vllm/vllm/model_executor/models/rope_utils.py`
- MXFP4 MoE: `LM/vllm/vllm/model_executor/layers/quantization/mxfp4.py`
- ModelOpt NVFP4: `LM/vllm/vllm/model_executor/layers/quantization/modelopt.py`
- TRTLLM bindings: `LM/vllm/vllm/utils/trtllm.py`

## SM120 Bring-Up Sequence (Checklist)
1. Confirm GPT-OSS sinks + sliding window requirements in vLLM attention path.
2. Pick initial attention backend that supports sinks on SM120 (likely FlashAttention).
3. Verify that backend supports sliding window for alternating layers.
4. Enable NVFP4 kv-cache in vLLM; confirm head_dim compatibility.
5. Validate kv-cache scale tensors and fused rope+cache path.
6. Enable MXFP4 MoE with FlashInfer backend on SM120.
7. Verify FlashInfer cutlass/trtllm MoE kernels are selected.

## NVFP4 KV Cache Test Checklist (Do Not Run Yet)
- `pytest -q LM/vllm/tests/kernels/quantization/test_nvfp4_nibble_order.py`
- `pytest -q LM/vllm/tests/kernels/quantization/test_nvfp4_kv_cache_unpack.py`
- `pytest -q LM/vllm/tests/v1/attention/test_nvfp4_fused_rope_cache.py`
- `pytest -q LM/vllm/tests/v1/attention/test_gpt_oss_nvfp4_backend_selection.py`
- `pytest -q LM/vllm/tests/kernels/attention/test_flashinfer_trtllm_attention.py`
- `pytest -q LM/vllm/tests/models/quantization/test_nvfp4.py`
8. Ensure gpt-oss MoE weights map to vLLM names and expert layout.

## NVFP4 Test Matrix (GPT-OSS, DP=1)
### Preflight
- Ensure `_C` is rebuilt for SM100/SM120 before NVFP4 tests.
  - `TORCH_CUDA_ARCH_LIST="10.0a;12.0a" cmake -S /workspace/aimo/LM/vllm -B /workspace/aimo/LM/vllm/build/temp.linux-x86_64-cpython-311`
  - `cmake --build /workspace/aimo/LM/vllm/build/temp.linux-x86_64-cpython-311 --target _C`
  - `cp /workspace/aimo/LM/vllm/build/temp.linux-x86_64-cpython-311/_C.abi3.so /workspace/aimo/LM/vllm/vllm/_C.abi3.so`

### KV Cache + NVFP4 Core Correctness
- `pytest -q LM/vllm/tests/kernels/quantization/test_nvfp4_nibble_order.py`
- `pytest -q LM/vllm/tests/kernels/quantization/test_nvfp4_kv_cache_unpack.py`
- `pytest -q LM/vllm/tests/v1/attention/test_nvfp4_fused_rope_cache.py`
  - SM120 uses a slightly higher atol in the test to account for nvfp4 rounding.
- Set `VLLM_LOG_FUSED_NVFP4=1` to log when fused RoPE+NVFP4 cache is exercised.
  - Logs are suppressed during torch.compile; use `enforce_eager=True` for debug runs.

### NVFP4 Attention Read Kernels (TRTLLM / Triton)
- `pytest -q LM/vllm/tests/kernels/attention/test_flashinfer_trtllm_attention.py`
- `pytest -q LM/vllm/tests/kernels/attention/test_attention.py` (NVFP4 configs only)

### NVFP4 Quant Kernels (for GPT-OSS MoE / Linear Paths)
- `pytest -q LM/vllm/tests/kernels/quantization/test_nvfp4_quant.py`
- `pytest -q LM/vllm/tests/kernels/quantization/test_nvfp4_scaled_mm.py`
- `pytest -q LM/vllm/tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py`
- `pytest -q LM/vllm/tests/kernels/quantization/test_nvfp4_qutlass.py`
- `pytest -q LM/vllm/tests/kernels/quantization/test_silu_mul_nvfp4_quant.py`

### NVFP4 MoE Paths (GPT-OSS)
- `pytest -q LM/vllm/tests/kernels/moe/test_nvfp4_moe.py`
- `pytest -q LM/vllm/tests/kernels/moe/test_flashinfer_moe.py`

### Planned Tests (To Add)
- `LM/vllm/tests/v1/attention/test_nvfp4_cache_constraints.py`
  - Page size, scale shape, head_size constraints for NVFP4 KV cache.
- `LM/vllm/tests/models/test_gpt_oss_nvfp4.py`
  - GPT-OSS prefill+decode vs BF16 outputs + NVFP4 cache dtype validation.
- `LM/vllm/tests/v1/attention/test_nvfp4_backend_lock.py`
  - Ensure nvfp4_backend is TRTLLM/TRITON and no dequant fallback.

### Benchmarks (Performance Verification)
- `python LM/vllm/benchmarks/kernels/benchmark_fused_rope_cache_nvfp4.py`
- `python scripts/run_gpt_oss_throughput.py --gpu-memory-utilization 0.8 --max-num-seqs 8`

### SM120 Baselines (ctx 65536, out 128, nvfp4, gpu_mem=0.8)
- Drift (cutlass consolidated, batch 8):
  - Load weights: 10.29s; model load: 10.88s
  - 0.45 req/s, 29.73k tokens/s, 57.95 output tokens/s
- HF (pure, no drift_meta, batch 8):
  - Load weights: 10.08s; model load: 10.78s
  - 0.45 req/s, 29.55k tokens/s, 57.60 output tokens/s
- HF (with drift_meta present, batch 8, reference):
  - 0.46 req/s, 30.19k tokens/s, 58.86 output tokens/s
- Drift (MoE backend=throughput, batch 8):
  - Env: `VLLM_FLASHINFER_MOE_BACKEND=throughput` `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1`
  - Load weights: 101.51s; model load: 102.83s (likely cold page cache)
  - 0.44 req/s, 28.94k tokens/s, 56.41 output tokens/s
  - Results: `results/gpt_oss_throughput_drift_nvfp4_moe_throughput.json`, `results/gpt_oss_throughput_drift_nvfp4_moe_throughput.log`
- Drift (block_size=32, batch 8):
  - Load weights: 38.49s; model load: 39.32s
  - 0.44 req/s, 28.88k tokens/s, 56.30 output tokens/s
  - Results: `results/gpt_oss_throughput_drift_nvfp4_block32.json`, `results/gpt_oss_throughput_drift_nvfp4_block32.log`
- Results: `results/gpt_oss_throughput_drift_b8.json`, `results/gpt_oss_throughput_hf_pure.json`, `results/gpt_oss_throughput_hf.json`
- Note: load times are cache-sensitive; runs may warm the page cache.

### SM120 Cold-Cache Comparison (best effort)
- `drop_caches` failed in this container (`/proc/sys/vm/drop_caches` is read-only), so these are best-effort cold-ish runs.
- Drift (cutlass consolidated, batch 8):
  - Load weights: 35.99s; model load: 36.52s
  - 0.41 req/s, 26.88k tokens/s, 52.39 output tokens/s
- HF (pure, no drift_meta, batch 8):
  - Load weights: 39.78s; model load: 40.65s
  - 0.46 req/s, 30.51k tokens/s, 59.47 output tokens/s
- Results: `results/gpt_oss_throughput_drift_cold.json`, `results/gpt_oss_throughput_hf_pure_cold.json`

### SM120 Batch Sweep (Drift, gpu_mem=0.8)
- Batch 4: 0.14 req/s, 9.47k tokens/s, 18.46 output tokens/s
- Batch 8: 0.45 req/s, 29.73k tokens/s, 57.95 output tokens/s
- Batch 12: 0.46 req/s, 29.99k tokens/s, 58.46 output tokens/s
- Results: `results/gpt_oss_throughput_drift_b4.json`, `results/gpt_oss_throughput_drift_b8.json`, `results/gpt_oss_throughput_drift_b12.json`

### SM120 BF16 KV Cache (Drift, ctx 65536, out 128, batch 8)
- Env: `VLLM_USE_TRTLLM_ATTENTION=0` (FlashInfer TRTLLM attention path fails on SM120)
- Backend: `TRITON_ATTN`
- Load weights: 91.82s; model load: 93.08s
- 0.07 req/s, 4.65k tokens/s, 9.07 output tokens/s
- Results: `results/gpt_oss_throughput_drift_bf16.json`, `results/gpt_oss_throughput_drift_bf16.log`
- `python LM/vllm/benchmarks/kernels/benchmark_reshape_and_cache_flash.py` (NVFP4 mode)
- Planned: `LM/vllm/benchmarks/kernels/benchmark_paged_attention_nvfp4.py`
- Planned: `LM/vllm/benchmarks/gpt_oss_nvfp4_e2e.py` (throughput/latency vs TRT-LLM)
- Single-GPU throughput harness:
  - `python scripts/run_gpt_oss_throughput.py`
9. Add safetensors metadata reader for packed weights (no repack).
10. Add safetensors writer for packed weights (TurboMind -> packed).
11. Add packed weight loader path (bypass transpose/permute).
12. Validate sinks tensors loaded as float32.
13. Confirm sliding window metadata propagation per layer.
14. Verify Eagle3 draft model loads and captures aux layers.
15. Verify spec decode is compatible with chosen attention backend.
16. Build a single-GPU perf harness (ctx 65536, batch 8).
17. Lock deterministic settings (disable autotune noise during profiling).
18. Profile decode/prefill kernels (nsys).
19. Iterate backend envs for MoE and GEMM on SM120.
20. Measure load time with packed safetensors vs default.
21. Validate correctness (logits parity vs baseline).
22. Validate long-context stability (64k).
23. Document final vLLM launch args + env vars.
24. Record perf baselines for SM120.
25. Only after SM120 is stable, extend to SM90 (H100).

## SM120 Runtime Knobs (Initial Defaults)
- `--max-model-len 65536`
- `--max-num-seqs 8`
- `--kv-cache-dtype nvfp4`
- `--attention-backend FLASHINFER`
- `--attention-config.use_trtllm_attention=1`
- `--speculative-model /workspace/aimo/models/gpt-oss-120b-eagle3`
- `--speculative-steps` / `--speculative-num-draft-tokens` as needed

## SM120 Env Knobs (Initial Defaults)
- `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1`
- `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1` (if accuracy allows)
- `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` (TRTLLM path, requires fused MoE)
- `VLLM_NVFP4_GEMM_BACKEND=flashinfer-trtllm` (if installed)
- `VLLM_USE_TRTLLM_MOE=1` (optional, uses TRTLLM MoE custom ops when available)
- `VLLM_KV_CACHE_LAYOUT=HND` (required for TRTLLM prefill/decode path)

## Tests and Validation (TRTLLM MoE Custom Ops)
- Sanity: confirm TRTLLM custom ops are visible before enabling:
```bash
VLLM_TRTLLM_PYTHON_PATH=/workspace/aimo/LM/TensorRT-LLM \
python - <<'PY'
from vllm.utils.trtllm import has_trtllm_fp8_block_scale_moe, has_trtllm_fp4_block_scale_moe
print("fp8_block_scale_moe:", has_trtllm_fp8_block_scale_moe())
print("fp4_block_scale_moe:", has_trtllm_fp4_block_scale_moe())
PY
```
- NVFP4 correctness (GPT-OSS): run a short decode with `VLLM_USE_TRTLLM_MOE=1` and compare logits to FlashInfer path (same prompt, same seed).
- FP8 block-scale correctness: run a small MoE model with `VLLM_USE_TRTLLM_MOE=1` and compare outputs vs FlashInfer TRTLLM path within tolerance.
- Throughput: record decode TPS with and without `VLLM_USE_TRTLLM_MOE=1` on SM120; keep batch size and max length fixed.

## Known Gaps (Must Fix)
- TRTLLM prefill on SM120 is still experimental; verify stability and correctness.
- FlashInfer prefill (non-TRTLLM) still lacks sinks support.
- SM120 Triton mxfp4 kernels are gated by upstream fixes.
- NVFP4 fused RoPE+cache is disabled for FlashInfer/TRTLLM backends (requires Triton).
- FlashInfer MXFP4 paths require fused MoE kernels; otherwise fallback is Marlin/Triton.

## Deliverables
- Packed safetensors format + converter for TurboMind.
- vLLM loader for packed metadata (no repack).
- SM120 kernel stack with sinks + sliding window correctness.
- Reproducible perf harness and baseline logs.
