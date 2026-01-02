# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu.spec_decode.eagle_cudagraph import EagleCudaGraphManager

logger = init_logger(__name__)


# ============================================================================
# Helper functions ported from SGLang for dynamic tree construction
# ============================================================================


def fast_topk(
    x: torch.Tensor, k: int, dim: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fast top-k selection. Falls back to torch.topk."""
    return torch.topk(x, k, dim=dim)


@torch.compile(dynamic=True)
def select_top_k_tokens(
    step: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
    """Select top-k tokens at each step of tree construction.

    Ported from SGLang's spec_utils.py.
    """
    if step == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        if hidden_states is not None:
            hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )
    else:
        # The later decode steps
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk, topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten()

        if hidden_states.shape[0] > 0:
            selected_input_index = topk_cs_index.flatten() // topk + torch.arange(
                0, hidden_states.shape[0], step=topk, device=topk_index.device
            ).repeat_interleave(topk)
            hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (step - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Organize draft results with top-k selection across all levels.

    Ported from SGLang's eagle_utils.py.
    """
    score_list_cat = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list_cat, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(
            batch_size, 0, device=parents_list[0].device, dtype=torch.long
        )

    return parent_list, top_scores_index, draft_tokens


def organize_draft_results_tensor(
    score_buffer: torch.Tensor,
    token_buffer: torch.Tensor,
    parent_buffer: torch.Tensor,
    num_draft_token: int,
    topk: int,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Organize draft results from pre-allocated tensors (graph-friendly)."""

    # Select top-k across all candidates
    # score_buffer has shape [BS, num_steps * topk]
    top_scores = torch.topk(score_buffer, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices

    # Sort indices for consistent ordering
    top_scores_index = torch.sort(top_scores_index).values

    # Gather selected tokens
    draft_tokens = torch.gather(token_buffer, index=top_scores_index, dim=1)

    # Parent list: use all but the last step's parents
    # SGLang logic: torch.cat(parents_list[:-1], dim=1)
    # parent_buffer has shape [BS, num_steps * topk]
    if num_steps > 1:
        cutoff = (num_steps - 1) * topk
        parent_list = parent_buffer[:, :cutoff]
    else:
        batch_size = parent_buffer.shape[0]
        parent_list = torch.empty(
            batch_size, 0, device=parent_buffer.device, dtype=torch.long
        )

    return parent_list, top_scores_index, draft_tokens


# ============================================================================
# End of ported helper functions
# ============================================================================


class EagleSpeculator:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.method = self.speculative_config.method
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.draft_model_config = self.speculative_config.draft_model_config

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.pin_memory = is_pin_memory_available()
        self.dtype = vllm_config.model_config.dtype

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            inputs_embeds_size=self.inputs_embeds_size,
            vocab_size=self.vocab_size,
            dtype=self.dtype,
            device=device,
            pin_memory=self.pin_memory,
        )
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self.temperature = torch.zeros(
            self.max_num_reqs,
            dtype=torch.float32,
            device=device,
        )
        self.seeds = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int64,
            device=device,
        )
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        if self.method == "dynamic_eagle":
            self.topk = self.speculative_config.speculative_eagle_topk
            self.num_draft_tokens = self.speculative_config.speculative_num_draft_tokens
            # SGLang: num_verify_tokens = num_draft_tokens + 1
            self.num_verify_tokens = self.num_draft_tokens + 1

            self.dyn_tree_mask = torch.ones(
                (self.max_num_reqs, self.num_verify_tokens, self.num_verify_tokens),
                dtype=torch.bool,
                device=device,
            )
            self.dyn_positions = torch.zeros(
                (self.max_num_reqs * self.num_verify_tokens),
                dtype=torch.long,
                device=device,
            )
            self.retrive_index = torch.full(
                (self.max_num_reqs, self.num_verify_tokens),
                -1,
                dtype=torch.long,
                device=device,
            )
            self.retrive_next_token = torch.full(
                (self.max_num_reqs, self.num_verify_tokens),
                -1,
                dtype=torch.long,
                device=device,
            )
            self.retrive_next_sibling = torch.full(
                (self.max_num_reqs, self.num_verify_tokens),
                -1,
                dtype=torch.long,
                device=device,
            )
            # draft_tokens_tree: total draft tokens to be verified
            self.draft_tokens_tree = torch.zeros(
                (self.max_num_reqs, self.num_verify_tokens),
                dtype=torch.int64,
                device=device,
            )

            self.parent_list_tree = torch.zeros(
                (self.max_num_reqs, self.topk * (self.num_speculative_steps - 1) + 1),
                dtype=torch.int64,
                device=device,
            )
            self.selected_index_tree = torch.zeros(
                (self.max_num_reqs, self.num_verify_tokens - 1),
                dtype=torch.int64,
                device=device,
            )
            self.verified_seq_len_tree = torch.zeros(
                self.max_num_reqs, dtype=torch.int64, device=device
            )

            # Buffers for CUDA graph loop capture
            total_candidates = self.num_speculative_steps * self.topk
            self.score_buffer = torch.zeros(
                (self.max_num_reqs, total_candidates),
                dtype=torch.float32,
                device=device,
            )
            self.token_buffer = torch.zeros(
                (self.max_num_reqs, total_candidates),
                dtype=torch.int64,
                device=device,
            )
            self.parent_buffer = torch.zeros(
                (self.max_num_reqs, total_candidates),
                dtype=torch.int64,
                device=device,
            )

            # Input buffers for the loop
            self.draft_input_logits = torch.zeros(
                (self.max_num_reqs, self.vocab_size),
                dtype=self.dtype,  # Logits are usually float32? Model output dtype.
                device=device,
            )
            # Hidden states buffer - initial state for loop
            self.draft_input_hidden_states = torch.zeros(
                (self.max_num_reqs, self.hidden_size),
                dtype=self.dtype,
                device=device,
            )

            # Intermediate loop states
            self.loop_scores = torch.zeros(
                (self.max_num_reqs, self.topk),
                dtype=torch.float32,
                device=device,
            )
            self.is_dynamic_graph_captured = False

        self.cudagraph_manager = EagleCudaGraphManager(vllm_config, device)

    def load_model(self, target_model: nn.Module) -> None:
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("eagle_head"):
            self.model = get_model(
                vllm_config=self.vllm_config, model_config=self.draft_model_config
            )

        share_lm_head = True
        if share_lm_head and hasattr(target_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_model.lm_head

    def set_attn(
        self,
        kv_cache_config: KVCacheConfig,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        block_tables: BlockTables,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.attn_metadata_builders = attn_metadata_builders
        self.block_tables = block_tables

    @torch.inference_mode()
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any],
        num_tokens_across_dp: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
        ):
            ret_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=None,
            )
        if self.method == "mtp":
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    def generate_draft(
        self,
        num_reqs: int,
        attn_metadata: dict[str, Any],
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        pos = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc.gpu[: num_reqs + 1]
        for step in range(1, self.num_speculative_steps):
            # Run the eagle model.
            last_hidden_states, hidden_states = self.run_model(
                num_reqs, attn_metadata, num_tokens_across_dp
            )
            logits = self.model.compute_logits(last_hidden_states)

            # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
            # used for draft and target sampling.
            draft_tokens = gumbel_sample(
                logits,
                self.temperature[:num_reqs],
                self.seeds[:num_reqs],
                pos + 1,
                apply_temperature=True,
            )
            self.draft_tokens[:num_reqs, step] = draft_tokens

            if step < self.num_speculative_steps - 1:
                # Update the inputs for the next step.
                update_eagle_inputs(
                    draft_tokens,
                    hidden_states,
                    self.input_buffers,
                    self.hidden_states,
                    self.max_model_len,
                )
                self.block_tables.compute_slot_mappings(query_start_loc, pos)

    def capture_model(self) -> None:
        if self.num_speculative_steps == 1:
            return
        logger.info("Capturing model for Eagle speculator...")
        self.cudagraph_manager.capture(
            self.generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_metadata_builders,
            self.kv_cache_config,
        )

    @torch.no_grad()
    def _run_dynamic_draft_loop_captured(self):
        """Internal method designed to be captured into a CUDA graph."""
        # Use captured input/output buffers directly
        # self.input_buffers contains: input_ids, positions, hidden_states
        # self.vllm_config, etc. are static

        print(f"[Dynamic Eagle Graph] Entering captured loop...")
        num_reqs = self.input_buffers.num_reqs
        topk = self.topk
        num_tokens_total = num_reqs * topk
        # ... rest of the code
        batch_size = num_tokens_total // topk
        num_steps = self.num_speculative_steps

        # Initial inputs from buffers
        logits = self.draft_input_logits[:batch_size]
        hidden_states = self.draft_input_hidden_states[:batch_size]
        scores = self.loop_scores[:batch_size]

        # We need the loop state across steps
        # Initial Step 0 setup
        probs = torch.softmax(logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, topk, dim=-1)

        # Loop for num_steps
        for step in range(num_steps):
            # Select top-k and prepare next inputs
            # select_top_k_tokens returns (input_ids, hidden_states, scores, tree_info)
            # tree_info = (scores, input_ids, parents)
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                step, topk_p, topk_index, hidden_states, scores, topk
            )

            # Write to buffers
            start_idx = step * topk
            end_idx = (step + 1) * topk

            # tree_info[0] is scores [BS, topk]
            self.score_buffer[:batch_size, start_idx:end_idx] = tree_info[0]
            # tree_info[1] is input_ids (tokens) [BS, topk]
            self.token_buffer[:batch_size, start_idx:end_idx] = tree_info[1]
            # tree_info[2] is parents [BS, topk]
            self.parent_buffer[:batch_size, start_idx:end_idx] = tree_info[2]

            # Break if last step (we don't need to run model for output of last step)
            if step == num_steps - 1:
                break

            # Prepare inputs for next step draft model forward
            # input_ids from select_top_k_tokens are flattened [BS*topk]?
            # select_top_k_tokens returns input_ids shape [BS*topk]

            # Update input buffers
            num_input_tokens = batch_size * topk
            self.input_buffers.input_ids[:num_input_tokens] = input_ids
            # Update positions (in-place add works in graph)
            self.input_buffers.positions[:num_input_tokens].add_(1)

            # Run draft model
            # Note: We must use set_forward_context logic if not implicit
            # run_model handles context.
            last_hidden, hidden_states = self.run_model(
                num_input_tokens,
                attn_metadata,
                num_tokens_across_dp,
            )

            # Compute logits -> probs -> topk for next step
            draft_logits = self.model.compute_logits(last_hidden)
            probs = torch.softmax(draft_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, topk, dim=-1)

    @torch.inference_mode()
    def generate_dynamic_tree_draft(
        self,
        batch_size: int,
        logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        attn_metadata: dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform dynamic tree-based draft token generation."""
        topk = self.topk
        num_draft_tokens = self.num_draft_tokens
        num_steps = self.num_speculative_steps

        # Set verified sequence lengths
        self.verified_seq_len_tree[:batch_size] = seq_lens[:batch_size].to(torch.int64)

        # Check if we can use CUDA graph
        # Graph key is num_tokens = batch_size * topk
        graph_key = batch_size * topk

        if (
            self.cudagraph_manager.cudagraph_mode != CUDAGraphMode.NONE
            and graph_key in self.cudagraph_manager.graphs
        ):
            # USE CUDA GRAPH REPLAY
            print(f"[Dynamic Eagle] Running captured graph for BS={batch_size}")
            # 1. Copy inputs to static buffers
            self.draft_input_logits[:batch_size] = logits
            self.draft_input_hidden_states[:batch_size] = hidden_states

            # 2. Replay graph
            self.cudagraph_manager.run(graph_key)

            # 3. Read outputs from buffers and organize
            # We pass the full buffers; the helper handles slicing based on batch_size
            parent_list, top_scores_index, organized_draft_tokens = (
                organize_draft_results_tensor(
                    self.score_buffer,
                    self.token_buffer,
                    self.parent_buffer,
                    num_draft_tokens,
                    topk,
                    num_steps,
                )
            )

            # Slice the results for return (organize helper returns full batch slice)
            parent_list = parent_list[:batch_size]
            top_scores_index = top_scores_index[:batch_size]
            organized_draft_tokens = organized_draft_tokens[:batch_size]
            print(f"[Dynamic Eagle] Captured graph run complete for BS={batch_size}")

        else:
            # FALLBACK TO PYTHON LOOP (Original implementation)
            print(f"[Dynamic Eagle] Falling back to Python loop for BS={batch_size}")
            # Return values - accumulate tree info across steps
            score_list: List[torch.Tensor] = []
            token_list: List[torch.Tensor] = []
            parents_list: List[torch.Tensor] = []

            # Get initial top-k from logits
            probs = torch.softmax(logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, topk, dim=-1)

            # For tracking cumulative scores across steps
            scores = None

            # Forward multiple steps
            for step in range(num_steps):
                # Select top-k tokens and update hidden states
                input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                    step, topk_p, topk_index, hidden_states, scores, topk
                )
                score_list.append(tree_info[0])
                token_list.append(tree_info[1])
                parents_list.append(tree_info[2])

                # Skip last forward - we only need (num_steps - 1) draft model runs
                if step == num_steps - 1:
                    break

                # Prepare inputs for draft model forward
                num_tokens = input_ids.shape[0]
                self.input_buffers.input_ids[:num_tokens] = input_ids

                # Update positions for this step
                self.input_buffers.positions[:num_tokens].add_(1)

                # Run draft model forward
                last_hidden, hidden_states = self.run_model(
                    num_tokens,
                    attn_metadata,
                    num_tokens_across_dp=None,
                )

                # Compute logits and get top-k for next step
                draft_logits = self.model.compute_logits(last_hidden)
                probs = torch.softmax(draft_logits, dim=-1)
                topk_p, topk_index = fast_topk(probs, topk, dim=-1)

            # Organize results with top-k selection across all levels
            parent_list, top_scores_index, organized_draft_tokens = (
                organize_draft_results(
                    score_list, token_list, parents_list, num_draft_tokens
                )
            )
            print(f"[Dynamic Eagle] Python loop run complete for BS={batch_size}")

        # Common output storage
        self.parent_list_tree[:batch_size, : parent_list.shape[1]] = parent_list
        self.selected_index_tree[:batch_size, : top_scores_index.shape[1]] = (
            top_scores_index.to(torch.int64)
        )

        # Store draft tokens - add root token first
        self.draft_tokens_tree[:batch_size, 0] = draft_tokens
        # organized_draft_tokens has shape [batch, num_draft_tokens - 1]
        self.draft_tokens_tree[:batch_size, 1:num_draft_tokens] = organized_draft_tokens

        return parent_list, top_scores_index, organized_draft_tokens

    def capture_dynamic_draft_graphs(self):
        """Capture CUDA graphs for the dynamic draft loop."""
        if self.cudagraph_manager.cudagraph_mode == CUDAGraphMode.NONE:
            return

        # Iterate over supported batch sizes for decode
        for batch_size in self.cudagraph_manager.cudagraph_sizes:
            # We capture for input size = batch_size * topk
            num_tokens = batch_size * self.topk

            # Avoid re-capturing if already exists (though unlikely)
            if num_tokens in self.cudagraph_manager.graphs:
                continue

            try:
                print(f"[Dynamic Eagle] Capturing graph for BS={batch_size}...")
                self.cudagraph_manager.capture_graph(
                    num_tokens,
                    self._run_dynamic_draft_loop_captured,
                    self.input_buffers,
                    self.block_tables,
                    self.attn_metadata_builders,
                    self.kv_cache_config,
                )
                print(f"[Dynamic Eagle] Graph captured for BS={batch_size}.")
            except Exception as e:
                # Fallback safely if capture fails (e.g. memory)
                print(
                    f"Warning: Failed to capture dynamic eagle graph for BS={batch_size}: {e}"
                )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        sampling_metadata: SamplingMetadata,
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [num_reqs]
        last_sampled: torch.Tensor,
        # [num_reqs]
        next_prefill_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # Lazy CUDA graph capture for Dynamic Eagle.
        # Triggered on first propose call, AFTER torch.compile warmup is complete.
        if self.method == "dynamic_eagle" and not getattr(
            self, "is_dynamic_graph_captured", False
        ):
            print("[Dynamic Eagle] Capturing CUDA Graphs for Draft Loop...")
            self.capture_dynamic_draft_graphs()
            self.is_dynamic_graph_captured = True
            print("[Dynamic Eagle] CUDA Graph Capture Complete.")

        # NOTE(woosuk): To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of eagle's input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions. By doing so,
        # we can also reuse the attention metadata (e.g., query_start_loc,
        # seq_lens) of the target model.
        if aux_hidden_states:
            assert self.method == "eagle3"
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        num_tokens = input_batch.num_tokens_after_padding
        self.hidden_states[:num_tokens] = hidden_states

        # Get the input ids and last token indices for the speculator.
        last_token_indices = prepare_eagle_inputs(
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
        )

        # Prefill: Run the eagle speculator with eager mode.
        # TODO(woosuk): Support CUDA graph for prefill.
        last_hidden_states, hidden_states = self.run_model(
            num_tokens,
            input_batch.attn_metadata,
            num_tokens_across_dp=None,  # FIXME
        )
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        num_reqs = input_batch.num_reqs
        cu_num_logits = input_batch.cu_num_logits[:num_reqs]
        # NOTE(woosuk): For draft sampling, we only consider the temperature
        # and ignore the other sampling parameters such as top_k and top_p,
        # for simplicity and performance.
        # While this may slightly degrade the acceptance rate, it does not
        # affect the output distribution after rejection sampling.
        temperature = self.temperature[:num_reqs]
        seeds = self.seeds[:num_reqs]
        pos = self.input_buffers.positions[:num_reqs]
        if self.method == "dynamic_eagle":
            # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
            # used for draft and target sampling.
            draft_tokens = gumbel_sample(
                logits, temperature, seeds, pos + 1, apply_temperature=True
            )
            self.draft_tokens[:num_reqs, 0] = draft_tokens

            if self.num_speculative_steps > 1:
                # Lazy capture if not done yet
                if (
                    self.cudagraph_manager.cudagraph_mode != CUDAGraphMode.NONE
                    and not self.cudagraph_manager.graphs
                ):
                    print(
                        f"[Dynamic Eagle] STARTING capture_dynamic_draft_graphs for method={self.method}"
                    )
                    self.capture_dynamic_draft_graphs()
                    print(f"[Dynamic Eagle] FINISHED capture_dynamic_draft_graphs")

            print(
                f"[Dynamic Eagle] Calling generate_dynamic_tree_draft for num_reqs={num_reqs}"
            )
            # Dynamic tree generation
            self.generate_dynamic_tree_draft(
                num_reqs,
                logits,
                draft_tokens,
                sample_hidden_states,
                input_batch.seq_lens,
                input_batch.attn_metadata,
            )
            print(f"[Dynamic Eagle] generate_dynamic_tree_draft complete")

            # Build tree structure for verification using the ported kernel
            from vllm._custom_ops import ops

            ops.build_tree_kernel_efficient(
                self.parent_list_tree,
                self.selected_index_tree,
                self.verified_seq_len_tree,
                self.dyn_tree_mask,
                self.dyn_positions,
                self.retrive_index,
                self.retrive_next_token,
                self.retrive_next_sibling,
                self.topk,
                self.num_speculative_steps,
                self.num_verify_tokens,
                0,  # FULL_MASK
            )

            # Return the draft tokens organized in a tree (flat for now, as expected by v1 worker)
            return self.draft_tokens_tree[:num_reqs]

        # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise

        # used for draft and target sampling.
        draft_tokens = gumbel_sample(
            logits, temperature, seeds, pos + 1, apply_temperature=True
        )
        if self.num_speculative_steps == 1:
            # Early exit.
            return draft_tokens.view(-1, 1)

        # Save the draft tokens for the first step.
        self.draft_tokens[:num_reqs, 0] = draft_tokens
        # Prepare the inputs for the decode steps.
        prepare_eagle_decode(
            draft_tokens,
            hidden_states,
            last_token_indices,
            input_batch.seq_lens,
            num_rejected,
            self.input_buffers,
            self.hidden_states,
            self.max_model_len,
            self.max_num_reqs,
        )
        query_start_loc = self.input_buffers.query_start_loc
        query_start_loc_gpu = query_start_loc.gpu[: num_reqs + 1]
        slot_mappings = self.block_tables.compute_slot_mappings(
            query_start_loc_gpu, pos
        )

        cudagraph_size = self.cudagraph_manager.get_cudagraph_size(num_reqs)
        if cudagraph_size is not None:
            # Run CUDA graph.
            self.cudagraph_manager.run(cudagraph_size)
            return self.draft_tokens[:num_reqs]

        # Run eager mode.
        query_start_loc.np[: num_reqs + 1] = np.arange(num_reqs + 1)
        query_start_loc_cpu = query_start_loc.cpu[: num_reqs + 1]
        # HACK(woosuk)
        seq_lens_np = np.full(num_reqs, self.max_model_len, dtype=np.int32)
        block_tables = [x[:num_reqs] for x in self.block_tables.input_block_tables]

        # FIXME(woosuk): This is UNSAFE!!
        attn_metadata = build_attn_metadata(
            attn_metadata_builders=self.attn_metadata_builders,
            num_reqs=num_reqs,
            num_tokens=num_reqs,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=self.input_buffers.seq_lens[:num_reqs],
            seq_lens_np=seq_lens_np,
            num_computed_tokens_cpu=None,  # FIXME
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )
        self.generate_draft(num_reqs, attn_metadata, num_tokens_across_dp=None)  # FIXME
        return self.draft_tokens[:num_reqs]


@triton.jit
def _prepare_eagle_inputs_kernel(
    last_token_indices_ptr,
    eagle_input_ids_ptr,
    eagle_positions_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    # Get the true query length and next token after accounting for rejected tokens.
    num_rejected = tl.load(num_rejected_ptr + batch_idx)
    query_len -= num_rejected

    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + batch_idx).to(tl.int32)
    else:
        # Chunked prefilling.
        # Get the next prefill token.
        next_token = tl.load(next_prefill_tokens_ptr + batch_idx)

    # Shift target_input_ids by one.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(eagle_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + batch_idx, last_token_index)
    tl.store(eagle_input_ids_ptr + last_token_index, next_token)

    # Copy positions.
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(eagle_positions_ptr + query_start + block, target_pos, mask=mask)


def prepare_eagle_inputs(
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [num_reqs]
    last_sampled: torch.Tensor,
    # [num_reqs]
    next_prefill_tokens: torch.Tensor,
) -> torch.Tensor:
    num_reqs = input_batch.num_reqs
    last_token_indices = torch.empty(
        num_reqs,
        dtype=torch.int64,
        device=num_sampled.device,
    )
    _prepare_eagle_inputs_kernel[(num_reqs,)](
        last_token_indices,
        input_buffers.input_ids,
        input_buffers.positions,
        input_batch.input_ids,
        input_batch.positions,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
        BLOCK_SIZE=1024,
    )
    return last_token_indices


@triton.jit
def _prepare_eagle_docode_kernel(
    draft_tokens_ptr,
    output_hidden_states_ptr,
    output_hidden_states_stride,
    last_token_indices_ptr,
    target_seq_lens_ptr,
    num_rejected_ptr,
    input_ids_ptr,
    positions_ptr,
    input_hidden_states_ptr,
    input_hidden_states_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    hidden_size,
    max_model_len,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        # Compute query_start_loc. Pad it with the last query_start_loc
        # for CUDA graphs.
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block < num_reqs, block, num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    # draft token -> input id.
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    tl.store(input_ids_ptr + req_idx, draft_token)

    # output hidden states -> input hidden states.
    src_idx = tl.load(last_token_indices_ptr + req_idx)
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        output_hidden_states = tl.load(
            output_hidden_states_ptr + src_idx * output_hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            input_hidden_states_ptr + req_idx * input_hidden_states_stride + block,
            output_hidden_states,
            mask=mask,
        )

    # Compute position and seq_lens.
    # NOTE(woosuk): To prevent out-of-range access, we clamp these values
    # if they reach the max model length.
    position = tl.load(positions_ptr + req_idx)
    position = tl.minimum(position + 1, max_model_len - 1)
    tl.store(positions_ptr + req_idx, position)

    target_seq_len = tl.load(target_seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    seq_len = target_seq_len - num_rejected
    seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def prepare_eagle_decode(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    last_token_indices: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    input_hidden_states: torch.Tensor,
    max_model_len: int,
    max_num_reqs: int,
):
    num_reqs = draft_tokens.shape[0]
    hidden_size = output_hidden_states.shape[-1]
    _prepare_eagle_docode_kernel[(num_reqs + 1,)](
        draft_tokens,
        output_hidden_states,
        output_hidden_states.stride(0),
        last_token_indices,
        target_seq_lens,
        num_rejected,
        input_buffers.input_ids,
        input_buffers.positions,
        input_hidden_states,
        input_hidden_states.stride(0),
        input_buffers.query_start_loc.gpu,
        input_buffers.seq_lens,
        hidden_size,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _update_eagle_inputs_kernel(
    input_ids_ptr,
    positions_ptr,
    input_hidden_states_ptr,
    input_hidden_states_stride,
    seq_lens_ptr,
    max_model_len,
    draft_tokens_ptr,
    output_hidden_states_ptr,
    output_hidden_states_stride,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    # Draft token -> Input ID.
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Output hidden states -> Input hidden states.
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        output_hidden_states = tl.load(
            output_hidden_states_ptr + req_idx * output_hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            input_hidden_states_ptr + req_idx * input_hidden_states_stride + block,
            output_hidden_states,
            mask=mask,
        )

    # Increment position and seq_lens.
    # NOTE(woosuk): To prevent out-of-range access, we clamp these values
    # if they reach the max model length.
    position = tl.load(positions_ptr + req_idx)
    position = tl.minimum(position + 1, max_model_len - 1)
    tl.store(positions_ptr + req_idx, position)

    seq_len = tl.load(seq_lens_ptr + req_idx)
    seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def update_eagle_inputs(
    draft_tokens: torch.Tensor,
    output_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    hidden_states: torch.Tensor,
    max_model_len: int,
):
    num_reqs, hidden_size = output_hidden_states.shape
    _update_eagle_inputs_kernel[(num_reqs,)](
        input_buffers.input_ids,
        input_buffers.positions,
        hidden_states,
        hidden_states.stride(0),
        input_buffers.seq_lens,
        max_model_len,
        draft_tokens,
        output_hidden_states,
        output_hidden_states.stride(0),
        hidden_size,
        BLOCK_SIZE=1024,
    )
