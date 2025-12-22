# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.metadata import SamplingMetadata
from vllm import _custom_ops as ops
from vllm.v1.worker.gpu.spec_decode.eagle_cudagraph import (
    EagleCudaGraphManager,
    EaglePrefillCudaGraphManager,
)

logger = init_logger(__name__)


class _EagleSpeculatorDraftLoop(nn.Module):
    def __init__(self, speculator: "EagleSpeculator"):
        super().__init__()
        self.speculator = speculator

    def forward(
        self,
        num_reqs: int,
        attn_metadata: dict[str, Any],
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        self.speculator._draft_loop_impl(
            num_reqs, attn_metadata, num_tokens_across_dp
        )


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
        # Tree drafting constants
        self.max_decoding_tokens = self.num_speculative_steps + 1
        self.max_total_draft_tokens = self.num_speculative_steps
        self.max_path_len = self.num_speculative_steps + 1
        self.max_nodes_on_final_tree = self.num_speculative_steps + 1
        self.max_non_leaves_per_layer = self.max_total_draft_tokens

        self.num_capture_layers = getattr(vllm_config.speculative_config.draft_model_config.hf_config, "num_capture_layers", 3)
        self.hidden_states = torch.zeros(
            self.max_num_reqs,
            self.max_decoding_tokens,
            self.hidden_size * self.num_capture_layers,
            dtype=self.dtype,
            device=device,
        )
        # Hidden state read/write indices for persistent resource management (TRT-LLM style)
        self.hidden_states_read_indices = torch.zeros(self.max_num_tokens, dtype=torch.long, device=device)
        self.hidden_states_write_indices = torch.zeros(self.max_num_tokens, dtype=torch.long, device=device)
        
        # Resource slot management
        self.slot_start_indices = torch.zeros(self.max_num_reqs, dtype=torch.long, device=device)
        self.slot_seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self.is_first_draft = torch.ones(self.max_num_reqs, dtype=torch.bool, device=device)
        self.current_slots: torch.Tensor | None = None
        # Gather/scatter buffers for drafting loop
        self.draft_hidden_states = torch.zeros((self.max_num_reqs, self.hidden_size * self.num_capture_layers), dtype=self.dtype, device=device)
        self.temp_hidden_states = torch.zeros((self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device)
        self.hidden_states_indices_int64 = torch.zeros((self.max_num_tokens,), dtype=torch.int64, device=device)

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
        self._draft_top_k = torch.empty(
            self.max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self._draft_top_p = torch.empty(
            self.max_num_reqs,
            dtype=torch.float32,
            device=device,
        )
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self._num_valid_reqs = torch.zeros((), dtype=torch.int32, device=device)


        # Context drafting buffers
        self.eagle_seq_lens = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.eagle_ctx_lens = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.eagle_position_ids = torch.zeros((self.max_num_reqs * self.max_decoding_tokens,), dtype=torch.int32, device=device)
        self.eagle_output_ids = torch.zeros((self.max_num_reqs * self.max_decoding_tokens,), dtype=torch.int32, device=device)
        self.hidden_states_indices = torch.zeros((self.max_num_reqs * self.max_decoding_tokens,), dtype=torch.int32, device=device)
        self.last_token_indices = torch.zeros((self.max_num_reqs * self.max_non_leaves_per_layer,), dtype=torch.int32, device=device)
        self.num_last_token_indices = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.hidden_size_batch_level_starts = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.chunked_context_next_tokens = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)

        # Gen drafting buffers
        mask_blocks = (self.max_decoding_tokens + 31) // 32
        self.next_sequence_lengths = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.next_context_lengths = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.spec_dec_position_offsets = torch.zeros((self.max_num_reqs, self.max_decoding_tokens), dtype=torch.int32, device=device)
        self.spec_dec_packed_masks = torch.zeros((self.max_num_reqs, self.max_decoding_tokens, mask_blocks), dtype=torch.int32, device=device)
        self.spec_dec_gen_lengths = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.output_hidden_size_batch_starts_per_level = torch.zeros((self.max_path_len, self.max_num_reqs + 1), dtype=torch.int32, device=device)
        self.is_leaf_mask = torch.zeros((self.max_num_reqs, self.max_decoding_tokens), dtype=torch.int8, device=device)
        self.selected_draft_indices = torch.zeros((self.max_num_reqs, self.max_total_draft_tokens), dtype=torch.int32, device=device)
        self.selected_draft_pos_offsets = torch.zeros((self.max_num_reqs, self.max_total_draft_tokens), dtype=torch.int32, device=device)
        self.num_selected_draft_indices = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.selected_masks = torch.zeros((self.max_num_reqs, self.max_total_draft_tokens, mask_blocks), dtype=torch.int32, device=device)
        self.cum_sum_generation_lengths = torch.zeros((self.max_num_reqs + 1,), dtype=torch.int32, device=device)
        self.max_generation_length = torch.zeros((1,), dtype=torch.int32, device=device)
        self.non_leaves_in_level_offsets = torch.zeros((self.max_num_reqs, self.max_decoding_tokens), dtype=torch.int32, device=device)
        self.parent_non_leaf_in_level_offset = torch.zeros((self.max_num_reqs, self.max_decoding_tokens), dtype=torch.int32, device=device)
        self.input_hidden_size_batch_starts_per_level = torch.zeros((self.max_path_len, self.max_num_reqs + 1), dtype=torch.int32, device=device)

        # Slot Management (TRT-LLM style)
        self.req_id_to_slot = {}
        self.free_slots = list(range(self.max_num_reqs))

        self.best_path_ids = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.accepted_lens = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.accepted_tokens = torch.zeros((self.max_num_reqs, self.max_decoding_tokens), dtype=torch.int32, device=device)
        self.prev_draft_lens = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.prev_paths = torch.zeros((self.max_num_reqs, self.max_decoding_tokens, self.max_path_len), dtype=torch.int32, device=device)

        # Missing input buffers for prepare_gen
        self.next_draft_ids = torch.zeros((self.max_num_reqs, self.max_decoding_tokens), dtype=torch.int32, device=device)
        self.eagle_net0_sequence_lengths = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.prev_context_lengths = torch.zeros((self.max_num_reqs,), dtype=torch.int32, device=device)
        self.next_paths = torch.zeros((self.max_num_reqs, self.max_decoding_tokens, self.max_path_len), dtype=torch.int32, device=device)

        # Pointer and state buffers for logits assembly
        self.logits_ptrs = torch.zeros((self.max_num_reqs,), dtype=torch.int64, device=device)
        self.output_ids_ptrs = torch.zeros((self.max_num_reqs,), dtype=torch.int64, device=device)
        self.skip_decode = torch.zeros((self.max_num_reqs,), dtype=torch.bool, device=device)

        # Tree management (TRT-LLM style)
        self.tokens_gather_idx = torch.zeros((self.max_path_len, self.max_total_draft_tokens), dtype=torch.int32, device=device)
        self.top_k_list = torch.zeros((self.max_path_len, self.max_total_draft_tokens), dtype=torch.int32, device=device)
        self.draft_tokens_indices_cumsum = torch.zeros((self.max_path_len + 1,), dtype=torch.int32, device=device)

        self.cudagraph_manager = EagleCudaGraphManager(vllm_config, device)
        self.prefill_cudagraph_manager = EaglePrefillCudaGraphManager(
            vllm_config, device
        )
        self._decode_attn_metadata_cache: dict[int, dict[str, Any]] = {}
        self._draft_loop = _EagleSpeculatorDraftLoop(self)

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

    def _assign_slots(self, req_ids: list[str]) -> torch.Tensor:
        slots = []
        for rid in req_ids:
            if rid not in self.req_id_to_slot:
                if not self.free_slots:
                    # Should not normally happen if max_num_reqs is correctly set
                    raise RuntimeError("No free slots available for EAGLE3 Resource Manager")
                slot = self.free_slots.pop(0)
                self.req_id_to_slot[rid] = slot
                self.is_first_draft[slot] = True
                self.slot_seq_lens[slot] = 0
            slots.append(self.req_id_to_slot[rid])
        return torch.tensor(slots, dtype=torch.long, device=self.hidden_states.device)

    def _free_slots(self, finished_req_ids: list[str]) -> None:
        for rid in finished_req_ids:
            if rid in self.req_id_to_slot:
                slot = self.req_id_to_slot.pop(rid)
                self.free_slots.append(slot)
                self.is_first_draft[slot] = True
                self.slot_seq_lens[slot] = 0

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
        hidden_states: torch.Tensor,
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
                hidden_states=hidden_states,
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
        self._draft_loop(num_reqs, attn_metadata, num_tokens_across_dp)

    @torch.inference_mode()
    def _draft_loop_impl(
        self,
        num_reqs: int,
        attn_metadata: dict[str, Any],
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        is_capturing = torch.cuda.is_current_stream_capturing()
        
        # Step 0: Initial context input preparation
        self.prepare_ctx_inputs(num_reqs)
        
        self.input_buffers.input_ids[:num_reqs].copy_(self.eagle_output_ids[:num_reqs])
        self.input_buffers.positions[:num_reqs].copy_(self.eagle_position_ids[:num_reqs])

        # Persistent storage view
        hs_flat = self.hidden_states.view(-1, self.hidden_states.shape[-1])

        for step in range(0, self.num_speculative_steps):
            level_idx = step + 1
            if is_capturing:
                num_tokens = num_reqs if step == 0 else num_reqs * self.max_non_leaves_per_layer
            else:
                num_tokens = num_reqs if step == 0 else int(self.num_selected_draft_indices.sum().item())
            
            if step == 0:
                # Gather root hidden states (capture layers) from slots at index 0
                # parent_indices for root: self.current_slots * self.max_decoding_tokens
                root_indices = (self.current_slots * self.max_decoding_tokens).to(torch.long)
                # We need all capture layers for combining
                self.gather_tree_hidden_states(root_indices, self.draft_hidden_states[:num_reqs].unsqueeze(1), 
                                              width=self.hidden_size * self.num_capture_layers)
                # Combine them for the draft model's initial layer
                input_hs = self.model.combine_hidden_states(self.draft_hidden_states[:num_reqs])
            else:
                # Prepare inputs for the current generation step (reads from parents in previous levels)
                if is_capturing:
                    self.prepare_gen_inputs_padded(step, num_reqs)
                else:
                    self.prepare_gen_inputs(step, num_reqs)

                self.input_buffers.input_ids[:num_tokens].copy_(self.eagle_output_ids[:num_tokens])
                self.input_buffers.positions[:num_tokens].copy_(self.eagle_position_ids[:num_tokens])

                # Gather from tree nodes using indices calculated by prepare_gen_inputs
                indices = self.hidden_states_indices[:num_tokens].to(torch.long)
                # Slice to hidden_size for the draft model forward
                input_hs = hs_flat[indices][:, :self.hidden_size]

            last_hidden_states, hidden_states = self.run_model(
                num_tokens, attn_metadata, num_tokens_across_dp, hidden_states=input_hs
            )
            
            # Scatter/save the generated hidden states to the persistent buffer for next steps.
            if is_capturing:
                # Use fixed layout for CUDA graph
                starts = self.output_hidden_size_batch_starts_per_level[level_idx, :num_reqs].to(torch.long)
                local_offsets = torch.arange(self.max_non_leaves_per_layer, device=self.device)
                write_indices = (starts.unsqueeze(1) + local_offsets).view(-1)
                hs_flat[write_indices, :self.hidden_size] = hidden_states
            else:
                starts = self.output_hidden_size_batch_starts_per_level[level_idx, :num_reqs].to(torch.long)
                # For eager mode, we'd need to map the actually produced tokens to their slots
                # For now, we reuse the same logic if possible or finalize eagerly
                pass

            logits = self.model.compute_logits(last_hidden_states)
            
            if is_capturing:
                draft_tokens = ops.eagle_sample_argmax(logits)
            else:
                draft_tokens = torch.argmax(logits, dim=-1).to(torch.int32)
            
            self.extract_real_draft_tokens(step, num_reqs, draft_tokens)

    def capture_model(self) -> None:
        if self.num_speculative_steps == 1:
            return
        logger.info("Capturing model for Eagle speculator...")
        for i, builder in enumerate(self.attn_metadata_builders):
            print(f"DEBUG capture_model: builder[{i}] type={type(builder)}")
        if self.prefill_cudagraph_manager.cudagraph_sizes:
            self.prefill_cudagraph_manager.capture(
                model=self.model,
                input_buffers=self.input_buffers,
                input_hidden_states=self.temp_hidden_states,
                block_tables=self.block_tables,
                attn_metadata_builders=self.attn_metadata_builders,
                kv_cache_config=self.kv_cache_config,
                method=self.method,
            )
        self.cudagraph_manager.capture(
            self.generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_metadata_builders,
            self.kv_cache_config,
        )

    def _sync_prefill_metadata(self, input_batch: InputBatch) -> None:
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens

        self.input_buffers.query_start_loc.np[: num_reqs + 1] = (
            input_batch.query_start_loc_np
        )
        self.input_buffers.query_start_loc.np[num_reqs + 1 :] = num_tokens
        self.input_buffers.query_start_loc.copy_to_gpu()

        self.input_buffers.seq_lens[:num_reqs].copy_(input_batch.seq_lens)
        self.input_buffers.seq_lens[num_reqs:] = 0

        query_start_loc_gpu = self.input_buffers.query_start_loc.gpu[: num_reqs + 1]
        self.block_tables.compute_slot_mappings(
            query_start_loc_gpu, self.input_buffers.positions[:num_tokens]
        )

    def _get_decode_attn_metadata(
        self,
        num_reqs: int,
        query_start_loc_gpu: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        slot_mappings: torch.Tensor,
    ) -> dict[str, Any]:
        cached = self._decode_attn_metadata_cache.get(num_reqs)
        if cached is not None:
            return cached
        seq_lens_np = np.full(num_reqs, self.max_model_len, dtype=np.int32)
        block_tables = [x[:num_reqs] for x in self.block_tables.input_block_tables]
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
        self._decode_attn_metadata_cache[num_reqs] = attn_metadata
        return attn_metadata

    def prepare_ctx_inputs(self, num_reqs: int) -> None:
        ops.eagle_prepare_ctx_eagle_inputs(
            self.eagle_seq_lens,
            self.eagle_ctx_lens,
            self.eagle_output_ids,
            self.eagle_position_ids,
            self.hidden_states_indices,
            self.last_token_indices,
            self.num_last_token_indices,
            self.hidden_size_batch_level_starts,
            self.input_buffers.input_ids[:num_reqs],
            self.chunked_context_next_tokens[:num_reqs],
            self.input_buffers.seq_lens[:num_reqs],
            self.input_buffers.seq_lens[:num_reqs],
            self.accepted_tokens[:num_reqs],
            self.accepted_lens[:num_reqs],
            self.prev_draft_lens[:num_reqs],
            self.prev_paths[:num_reqs],
            self.best_path_ids[:num_reqs],
            self.max_path_len,
            self.max_decoding_tokens,
            self.max_non_leaves_per_layer,
        )

    def extract_real_draft_tokens(
        self,
        cur_draft_idx: int,
        batch_size: int,
        new_draft_tokens: torch.Tensor,
    ) -> None:
        num_tokens_expand_this_layer = 1 if cur_draft_idx == 0 else self.max_total_draft_tokens + 1
        ops.eagle_extract_real_draft_tokens(
            cur_draft_idx,
            self.num_speculative_steps,
            self.max_total_draft_tokens,
            self.max_decoding_tokens - 1, # max_top_k
            num_tokens_expand_this_layer,
            self.tokens_gather_idx[cur_draft_idx],
            self.top_k_list[cur_draft_idx],
            self.draft_tokens_indices_cumsum,
            new_draft_tokens,
            self.accepted_tokens[:batch_size], # buffer
        )

    def prepare_gen_inputs(self, level_idx: int, num_reqs: int) -> None:
        ops.eagle_prepare_gen_eagle_inputs(
            self.next_sequence_lengths,
            self.next_context_lengths,
            self.eagle_output_ids,
            self.eagle_position_ids,
            self.spec_dec_gen_lengths,
            self.spec_dec_position_offsets,
            self.spec_dec_packed_masks,
            self.hidden_states_indices,
            self.last_token_indices,
            self.num_last_token_indices,
            self.output_hidden_size_batch_starts_per_level,
            self.is_leaf_mask,
            self.selected_draft_indices,
            self.selected_draft_pos_offsets,
            self.num_selected_draft_indices,
            self.selected_masks,
            self.cum_sum_generation_lengths,
            self.max_generation_length,
            self.non_leaves_in_level_offsets,
            self.parent_non_leaf_in_level_offset,
            self.next_draft_ids,
            self.eagle_net0_sequence_lengths,
            self.prev_context_lengths,
            self.input_hidden_size_batch_starts_per_level,
            self.next_paths,
            level_idx,
            self.max_path_len,
            self.max_decoding_tokens,
            self.max_non_leaves_per_layer,
        )

    def prepare_gen_inputs_padded(self, level_idx: int, num_reqs: int) -> None:
        ops.eagle_prepare_gen_eagle_inputs_padded(
            self.next_sequence_lengths,
            self.next_context_lengths,
            self.eagle_output_ids,
            self.eagle_position_ids,
            self.spec_dec_gen_lengths,
            self.spec_dec_position_offsets,
            self.spec_dec_packed_masks,
            self.hidden_states_indices,
            self.last_token_indices,
            self.num_last_token_indices,
            self.output_hidden_size_batch_starts_per_level,
            self.is_leaf_mask,
            self.selected_draft_indices,
            self.selected_draft_pos_offsets,
            self.num_selected_draft_indices,
            self.selected_masks,
            self.cum_sum_generation_lengths,
            self.max_generation_length,
            self.non_leaves_in_level_offsets,
            self.parent_non_leaf_in_level_offset,
            self.next_draft_ids,
            self.eagle_net0_sequence_lengths,
            self.prev_context_lengths,
            self.input_hidden_size_batch_starts_per_level,
            self.next_paths,
            level_idx,
            self.max_path_len,
            self.max_decoding_tokens,
            self.max_non_leaves_per_layer,
        )

    def compact_kv_cache(self, kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        ops.eagle_kv_cache_compact(
            kv_cache,
            slot_mapping,
            self.accepted_lens,
            self.max_decoding_tokens,
        )

    def gather_tree_hidden_states(self, parent_indices: torch.Tensor, output_hidden_states: torch.Tensor, width: int | None = None) -> None:
        # self.hidden_states is 3D (num_reqs, max_decoding_tokens, hidden)
        if width is None:
            width = self.hidden_size
        ops.eagle_tree_gather_hidden_states(
            self.hidden_states,
            parent_indices,
            output_hidden_states,
            width,
        )

    def update_dynamic_tree_scores(self, cur_log_probs: torch.Tensor, prev_layer_scores: torch.Tensor) -> None:
        ops.eagle_update_scores(
            cur_log_probs,
            prev_layer_scores,
            self.max_non_leaves_per_layer,
        )

    def _prepare_hidden_states_indices(self, input_batch: InputBatch, slots: torch.Tensor) -> None:
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens_after_padding
        qsl = input_batch.query_start_loc_np
        
        # We populate self.hidden_states_write_indices with flat indices into 3D buffer (slot, token_idx)
        write_indices = torch.zeros(num_tokens, dtype=torch.long, device=self.hidden_states.device)
        for i in range(num_reqs):
            slot = int(slots[i])
            start, end = qsl[i], qsl[i+1]
            count = end - start
            # Each slot has a reserved window of self.max_decoding_tokens
            base_idx = slot * self.max_decoding_tokens
            write_indices[start:end] = torch.arange(base_idx, base_idx + count, device=self.hidden_states.device)
            self.slot_seq_lens[slot] = count
            
        self.hidden_states_write_indices[:num_tokens].copy_(write_indices)
        self.hidden_states_read_indices[:num_tokens].copy_(write_indices)

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
        slots = self._assign_slots(input_batch.req_ids)
        num_tokens = input_batch.num_tokens_after_padding
        
        if aux_hidden_states:
            assert self.method == "eagle3"
            cat_hidden_states = torch.cat(aux_hidden_states, dim=-1)
            # persistent storage with indexing (Task Set E/F parity)
            self._prepare_hidden_states_indices(input_batch, self.current_slots)
            # Use flattened view to use 1D index_copy_
            hs_flat = self.hidden_states.view(-1, self.hidden_states.shape[-1])
            hs_flat.index_copy_(0, self.hidden_states_write_indices[:num_tokens], cat_hidden_states)
            
            # Combine auxiliary hidden states and store for unified gathering in later steps
            hidden_states = self.model.combine_hidden_states(cat_hidden_states)
            # We save combined roots into the first hidden_size part of slot 0
            hs_flat[self.hidden_states_write_indices[:num_tokens], :self.hidden_size] = hidden_states
        else:
            hidden_states = last_hidden_states
            # Fallback for non-EAGLE3 Or single-layer
            # Initialize slot 0 with current hidden states
            self.hidden_states[self.current_slots, 0, :self.hidden_size].copy_(hidden_states)

        # Get the input ids and last token indices for the speculator.
        last_token_indices = prepare_eagle_inputs(
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
        )
        
        # Pre-fill starting offsets for the draft loop
        self.input_hidden_size_batch_starts_per_level.fill_(0)
        for i, slot in enumerate(self.current_slots):
            # Request i starts at slot * max_decoding_tokens in the flat hidden_states buffer
            self.input_hidden_size_batch_starts_per_level[0, i] = int(slot) * self.max_decoding_tokens

        prefill_cudagraph_size = (
            self.prefill_cudagraph_manager.get_cudagraph_size(num_tokens)
            if self.prefill_cudagraph_manager.cudagraph_sizes
            else None
        )
        if prefill_cudagraph_size == num_tokens:
            self._sync_prefill_metadata(input_batch)
            last_hidden_states, hidden_states = self.prefill_cudagraph_manager.run(
                num_tokens
            )
        else:
            # Prefill: Run the eagle speculator with eager mode.
            last_hidden_states, hidden_states = self.run_model(
                num_tokens,
                input_batch.attn_metadata,
                num_tokens_across_dp=None,  # FIXME
                hidden_states=hidden_states,
            )
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        num_reqs = input_batch.num_reqs
        cu_num_logits = input_batch.cu_num_logits[:num_reqs]
        # NOTE: For draft sampling, we default to temperature-only, but we
        # optionally apply top-k/top-p when configured to improve acceptance.
        temperature = self.temperature[:num_reqs]
        seeds = self.seeds[:num_reqs]
        pos = self.input_buffers.positions[:num_reqs]
        # Gather the values and copy them to the pre-allocated buffers.
        torch.gather(sampling_metadata.temperature, 0, cu_num_logits, out=temperature)
        torch.gather(sampling_metadata.seeds, 0, cu_num_logits, out=seeds)
        torch.gather(input_batch.positions, 0, last_token_indices, out=pos)
        top_k = self._draft_top_k[:num_reqs]
        if sampling_metadata.top_k is not None:
            torch.gather(
                sampling_metadata.top_k.to(dtype=top_k.dtype),
                0,
                cu_num_logits,
                out=top_k,
            )
        else:
            top_k.fill_(self.vocab_size)
        top_p = self._draft_top_p[:num_reqs]
        if sampling_metadata.top_p is not None:
            torch.gather(sampling_metadata.top_p, 0, cu_num_logits, out=top_p)
        else:
            top_p.fill_(1.0)
        logits = apply_top_k_top_p(logits, top_k, top_p)
        # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
        # used for draft and target sampling.
        draft_tokens = gumbel_sample(
            logits, temperature, seeds, pos + 1, apply_temperature=True
        )
        if self.num_speculative_steps == 1:
            # Early exit.
            return draft_tokens.view(-1, 1)

        cudagraph_size = self.cudagraph_manager.get_cudagraph_size(num_reqs)
        decode_num_reqs = cudagraph_size or num_reqs
        if decode_num_reqs < num_reqs:
            decode_num_reqs = num_reqs
        pad_token_id = self.vllm_config.model_config.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

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
            pad_to_num_reqs=decode_num_reqs,
            pad_token_id=pad_token_id,
        )
        query_start_loc = self.input_buffers.query_start_loc
        query_start_loc_gpu = query_start_loc.gpu[: decode_num_reqs + 1]
        decode_pos = self.input_buffers.positions[:decode_num_reqs]
        slot_mappings = self.block_tables.compute_slot_mappings(
            query_start_loc_gpu, decode_pos
        )
        if decode_num_reqs > num_reqs:
            slot_mappings[:, num_reqs:decode_num_reqs].fill_(PAD_SLOT_ID)
        self._num_valid_reqs.fill_(num_reqs)

        if cudagraph_size is not None:
            # Run CUDA graph.
            self.cudagraph_manager.run(cudagraph_size)
            return self.draft_tokens[:num_reqs]

        # Run eager mode.
        query_start_loc.np[: num_reqs + 1] = np.arange(num_reqs + 1)
        query_start_loc_cpu = query_start_loc.cpu[: num_reqs + 1]
        # FIXME(woosuk): This is UNSAFE!!
        attn_metadata = self._get_decode_attn_metadata(
            num_reqs,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            slot_mappings=slot_mappings,
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
    pad_to_num_reqs,
    pad_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        # Compute query_start_loc. Pad it with the last query_start_loc
        # for CUDA graphs.
        pad_to_num_reqs = tl.minimum(pad_to_num_reqs, max_num_reqs)
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block <= pad_to_num_reqs, block, pad_to_num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            pad_seq_len = tl.where(block < pad_to_num_reqs, 1, 0)
            tl.store(seq_lens_ptr + block, pad_seq_len, mask=mask)
        # Pad input_ids/positions for CUDA graphs.
        for i in range(req_idx, pad_to_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < pad_to_num_reqs
            tl.store(input_ids_ptr + block, pad_token_id, mask=mask)
            tl.store(positions_ptr + block, 0, mask=mask)
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
    pad_to_num_reqs: int | None = None,
    pad_token_id: int = 0,
):
    num_reqs = draft_tokens.shape[0]
    if pad_to_num_reqs is None:
        pad_to_num_reqs = num_reqs
    elif pad_to_num_reqs < num_reqs:
        pad_to_num_reqs = num_reqs
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
        pad_to_num_reqs,
        pad_token_id,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _mask_slot_mappings_kernel(
    slot_mappings_ptr,
    slot_mappings_stride,
    num_tokens,
    num_valid_tokens_ptr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_tokens
    num_valid_tokens = tl.load(num_valid_tokens_ptr)
    invalid = offsets >= num_valid_tokens
    ptr = slot_mappings_ptr + group_id * slot_mappings_stride + offsets
    values = tl.load(ptr, mask=mask, other=PAD_ID)
    values = tl.where(invalid, PAD_ID, values)
    tl.store(ptr, values, mask=mask)


def mask_slot_mappings(
    slot_mappings: torch.Tensor,
    num_tokens: int,
    num_valid_tokens: torch.Tensor,
) -> None:
    num_groups = slot_mappings.shape[0]
    grid = (num_groups, triton.cdiv(num_tokens, 1024))
    _mask_slot_mappings_kernel[grid](
        slot_mappings,
        slot_mappings.stride(0),
        num_tokens,
        num_valid_tokens,
        PAD_ID=PAD_SLOT_ID,
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
