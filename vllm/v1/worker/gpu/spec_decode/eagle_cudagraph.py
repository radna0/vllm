# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    capture_graphs,
    get_cudagraph_sizes,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.dp_utils import make_num_tokens_across_dp
from vllm.v1.worker.gpu.input_batch import InputBuffers


class EagleCudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device = device

        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None

        cudagraph_mode: CUDAGraphMode
        if self.compilation_config.cudagraph_mode is None:
            cudagraph_mode = CUDAGraphMode.NONE
        else:
            cudagraph_mode = self.compilation_config.cudagraph_mode
            if cudagraph_mode == CUDAGraphMode.FULL:
                # NOTE(woosuk): For Eagle, we only use CUDA graphs for decode.
                cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY

        self.cudagraph_mode = cudagraph_mode

        self.cudagraph_sizes = get_cudagraph_sizes(
            self.compilation_config.cudagraph_capture_sizes,
            self.max_num_reqs,
            self.max_num_tokens,
            self.cudagraph_mode,
        )

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()

    def get_cudagraph_size(self, num_tokens: int) -> int | None:
        return self.cudagraph_sizes.get(num_tokens)

    def capture_graph(
        self,
        num_tokens: int,
        generate_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        num_reqs = min(num_tokens, self.max_num_reqs)
        attn_metadata = prepare_inputs_to_capture(
            num_reqs,
            num_tokens,
            input_buffers,
            block_tables,
            attn_metadata_builders,
            self.max_model_len,
            kv_cache_config,
        )
        num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, num_tokens)

        # Warm up.
        generate_fn(num_tokens, attn_metadata, num_tokens_across_dp)

        # Capture the graph.
        assert num_tokens not in self.graphs
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.pool):
            generate_fn(num_tokens, attn_metadata, num_tokens_across_dp)
        self.graphs[num_tokens] = graph

    @torch.inference_mode()
    def capture(
        self,
        generate_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        capture_graphs(
            self.cudagraph_sizes,
            self.device,
            self.capture_graph,
            generate_fn=generate_fn,
            input_buffers=input_buffers,
            block_tables=block_tables,
            attn_metadata_builders=attn_metadata_builders,
            kv_cache_config=kv_cache_config,
        )

    def run(self, num_tokens: int) -> None:
        assert num_tokens in self.graphs
        self.graphs[num_tokens].replay()


class EaglePrefillCudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device = device

        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None

        cudagraph_mode: CUDAGraphMode
        if self.compilation_config.cudagraph_mode is None:
            cudagraph_mode = CUDAGraphMode.NONE
        else:
            cudagraph_mode = self.compilation_config.cudagraph_mode
            if cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
                cudagraph_mode = CUDAGraphMode.FULL
        self.cudagraph_mode = cudagraph_mode

        self.cudagraph_sizes = get_cudagraph_sizes(
            self.compilation_config.cudagraph_capture_sizes,
            self.max_num_reqs,
            self.max_num_tokens,
            self.cudagraph_mode,
        )

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.last_hidden_states: torch.Tensor | None = None
        self.hidden_states: torch.Tensor | None = None

    def get_cudagraph_size(self, num_tokens: int) -> int | None:
        return self.cudagraph_sizes.get(num_tokens)

    def capture_graph(
        self,
        num_tokens: int,
        model: torch.nn.Module,
        input_buffers: InputBuffers,
        input_hidden_states: torch.Tensor,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
        method: str,
    ) -> None:
        num_reqs = min(num_tokens, self.max_num_reqs)
        input_ids = input_buffers.input_ids[:num_tokens]
        positions = input_buffers.positions[:num_tokens]
        attn_metadata = prepare_inputs_to_capture(
            num_reqs,
            num_tokens,
            input_buffers,
            block_tables,
            attn_metadata_builders,
            self.max_model_len,
            kv_cache_config,
        )
        num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, num_tokens)

        # Warm up.
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
        ):
            outputs = model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=input_hidden_states[:num_tokens],
            )
            if method == "mtp":
                last_hidden_states = outputs
                hidden_states = outputs
            else:
                last_hidden_states, hidden_states = outputs
            if self.last_hidden_states is None:
                self.last_hidden_states = torch.empty_like(last_hidden_states)
            if self.hidden_states is None:
                self.hidden_states = torch.empty_like(hidden_states)

        # Capture the graph.
        assert num_tokens not in self.graphs
        graph = torch.cuda.CUDAGraph()
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=num_tokens_across_dp,
            ),
            torch.cuda.graph(graph, self.pool),
        ):
            outputs = model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=input_hidden_states[:num_tokens],
            )
            if method == "mtp":
                last_hidden_states = outputs
                hidden_states = outputs
            else:
                last_hidden_states, hidden_states = outputs
            assert self.last_hidden_states is not None
            assert self.hidden_states is not None
            self.last_hidden_states[:num_tokens] = last_hidden_states
            self.hidden_states[:num_tokens] = hidden_states
        self.graphs[num_tokens] = graph

    @torch.inference_mode()
    def capture(
        self,
        model: torch.nn.Module,
        input_buffers: InputBuffers,
        input_hidden_states: torch.Tensor,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
        method: str,
    ) -> None:
        capture_graphs(
            self.cudagraph_sizes,
            self.device,
            self.capture_graph,
            model=model,
            input_buffers=input_buffers,
            input_hidden_states=input_hidden_states,
            block_tables=block_tables,
            attn_metadata_builders=attn_metadata_builders,
            kv_cache_config=kv_cache_config,
            method=method,
        )

    def run(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert num_tokens in self.graphs
        self.graphs[num_tokens].replay()
        assert self.last_hidden_states is not None
        assert self.hidden_states is not None
        return (
            self.last_hidden_states[:num_tokens],
            self.hidden_states[:num_tokens],
        )
