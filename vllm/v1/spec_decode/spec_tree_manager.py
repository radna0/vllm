# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from itertools import accumulate
from typing import List

import torch


class SpecTreeManager:
    """Static tree precompute for EAGLE3 tree drafting."""

    def __init__(
        self,
        tree_choices: list[tuple[int, ...]],
        max_batch_size: int,
        device: torch.device,
    ) -> None:
        if not tree_choices:
            raise ValueError("speculative_token_tree must not be empty.")

        self.tree_choices = tree_choices
        self.tree_depth = max(len(path) for path in tree_choices)

        self.num_drafts_per_level: list[int] = [0] * self.tree_depth
        for node in tree_choices:
            self.num_drafts_per_level[len(node) - 1] += 1
        self.cu_drafts_per_level: list[int] = [self.num_drafts_per_level[0]]
        for level in range(1, self.tree_depth):
            self.cu_drafts_per_level.append(
                self.cu_drafts_per_level[-1] + self.num_drafts_per_level[level]
            )
        self.total_drafts = self.cu_drafts_per_level[-1]

        self.index_mapping = {tuple(path): i + 1 for i, path in enumerate(tree_choices)}
        self.nodes_per_level: list[list[int]] = [
            [] for _ in range(self.tree_depth + 1)
        ]
        self.nodes_per_level[0].append(0)
        for path in tree_choices:
            self.nodes_per_level[len(path)].append(self.index_mapping[tuple(path)])

        child_nodes: list[list[int]] = [[] for _ in range(self.total_drafts + 1)]
        for path in tree_choices:
            node_idx = self.index_mapping[tuple(path)]
            if len(path) == 1:
                parent_idx = 0
            else:
                parent_idx = self.index_mapping[tuple(path[:-1])]
            child_nodes[parent_idx].append(node_idx)
        self.child_nodes = child_nodes

        self.children_per_parent_per_level: list[torch.Tensor] = []
        self.parent_indices_per_level: list[torch.Tensor | None] = []
        self.child_indices_per_level: list[torch.Tensor | None] = []
        self.max_topk_per_level: list[int] = []
        self.top_k_list: list[torch.Tensor] = []
        for level in range(self.tree_depth):
            parent_nodes = self.nodes_per_level[level]
            child_counts: list[int] = []
            parent_indices: list[int] = []
            child_indices: list[int] = []
            for parent_pos, node_idx in enumerate(parent_nodes):
                num_children = len(child_nodes[node_idx])
                child_counts.append(num_children)
                for child_pos in range(num_children):
                    parent_indices.append(parent_pos)
                    child_indices.append(child_pos)
            child_counts_tensor = torch.tensor(
                child_counts, dtype=torch.int32, device=device
            )
            self.children_per_parent_per_level.append(child_counts_tensor)
            top_k_list_cpu = torch.tensor(
                child_counts, dtype=torch.int32, device="cpu", pin_memory=True
            )
            self.top_k_list.append(top_k_list_cpu)
            expected_children = self.num_drafts_per_level[level]
            if sum(child_counts) != expected_children:
                raise ValueError(
                    "speculative_token_tree is inconsistent: expected "
                    f"{expected_children} children at level {level + 1}, "
                    f"got {sum(child_counts)}"
                )
            if parent_indices:
                self.parent_indices_per_level.append(
                    torch.tensor(parent_indices, dtype=torch.int64, device=device)
                )
                self.child_indices_per_level.append(
                    torch.tensor(child_indices, dtype=torch.int64, device=device)
                )
            else:
                self.parent_indices_per_level.append(None)
                self.child_indices_per_level.append(None)
            self.max_topk_per_level.append(max(child_counts) if child_counts else 0)
        self.root_children_count = int(self.children_per_parent_per_level[0][0].item())
        self.top_k_list_cuda = [t.to(device=device) for t in self.top_k_list]
        self.max_top_k = 0
        for top_k_tensor in self.top_k_list:
            if top_k_tensor.numel():
                self.max_top_k = max(
                    self.max_top_k, int(top_k_tensor.max().item())
                )

        self.level_offsets: list[int] = [0]
        for level in range(1, self.tree_depth):
            self.level_offsets.append(self.cu_drafts_per_level[level - 1])

        self.tree_draft_pos_offsets = torch.arange(
            1, self.total_drafts + 1, device=device, dtype=torch.int32
        ).repeat(max_batch_size, 1)

        arange = torch.arange(max_batch_size + 1, device=device, dtype=torch.int32)
        self.query_start_loc: list[torch.Tensor] = []
        for total_drafts in self.cu_drafts_per_level:
            self.query_start_loc.append(total_drafts * arange)

        self.spec_dec_mask_matrix = self._build_mask_matrix(device)
        self.spec_dec_packed_mask = self._pack_mask(self.spec_dec_mask_matrix)
        self.spec_dec_position_offsets = self._build_position_offsets(device)
        self.spec_dec_packed_mask_for_drafter_model = self._build_drafter_packed_mask(
            device
        )
        self.tokens_gather_idx_for_drafter_model = (
            self._build_tokens_gather_idx_for_drafter_model(device)
        )
        self.draft_tokens_indices_cumsum = self._build_draft_tokens_indices_cumsum(
            device
        )
        self.hidden_states_read_indices_offset_for_drafter_model = (
            self._build_hidden_states_read_indices_offset_for_drafter_model(device)
        )
        self.draft_packed_masks: list[torch.Tensor] = []
        self.draft_position_offsets: list[torch.Tensor] = []
        for total_drafts in self.cu_drafts_per_level:
            if total_drafts <= 0:
                self.draft_packed_masks.append(
                    torch.empty((0, 0), dtype=torch.int32, device=device)
                )
                self.draft_position_offsets.append(
                    torch.empty((0,), dtype=torch.int32, device=device)
                )
                continue
            mask = self.spec_dec_mask_matrix[
                1 : 1 + total_drafts, 1 : 1 + total_drafts
            ]
            self.draft_packed_masks.append(self._pack_mask(mask))
            self.draft_position_offsets.append(
                self.spec_dec_position_offsets[1 : 1 + total_drafts]
            )

    def _build_drafter_packed_mask(self, device: torch.device) -> torch.Tensor:
        tree_len = self.total_drafts + 1
        tmp_mask = torch.zeros(
            (tree_len, tree_len), dtype=torch.int32, device=device
        )
        if tree_len > 1:
            tmp_mask[:-1, :-1] = self.spec_dec_mask_matrix[1:, 1:]
        return self._pack_mask(tmp_mask)

    def _build_tokens_gather_idx_for_drafter_model(
        self, device: torch.device
    ) -> list[torch.Tensor]:
        tokens_gather_idx: list[torch.Tensor] = []
        tokens_gather_idx.append(
            torch.tensor([0], dtype=torch.int32, device=device)
        )
        for cur_layer_nodes in self.nodes_per_level[1:]:
            gather_list = [
                node - 1
                for node in cur_layer_nodes
                if self.child_nodes[node]
            ]
            tokens_gather_idx.append(
                torch.tensor(gather_list, dtype=torch.int32, device=device)
            )
        return tokens_gather_idx

    def _build_draft_tokens_indices_cumsum(
        self, device: torch.device
    ) -> torch.Tensor:
        num_nodes_per_layer = [0]
        num_nodes_per_layer.extend(
            len(node_list) for node_list in self.nodes_per_level[1:]
        )
        return torch.tensor(
            list(accumulate(num_nodes_per_layer)),
            dtype=torch.int32,
            device=device,
        )

    def _build_hidden_states_read_indices_offset_for_drafter_model(
        self, device: torch.device
    ) -> torch.Tensor:
        offsets = torch.zeros(
            (self.total_drafts + 1), dtype=torch.int32, device=device
        )
        parent_nodes: list[int] = []
        for path in self.tree_choices:
            if len(path) == 1:
                parent_nodes.append(0)
            else:
                parent_nodes.append(self.index_mapping[tuple(path[:-1])])
        if parent_nodes:
            offsets[: self.total_drafts] = torch.tensor(
                parent_nodes, dtype=torch.int32, device=device
            )
        return offsets

    def _build_mask_matrix(self, device: torch.device) -> torch.Tensor:
        tree_len = self.total_drafts + 1
        mask = torch.eye(tree_len, dtype=torch.int32, device=device)
        mask[:, 0] = 1
        for i, path in enumerate(self.tree_choices):
            if len(path) == 1:
                continue
            row = i + 1
            for depth in range(len(path) - 1):
                ancestor = self.index_mapping[tuple(path[: depth + 1])]
                mask[row, ancestor] = 1
        return mask

    def _pack_mask(self, mask: torch.Tensor) -> torch.Tensor:
        tree_len = mask.shape[0]
        num_blocks = math.ceil(tree_len / 32)
        packed = torch.zeros(
            (tree_len, num_blocks), dtype=torch.int32, device=mask.device
        )
        for block_idx in range(num_blocks):
            start = block_idx * 32
            end = min(start + 32, tree_len)
            block_bits = mask[:, start:end]
            weights = torch.pow(
                2,
                torch.arange(end - start, dtype=torch.int32, device=mask.device),
            )
            packed[:, block_idx] = torch.sum(block_bits * weights, dim=-1)
        return packed

    def _build_position_offsets(self, device: torch.device) -> torch.Tensor:
        tree_len = self.total_drafts + 1
        offsets = torch.zeros(tree_len, dtype=torch.int32, device=device)
        offsets[0] = 0
        for i, path in enumerate(self.tree_choices):
            offsets[i + 1] = len(path)
        return offsets
