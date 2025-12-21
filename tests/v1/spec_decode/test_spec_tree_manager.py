# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.spec_tree_manager import SpecTreeManager


def test_spec_tree_manager_static_tree() -> None:
    tree_choices = [(0,), (1,), (0, 0), (0, 1), (1, 0)]
    mgr = SpecTreeManager(
        tree_choices=tree_choices,
        max_batch_size=4,
        device=torch.device("cpu"),
    )

    assert mgr.total_drafts == len(tree_choices)
    assert mgr.num_drafts_per_level == [2, 3]
    assert mgr.root_children_count == 2

    level0_children = mgr.children_per_parent_per_level[0].tolist()
    level1_children = mgr.children_per_parent_per_level[1].tolist()
    assert level0_children == [2]
    assert level1_children == [2, 1]

    assert mgr.parent_indices_per_level[1].tolist() == [0, 0, 1]
    assert mgr.child_indices_per_level[1].tolist() == [0, 1, 0]

    assert mgr.spec_dec_mask_matrix.shape == (len(tree_choices) + 1, len(tree_choices) + 1)
    assert mgr.spec_dec_packed_mask.shape[0] == len(tree_choices) + 1
    assert len(mgr.draft_packed_masks) == len(mgr.cu_drafts_per_level)
    assert mgr.draft_packed_masks[0].shape[0] == mgr.cu_drafts_per_level[0]
