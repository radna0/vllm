import pytest
import torch
import torch.nn.functional as F

# Depending on how it's exposed. In torch_bindings.cpp we used ops.def("tree_speculative_sampling_target_only", ...)
# and register_extension(vllm_C) usually exposes it under vllm._C or torch.ops.vllm
# vllm.utils usually imports _C.
# Let's assume it's under torch.ops.vllm for now as per `ops.impl` in torch_bindings.cpp (using Library("vllm", ...))?
# Wait, torch_bindings.cpp uses `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)`.
# The TORCH_EXTENSION_NAME is usually `vllm._C`.
# However, standard vllm uses `torch.library.Library("vllm", "FRAGMENT")` or similar?
# Let's check torch_bindings.cpp again. It defines `TORCH_LIBRARY_EXPAND` as `TORCH_LIBRARY`.
# And setup.py defines the extension name.
# Usually custom ops are accessed via `torch.ops.vllm.<op_name>` if the library name passed to TORCH_LIBRARY is "vllm".
# In vllm/csrc/torch_bindings.cpp: `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)`.
# If `TORCH_EXTENSION_NAME` is `vllm._C`, then it registers ops under that namespace?
# Actually, `TORCH_LIBRARY` takes a namespace. If `TORCH_EXTENSION_NAME` is passed, it uses that.
# In `setup.py`, the extension name is `vllm._C`.
# But `torch.ops` usually works with namespaces like `vllm`.
# If the library name is `vllm._C`, the ops might be `torch.ops.vllm._C`? That sounds wrong.
# Let's look at `torch_bindings.cpp` again.
# The user's code for `cutlass_scaled_fp4_mm` used `ops.def("cutlass_scaled_fp4_mm", ...)`
# and then python uses `import vllm._C` and calls `vllm._C.ops.cutlass_scaled_fp4_mm`?
# Or `torch.ops.vllm.cutlass_scaled_fp4_mm`?
# vllm/_custom_ops.py often wraps these.

from vllm import _custom_ops as ops


def tree_speculative_sampling_target_only(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,
):
    torch.ops.vllm.tree_speculative_sampling_target_only(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
        deterministic,
    )


test_cases = [
    (
        1,
        1,
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
    (
        0,  # threshold_single
        0,  # threshold_acc
        [1, 2, 18, -1, -1, -1, 11, -1, -1, -1, 12, 18],
        [[0, 1, 2, -1], [6, 10, 11, -1]],
        [2, 2],
    ),
]


@pytest.mark.parametrize(
    "threshold_single, threshold_acc, expected_predicts, expected_accept_index, expected_accept_token_num",
    test_cases,
)
def test_tree_speculative_sampling_target_only(
    threshold_single,
    threshold_acc,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
):
    """
    Tests the tree_speculative_sampling_target_only function using Pytest parameterization.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"

    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
        device=device,
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device=device)
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10

    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i, j]) < 10:
                target_logits[i, j, 18] = 10

    temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
    bs, num_draft_tokens = candidates.shape
    num_spec_step = len(expected_accept_index[0])
    predict_shape = (len(expected_predicts),)

    predicts = torch.full(predict_shape, -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, num_spec_step), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device=device)

    expanded_temperature = temperatures.unsqueeze(1).unsqueeze(1)
    target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
    draft_probs = torch.full_like(target_probs, 0, dtype=torch.float32, device=device)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)
    coins_for_final_sampling = torch.rand(bs, device=device).to(torch.float32)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    assert (
        predicts.tolist() == expected_predicts
    ), f"Predicts mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_index.tolist() == expected_accept_index
    ), f"Accept index mismatch for thresholds ({threshold_single}, {threshold_acc})"
    assert (
        accept_token_num.tolist() == expected_accept_token_num
    ), f"Accept token num mismatch for thresholds ({threshold_single}, {threshold_acc})"


if __name__ == "__main__":
    pytest.main([__file__])
