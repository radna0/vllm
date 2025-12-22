import torch
import vllm.envs

# Force load of custom ops
from vllm.engine.arg_utils import EngineArgs

print("Checking for SuffixDecodingTree op...")
if hasattr(torch.ops.vllm, "SuffixDecodingTree"):
    print("SUCCESS: SuffixDecodingTree found!")

    # Simple functional test
    try:
        max_depth = 5
        tree = torch.ops.vllm.SuffixDecodingTree(max_depth)
        print("Initialized SuffixDecodingTree")

        # Test extend
        seq_id = 1
        tokens = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        tree.extend(seq_id, tokens)
        print("Extended sequence 1")

        # Test speculate
        context = torch.tensor([3, 4], dtype=torch.int32)
        draft = tree.speculate(
            context,
            max_spec_tokens=3,
            max_spec_factor=1.0,
            max_spec_offset=0.0,
            min_token_prob=0.0,
            use_tree_spec=False,
        )
        print(f"Speculated tokens: {draft.token_ids}")
        assert len(draft.token_ids) > 0
        print("Speculation successful")

    except Exception as e:
        print(f"FAILURE during functional test: {e}")
        exit(1)

else:
    print("FAILURE: SuffixDecodingTree NOT found in torch.ops.vllm")
    exit(1)
