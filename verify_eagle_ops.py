import sys

try:
    import vllm._custom_ops as ops
except ImportError:
    print("Failed to import vllm._custom_ops")
    sys.exit(1)

required_ops = [
    "eagle_prepare_ctx_eagle_inputs",
    "eagle_extract_real_draft_tokens",
    "eagle_update_path",
    "eagle_update_scores",
    "eagle_sample_argmax",
]

missing = []
for op_name in required_ops:
    if not hasattr(ops, op_name):
        missing.append(op_name)

if missing:
    print(f"Missing EAGLE ops: {missing}")
    sys.exit(1)
else:
    print("All required EAGLE ops are present.")
    sys.exit(0)
