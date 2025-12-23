#!/bin/bash
set -e

BUILD_ artifact="cmake-build-release/vllm/_C.abi3.so"
TARGET_DIR="/workspace/aimo/LM/vllm/vllm"

echo "Checking for build artifact..."
if [ ! -f "$BUILD_artifact" ]; then
    echo "Error: Build artifact not found at $BUILD_artifact"
    echo "Compilation might still be running or failed."
    exit 1
fi

echo "Deploying _C.abi3.so to $TARGET_DIR..."
cp "$BUILD_artifact" "$TARGET_DIR/"

echo "Verifying import..."
python -c "import vllm._C; print('Successfully imported vllm._C')"

echo "Deployment and Verification Complete!"
