#!/bin/bash
# inject.sh: Hot-patch installed vLLM with local drift logic

if [ -z "$1" ]; then
    echo "Usage: bash inject.sh <TARGET_VLLM_DIR>"
    echo "Example: bash inject.sh /usr/local/lib/python3.11/site-packages/vllm"
    exit 1
fi

TARGET_DIR=$1
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/vllm"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR not found!"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory $TARGET_DIR not found!"
    exit 1
fi

echo "Injecting vLLM drift from $SOURCE_DIR to $TARGET_DIR..."

# Sync files
cp -rv "$SOURCE_DIR"/* "$TARGET_DIR"/

# Cleanup target pycache
find "$TARGET_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "✓ Injection complete."
