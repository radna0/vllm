#!/bin/bash
# install_kaggle.sh: Kaggle-optimized plugin installation

PLUGINS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_BASE="/kaggle/working"

# Check if we are in Kaggle
if [ ! -d "/kaggle" ]; then
    echo "Warning: Not in a Kaggle environment. Standard target will be used."
fi

# Find all directories with setup.py
find "$PLUGINS_DIR" -maxdepth 2 -name "setup.py" | while read setup_file; do
    plugin_path=$(dirname "$setup_file")
    plugin_name=$(basename "$plugin_path")
    
    echo "=========================================================="
    echo "Kaggle Installing: $plugin_name"
    echo "=========================================================="
    
    cd "$plugin_path"
    
    # Clean previous builds
    rm -rf build/ dist/ *.egg-info 2>/dev/null
    find . -name "*.so" -delete
    
    # Use H100 arch by default for Kaggle H100 notebooks
    export TORCH_CUDA_ARCH_LIST="9.0"
    export MAX_JOBS="1"
    
    # Build and install to /kaggle/working
    pip install . --no-build-isolation --target="$TARGET_BASE"
    
    if [ $? -eq 0 ]; then
        echo "✓ $plugin_name installed to $TARGET_BASE."
    else
        echo "✗ Error installing $plugin_name."
    fi
done
