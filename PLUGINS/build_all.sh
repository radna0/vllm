#!/bin/bash
# build_all.sh: Build all plugins in-place (no install)

PLUGINS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find all directories with setup.py
find "$PLUGINS_DIR" -maxdepth 2 -name "setup.py" | while read setup_file; do
    plugin_path=$(dirname "$setup_file")
    plugin_name=$(basename "$plugin_path")
    
    echo "=========================================================="
    echo "Building plugin: $plugin_name (In-place)"
    echo "=========================================================="
    
    cd "$plugin_path"
    
    # Clean and build
    rm -rf build/ dist/ 2>/dev/null
    python setup.py build_ext --inplace
    
    if [ $? -eq 0 ]; then
        echo "✓ $plugin_name built successfully."
    else
        echo "✗ Error building $plugin_name."
    fi
done
