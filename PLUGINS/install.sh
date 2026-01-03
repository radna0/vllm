#!/bin/bash
# install.sh: Standard installation for all plugins

PLUGINS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find all directories with setup.py
find "$PLUGINS_DIR" -maxdepth 2 -name "setup.py" | while read setup_file; do
    plugin_path=$(dirname "$setup_file")
    plugin_name=$(basename "$plugin_path")
    
    echo "=========================================================="
    echo "Installing plugin: $plugin_name from $plugin_path"
    echo "=========================================================="
    
    cd "$plugin_path"
    pip install . --no-build-isolation
    
    if [ $? -eq 0 ]; then
        echo "✓ $plugin_name installed successfully."
    else
        echo "✗ Error installing $plugin_name."
    fi
done
