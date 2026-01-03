# vLLM Drift Plugins

## Philosophy: No Source Build
We never, ever build vLLM from source. Instead, we:
1. **Inject**: Hot-patch the installed `vllm` package with our local Python logic (the "drift").
2. **Plugin**: Build and install modular CUDA/Triton extensions (PLUGINS) for high-performance kernels.

## Active PLUGINS

### 1. vllm_eagle
Optimized EAGLE speculative decoding kernels (Phase 2).
- **Features**: Warp-level reductions, fused sampling, fused draft-verify.
- **Speedup**: Up to 2.5x throughput for 120B-class models on H100.

---

## Usage Guide

### 1. Inject vLLM Logic
This "patches" your installed vLLM with the reasoning and speculative decoding logic found in `vllm-drift/vllm`.
```bash
bash PLUGINS/inject.sh /path/to/installed/vllm
```

### 2. Build & Install PLUGINS
Standard environment:
```bash
bash PLUGINS/install.sh
```

Kaggle environment:
```bash
bash PLUGINS/install_kaggle.sh
```

---

## Custom Plugin Development
To add a new plugin:
1. Create a folder in `PLUGINS/`.
2. Include a `setup.py` that builds a standalone extension.
3. Register it in `PLUGINS/README.md`.
