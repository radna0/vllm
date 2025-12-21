# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for loading TensorRT-LLM torch bindings."""

from __future__ import annotations

import ctypes
import functools
import importlib
import importlib.util
import os
import sys
import sysconfig
from pathlib import Path
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# QuantMode bit flags (keep in sync with TensorRT-LLM QuantMode).
_TRTLLM_INT8_KV_CACHE = 1 << 6
_TRTLLM_FP8_KV_CACHE = 1 << 7
_TRTLLM_NVFP4_KV_CACHE = 1 << 13


def _preload_shared_lib(path: Path) -> None:
    try:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        logger.debug_once("Failed to preload shared library %s: %s", path, exc)


def _preload_libpython() -> None:
    libdir = sysconfig.get_config_var("LIBDIR")
    if not libdir:
        return
    major, minor = sys.version_info[:2]
    candidates = [
        Path(libdir) / f"libpython{major}.{minor}.so.1.0",
        Path(libdir) / f"libpython{major}.{minor}.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            _preload_shared_lib(candidate)
            return


def _preload_torch_libs() -> None:
    torch_lib_dir = Path(torch.__file__).parent / "lib"
    if not torch_lib_dir.is_dir():
        return
    for lib in (
        "libc10.so",
        "libc10_cuda.so",
        "libtorch.so",
        "libtorch_cpu.so",
        "libtorch_cuda.so",
        "libtorch_python.so",
    ):
        candidate = torch_lib_dir / lib
        if candidate.exists():
            _preload_shared_lib(candidate)


def _preload_trtllm_libs(bindings_path: Path) -> None:
    lib_dir = envs.VLLM_TRTLLM_LIB_DIR
    if lib_dir:
        lib_root = Path(lib_dir)
    else:
        lib_root = bindings_path.parent / "libs"
    if not lib_root.is_dir():
        return
    for lib in (
        "libtensorrt_llm.so",
        "libdecoder_attention_0.so",
        "libdecoder_attention_1.so",
        "libth_common.so",
        "libpg_utils.so",
    ):
        candidate = lib_root / lib
        if candidate.exists():
            _preload_shared_lib(candidate)


def _maybe_add_trtllm_python_path() -> None:
    python_path = envs.VLLM_TRTLLM_PYTHON_PATH
    if not python_path:
        return
    for entry in python_path.split(os.pathsep):
        entry = entry.strip()
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)

def _select_bindings_candidate(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    py_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    for candidate in candidates:
        if py_tag in candidate.name:
            return candidate
    return sorted(candidates)[0]


def _find_local_trtllm_repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        repo_root = parent / "TensorRT-LLM"
        if (repo_root / "tensorrt_llm").is_dir():
            return repo_root
    return None


def _maybe_add_local_trtllm_repo() -> Path | None:
    repo_root = _find_local_trtllm_repo_root()
    if repo_root is None:
        return None
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def _find_bindings_path() -> Path | None:
    _maybe_add_trtllm_python_path()
    if envs.VLLM_TRTLLM_BINDINGS_PATH:
        candidate = Path(envs.VLLM_TRTLLM_BINDINGS_PATH)
        if candidate.is_file():
            return candidate
    candidates: list[Path] = []
    for entry in sys.path:
        base = Path(entry) / "tensorrt_llm"
        if not base.is_dir():
            continue
        candidates.extend(base.glob("bindings*.so"))
    selected = _select_bindings_candidate(candidates)
    if selected is not None:
        return selected
    repo_root = _maybe_add_local_trtllm_repo()
    if repo_root is not None:
        local_pkg = repo_root / "tensorrt_llm"
        selected = _select_bindings_candidate(
            list(local_pkg.glob("bindings*.so"))
        )
        if selected is not None:
            return selected
    return None


def _load_bindings_module(path: Path) -> Any | None:
    try:
        spec = importlib.util.spec_from_file_location("tensorrt_llm.bindings", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug_once("Failed to load TRTLLM bindings from %s: %s", path, exc)
        return None


@functools.cache
def get_trtllm_thop() -> Any | None:
    """Return TensorRT-LLM thop module if available, else None."""
    if envs.VLLM_TRTLLM_DISABLE:
        return None

    _maybe_add_trtllm_python_path()
    _maybe_add_local_trtllm_repo()

    try:
        from tensorrt_llm.bindings.internal import thop  # type: ignore

        return thop
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug_once("TRTLLM thop import failed: %s", exc)

    bindings_path = _find_bindings_path()
    if bindings_path is None:
        return None

    _preload_libpython()
    _preload_torch_libs()
    _preload_trtllm_libs(bindings_path)

    module = _load_bindings_module(bindings_path)
    if module is None:
        return None

    internal = getattr(module, "internal", None)
    if internal is not None and hasattr(internal, "thop"):
        return internal.thop
    if hasattr(module, "thop"):
        return module.thop
    return None


def has_trtllm_thop() -> bool:
    return get_trtllm_thop() is not None


@functools.cache
def load_trtllm_custom_ops() -> bool:
    """Load TensorRT-LLM torch custom ops if available."""
    if envs.VLLM_TRTLLM_DISABLE:
        return False

    _maybe_add_trtllm_python_path()

    bindings_path = _find_bindings_path()
    if bindings_path is not None:
        _preload_libpython()
        _preload_torch_libs()
        _preload_trtllm_libs(bindings_path)

    try:
        import tensorrt_llm._torch.custom_ops  # noqa: F401
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug_once("TRTLLM custom ops import failed: %s", exc)
        return False

    return hasattr(torch.ops, "trtllm")


def _has_trtllm_op(op_name: str) -> bool:
    if not load_trtllm_custom_ops():
        return False
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, op_name)


def has_trtllm_fp8_block_scale_moe() -> bool:
    return _has_trtllm_op("fp8_block_scale_moe_runner") and _has_trtllm_op(
        "fp8_quantize_1x128"
    )


def has_trtllm_fp4_block_scale_moe() -> bool:
    return _has_trtllm_op("fp4_block_scale_moe_runner") and _has_trtllm_op(
        "fp4_quantize"
    )


def get_trtllm_kv_cache_quant_mode(kv_cache_dtype: str | None) -> int:
    if kv_cache_dtype is None or kv_cache_dtype in ("auto", "bfloat16"):
        return 0
    if kv_cache_dtype.startswith("fp8"):
        return _TRTLLM_FP8_KV_CACHE
    if kv_cache_dtype == "nvfp4":
        return _TRTLLM_NVFP4_KV_CACHE
    if kv_cache_dtype == "int8":
        return _TRTLLM_INT8_KV_CACHE
    raise ValueError(f"Unsupported TRTLLM kv_cache_dtype: {kv_cache_dtype}")

def trtllm_attention_supports_nvfp4(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tokens_per_block: int | None,
    mask_type: int,
    kv_cache_dtype: str | None = "nvfp4",
    use_paged_context_fmha: bool = True,
    is_mla_enable: bool = False,
) -> bool | None:
    """Return True/False if TRTLLM reports NVFP4 attention support, else None."""
    if kv_cache_dtype != "nvfp4":
        return True
    if not _has_trtllm_op("attention_supports_nvfp4_output"):
        return None
    try:
        quant_mode = get_trtllm_kv_cache_quant_mode(kv_cache_dtype)
        return bool(
            torch.ops.trtllm.attention_supports_nvfp4_output(
                int(num_heads),
                int(num_kv_heads),
                int(head_size),
                None if tokens_per_block is None else int(tokens_per_block),
                int(mask_type),
                int(quant_mode),
                bool(use_paged_context_fmha),
                bool(is_mla_enable),
            )
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug_once("TRTLLM NVFP4 support check failed: %s", exc)
        return None


def run_trtllm_attention(**kwargs) -> None:
    """Invoke TRTLLM attention (thop) with explicit kwargs mapping."""
    thop = get_trtllm_thop()
    if thop is None:
        raise RuntimeError("TRTLLM thop bindings are not available.")
    thop.attention(**kwargs)


__all__ = [
    "get_trtllm_thop",
    "has_trtllm_thop",
    "get_trtllm_kv_cache_quant_mode",
    "load_trtllm_custom_ops",
    "has_trtllm_fp8_block_scale_moe",
    "has_trtllm_fp4_block_scale_moe",
    "trtllm_attention_supports_nvfp4",
    "run_trtllm_attention",
]
