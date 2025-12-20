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


def _find_bindings_path() -> Path | None:
    if envs.VLLM_TRTLLM_BINDINGS_PATH:
        candidate = Path(envs.VLLM_TRTLLM_BINDINGS_PATH)
        if candidate.is_file():
            return candidate
    for entry in sys.path:
        base = Path(entry) / "tensorrt_llm"
        if not base.is_dir():
            continue
        for candidate in base.glob("bindings*.so"):
            return candidate
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


def get_trtllm_kv_cache_quant_mode(kv_cache_dtype: str | None) -> int:
    if kv_cache_dtype is None or kv_cache_dtype == "auto":
        return 0
    if kv_cache_dtype.startswith("fp8"):
        return _TRTLLM_FP8_KV_CACHE
    if kv_cache_dtype == "nvfp4":
        return _TRTLLM_NVFP4_KV_CACHE
    if kv_cache_dtype == "int8":
        return _TRTLLM_INT8_KV_CACHE
    raise ValueError(f"Unsupported TRTLLM kv_cache_dtype: {kv_cache_dtype}")


__all__ = [
    "get_trtllm_thop",
    "has_trtllm_thop",
    "get_trtllm_kv_cache_quant_mode",
]
