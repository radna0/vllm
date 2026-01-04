# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
import queue
import traceback
from typing import Dict, List, Optional
from jupyter_client import KernelManager


class JupyterToolExecutor:
    """Robust, stateful tool execution using a pool of Jupyter kernels."""

    def __init__(self, pool_size: int = 8):
        self.pool_size = pool_size
        self.kernels: List[Dict] = []
        self.request_to_kernel: Dict[str, int] = {}
        self.kernel_lock = threading.Lock()

        print(f"[JupyterExecutor] Initializing pool of {pool_size} kernels...")
        for i in range(pool_size):
            self._start_kernel(i)
        print("[JupyterExecutor] Pool initialized.")

    def _start_kernel(self, index: int):
        """Start or restart a kernel at directed index."""
        try:
            km = KernelManager(kernel_name="python3")
            km.start_kernel()
            kc = km.client()
            kc.start_channels()

            # Pre-load common libraries
            kc.execute(
                "import numpy as np; import math; import sympy as sp; from sympy import *",
                silent=True,
            )

            kernel_info = {
                "manager": km,
                "client": kc,
                "index": index,
                "busy": False,
                "last_used": time.time(),
            }

            with self.kernel_lock:
                if index < len(self.kernels):
                    # Shutdown old kernel if it exists
                    try:
                        self.kernels[index]["client"].stop_channels()
                        self.kernels[index]["manager"].shutdown_kernel()
                    except:
                        pass
                    self.kernels[index] = kernel_info
                else:
                    self.kernels.append(kernel_info)
        except Exception as e:
            print(f"[JupyterExecutor] Failed to start kernel {index}: {e}")

    def _get_kernel_for_request(self, request_id: str) -> Dict:
        """Get assigned kernel or assign a new one (LRU)."""
        with self.kernel_lock:
            if request_id in self.request_to_kernel:
                idx = self.request_to_kernel[request_id]
                return self.kernels[idx]

            # Find least recently used kernel that is not busy
            # Sort by last_used
            sorted_kernels = sorted(self.kernels, key=lambda k: k["last_used"])
            for k in sorted_kernels:
                if not k["busy"]:
                    idx = k["index"]
                    self.request_to_kernel[request_id] = idx
                    k["last_used"] = time.time()
                    return k

            # Fallback: just pick the oldest if all are busy (rare)
            idx = sorted_kernels[0]["index"]
            self.request_to_kernel[request_id] = idx
            return self.kernels[idx]

    def execute(self, request_id: str, code: str, timeout: float = 60.0) -> str:
        """Execute code in a stateful kernel and return stdout/stderr."""
        kernel = self._get_kernel_for_request(request_id)
        kc = kernel["client"]

        try:
            kernel["busy"] = True
            kernel["last_used"] = time.time()

            # Send execution request
            msg_id = kc.execute(code)

            # Capture output
            outputs = []
            while True:
                try:
                    msg = kc.get_iopub_msg(timeout=timeout)
                    msg_type = msg["header"]["msg_type"]
                    content = msg["content"]

                    if msg_type == "stream":
                        outputs.append(content["text"])
                    elif msg_type == "display_data" or msg_type == "execute_result":
                        if "text/plain" in content["data"]:
                            outputs.append(content["data"]["text/plain"])
                    elif msg_type == "error":
                        outputs.append(
                            f"\n[Traceback]\n" + "\n".join(content["traceback"])
                        )

                    # Check if execution is finished
                    if msg_type == "status" and content["execution_state"] == "idle":
                        # Verify it's for our request
                        if msg["parent_header"].get("msg_id") == msg_id:
                            break
                except queue.Empty:
                    outputs.append(f"\n[Timeout] Execution exceeded {timeout}s")
                    break

            result = "".join(outputs).strip()
            if not result:
                result = "Success (no output)"
            return result

        except Exception as e:
            return f"[Execution Error] {e}\n{traceback.format_exc()}"
        finally:
            kernel["busy"] = False

    def shutdown(self):
        """Shutdown all kernels."""
        with self.kernel_lock:
            for k in self.kernels:
                try:
                    k["client"].stop_channels()
                    k["manager"].shutdown_kernel()
                except:
                    pass
            self.kernels.clear()
            self.request_to_kernel.clear()
