"""Python tool using Jupyter kernel for stateful execution."""

import os
import queue
import threading
import time
from typing import Any

# Harmony imports would normally go here, but for the benchmark we can use simple structures
# to stay compatible with the existing HarmonyTIRInferencer architecture.


def add_libs(code: str) -> str:
    """Add common math libraries to code."""
    return (
        "import math\nimport numpy as np\nimport sympy as sp\nfrom sympy import *\n"
        + code
    )


def ensure_last_print(code: str) -> str:
    """Ensure the last expression is printed."""
    lines = code.strip().split("\n")
    if lines and "print(" not in lines[-1] and "import" not in lines[-1]:
        if "#" in lines[-1]:
            lines[-1] = lines[-1].split("#")[0]
        lines[-1] = "print(" + lines[-1] + ")"
    return "\n".join(lines)


class LocalJupyterSession:
    """Stateful Jupyter kernel session for code execution."""

    # Class-level lock and port counter to avoid port conflicts
    _port_lock = threading.Lock()
    _next_port = 50000
    _max_port = 65535  # Maximum valid port number

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        """Get next available ports for kernel connection."""
        import socket

        with cls._port_lock:
            ports = []
            attempts = 0
            max_attempts = 100  # Prevent infinite loop

            while len(ports) < count and attempts < max_attempts:
                start_port = cls._next_port
                # Check if port range is available
                available = True
                for i in range(count):
                    port = start_port + i
                    if port > cls._max_port:
                        # Wrap around to beginning of port range
                        start_port = 50000
                        port = start_port + i

                    # Quick check if port is in use
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.settimeout(0.1)
                            result = s.connect_ex(("127.0.0.1", port))
                            if result == 0:
                                available = False
                                break
                    except Exception:
                        # If check fails, assume port might be in use
                        available = False
                        break

                if available:
                    ports = list(range(start_port, start_port + count))
                    cls._next_port = start_port + count
                    if cls._next_port > cls._max_port:
                        cls._next_port = 50000
                    break
                else:
                    # Try next range
                    cls._next_port += count
                    if cls._next_port > cls._max_port:
                        cls._next_port = 50000
                    attempts += 1

            if len(ports) < count:
                # Fallback: just return sequential ports without checking
                ports = list(range(cls._next_port, cls._next_port + count))
                cls._next_port += count
                if cls._next_port > cls._max_port:
                    cls._next_port = 50000

            return ports

    def __init__(self, connection_file: str | None = None, *, timeout: float = 120.0):
        try:
            from jupyter_client import BlockingKernelClient, KernelManager
        except ImportError as exc:
            raise RuntimeError("jupyter_client package required") from exc

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client: BlockingKernelClient
        self._km: KernelManager | None = None

        if connection_file:
            from pathlib import Path

            connection_path = Path(connection_file).expanduser()
            if not connection_path.exists():
                raise FileNotFoundError(f"Connection file not found: {connection_path}")
            client = BlockingKernelClient()
            client.load_connection_file(str(connection_path))
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
        else:
            # Allocate unique ports to avoid conflicts when running multiple kernels
            ports = self._get_next_ports(5)
            km = None
            max_retries = 3
            for retry in range(max_retries):
                try:
                    km = KernelManager()
                    km.shell_port = ports[0]
                    km.iopub_port = ports[1]
                    km.stdin_port = ports[2]
                    km.hb_port = ports[3]
                    km.control_port = ports[4]
                    km.start_kernel()
                    client = km.blocking_client()
                    client.start_channels()
                    client.wait_for_ready(timeout=self._default_timeout)
                    self._client = client
                    self._km = km
                    self._owns_kernel = True
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        # Try different ports
                        ports = self._get_next_ports(5)
                        if km is not None:
                            try:
                                km.shutdown_kernel(now=True)
                            except Exception:
                                pass
                    else:
                        # Last retry failed, raise the exception
                        raise RuntimeError(
                            f"Failed to start kernel after {max_retries} retries: {e}"
                        ) from e

    def execute(self, code: str, *, timeout: float | None = None) -> str:
        """Execute code and return combined stdout/stderr.
        timeout: WALL-CLOCK seconds limit for this execution.
        """
        import time
        import queue as _queue

        client = self._client
        effective_timeout = float(timeout or self._default_timeout)

        msg_id = client.execute(
            code, store_history=True, allow_stdin=False, stop_on_error=False
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        start = time.time()
        poll = (
            0.5  # seconds: small polling interval so we can enforce wall-clock timeout
        )

        def _timed_out() -> bool:
            return (time.time() - start) >= effective_timeout

        # iopub loop
        while True:
            if _timed_out():
                # interrupt the kernel to stop runaway execution
                try:
                    # BlockingKernelClient usually has interrupt_kernel
                    client.interrupt_kernel()
                except Exception:
                    try:
                        if self._owns_kernel and self._km is not None:
                            self._km.interrupt_kernel()
                    except Exception:
                        pass
                raise TimeoutError(
                    f"Python execution exceeded wall-time limit: {effective_timeout:.1f}s"
                )

            try:
                msg = client.get_iopub_msg(timeout=poll)
            except _queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # shell reply (also wall-time protected)
        while True:
            if _timed_out():
                try:
                    client.interrupt_kernel()
                except Exception:
                    try:
                        if self._owns_kernel and self._km is not None:
                            self._km.interrupt_kernel()
                    except Exception:
                        pass
                raise TimeoutError(
                    f"Python execution exceeded wall-time limit: {effective_timeout:.1f}s"
                )

            try:
                reply = client.get_shell_msg(timeout=poll)
            except _queue.Empty:
                continue

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            reply_content = reply.get("content", {})
            if reply_content.get("status") == "error":
                traceback_data = reply_content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = reply_content.get("ename", "")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            stdout = f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        if not stdout.strip():
            stdout = "[WARN] No output. Use print() to see results."
        return stdout

    def close(self):
        import contextlib

        with contextlib.suppress(Exception):
            self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

    def __del__(self):
        self.close()


class PythonTool:
    """Python execution tool using Jupyter kernel."""

    def __init__(
        self, execution_backend: str | None = None, local_jupyter_timeout: float = 60.0
    ):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._execution_lock = threading.Lock()
        self._jupyter_session: LocalJupyterSession | None = None
        # Lazy initialization to avoid port conflicts during object creation
        self._init_lock = threading.Lock()

        # Stats for benchmarking
        self.call_count = 0
        self.total_exec_time = 0.0

    def _ensure_session(self):
        """Lazily initialize the Jupyter session."""
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = LocalJupyterSession(
                        timeout=self._local_jupyter_timeout
                    )

    def reset_session(self):
        """Reset the Jupyter session state."""
        if self._jupyter_session:
            # Magic command to reset namespace
            self._jupyter_session.execute("%reset -f")

    @classmethod
    def get_tool_name(cls) -> str:
        return "python"

    @property
    def tool_config(self):
        """Return tool configuration for Harmony framework."""
        # Import here to avoid circular imports
        from openai_harmony import ToolNamespaceConfig

        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description="Execute Python code in a stateful Jupyter kernel environment. Use print() to see output.",
            tools=[],
        )

    def execute_code(self, code: str, timeout: float | None = None) -> str:
        """Directly execute code and return output as string."""
        self._ensure_session()
        start = time.perf_counter()
        try:
            output = self._jupyter_session.execute(code, timeout=timeout)
        except Exception as e:
            output = f"[ERROR] {e}"
        finally:
            elapsed = time.perf_counter() - start
            with self._execution_lock:
                self.call_count += 1
                self.total_exec_time += elapsed
        return output

    def get_stats(self):
        return {
            "call_count": self.call_count,
            "total_exec_time": self.total_exec_time,
            "avg_exec_time": self.total_exec_time / max(self.call_count, 1),
        }

    def close(self):
        if self._jupyter_session is not None:
            self._jupyter_session.close()
            self._jupyter_session = None

    def __del__(self):
        self.close()
