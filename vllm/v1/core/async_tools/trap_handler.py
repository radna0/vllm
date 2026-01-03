# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Trap handler for async tool calling.

Handles [TRAP] detection and KV cache strategy selection (keep/swap/recompute).
"""

import time
from typing import TYPE_CHECKING

from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request


class TrapHandler:
    """Handles [TRAP] detection and KV cache strategies."""

    # Cost coefficients (tunable based on hardware)
    SWAP_COST_PER_TOKEN = 0.0001  # Linear cost for swapping KV to CPU
    RECOMPUTE_COST_PER_TOKEN = 0.00001  # Quadratic cost for recomputing

    def __init__(self):
        self.tool_latency_estimates: dict[str, float] = {}
        self.recent_tool_durations: list[float] = []

    def on_trap_detected(
        self,
        request: "Request",
        kv_cache_manager: "KVCacheManager",
    ):
        """
        Called when [TRAP][END] is detected.

        Transition request to WAIT_TRAP status.
        Choose KV cache strategy: keep/swap/recompute.

        Args:
            request: Request that emitted trap
            kv_cache_manager: KV cache manager for swap/free operations
        """
        print(f"[TrapHandler] Trap detected for request {request.request_id}")

        # Transition to WAIT_TRAP
        request.status = RequestStatus.WAIT_TRAP
        request.async_tool_state.trap_seen = True
        request.async_tool_state.trap_timestamp = time.time()

        # Choose KV strategy
        strategy = self._choose_kv_strategy(request)
        print(f"[TrapHandler] Chosen KV strategy: {strategy}")

        if strategy == "swap":
            # Swap KV cache to CPU memory
            kv_cache_manager.swap_out(request.request_id)
        elif strategy == "recompute":
            # Free KV cache, will recompute on resume
            kv_cache_manager.free_blocks(request.request_id)
            # Mark for recompute (scheduler will handle)
            request.num_computed_tokens = 0

    def _choose_kv_strategy(self, request: "Request") -> str:
        """
        Choose KV strategy based on context length and estimated wait time.

        Strategy selection:
        - Keep: T_wait_est < min(T_swap, T_recompute)
        - Swap: T_swap < T_recompute and T_wait_est >= T_swap
        - Recompute: otherwise

        Args:
            request: Request to choose strategy for

        Returns:
            "keep", "swap", or "recompute"
        """
        L = request.num_tokens()
        T_wait_est = self._estimate_wait_time(request)

        # Estimate costs
        T_swap = self.SWAP_COST_PER_TOKEN * L
        T_recompute = self.RECOMPUTE_COST_PER_TOKEN * L * L

        # Choose strategy
        if T_wait_est < min(T_swap, T_recompute):
            return "keep"
        elif T_swap < T_recompute:
            return "swap"
        else:
            return "recompute"

    def _estimate_wait_time(self, request: "Request") -> float:
        """
        Estimate time until next interrupt arrives.

        Uses:
        1. Tool metadata (if available)
        2. Rolling average from recent tool durations
        3. Default fallback

        Args:
            request: Request to estimate wait time for

        Returns:
            Estimated wait time in seconds
        """
        # Use rolling average if available
        if self.recent_tool_durations:
            avg_duration = sum(self.recent_tool_durations) / len(
                self.recent_tool_durations
            )
            return avg_duration

        # Default: assume 1 second
        return 1.0

    def record_tool_duration(self, call_id: str, duration: float):
        """
        Record tool execution duration for future estimates.

        Args:
            call_id: Tool call ID
            duration: Execution duration in seconds
        """
        self.recent_tool_durations.append(duration)

        # Keep only last 10 durations
        if len(self.recent_tool_durations) > 10:
            self.recent_tool_durations.pop(0)
