# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Interrupt manager for async tool calling.

Manages tool result interrupts and injection into in-flight requests.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request


class InterruptManager:
    """Manages tool result interrupts and injection into requests."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.request_map: dict[str, "Request"] = {}

    def submit_call(self, request_id: str, call_id: str, call_body: str):
        """Register a tool call emitted by the model."""
        request = self.request_map.get(request_id)
        if request and request.async_tool_state:
            import time

            request.async_tool_state.call_registry[call_id] = {
                "body": call_body,
                "status": "pending",
                "created_at": time.time(),
            }
            # Throttled print
            if not hasattr(self, "_last_print_log"):
                self._last_print_log = 0
            if time.time() - self._last_print_log > 1.0:
                print(f"[InterruptManager] Registered call for {request_id}: {call_id}")
                self._last_print_log = time.time()

    def register_request(self, request: "Request"):
        """Register a request for interrupt injection."""
        self.request_map[request.request_id] = request

    def unregister_request(self, request_id: str):
        """Unregister a request."""
        self.request_map.pop(request_id, None)

    def enqueue_interrupt(
        self,
        request_id: str,
        call_id: str,
        payload: str,
        status: str = "ok",
    ):
        """
        Add interrupt to request's pending queue.

        Args:
            request_id: Request ID (session ID)
            call_id: Tool call ID
            payload: Tool result text
            status: "ok" or "error"
        """
        request = self.request_map.get(request_id)
        if request and request.async_tool_state:
            # Update call registry
            if call_id in request.async_tool_state.call_registry:
                import time

                request.async_tool_state.call_registry[call_id].update(
                    {
                        "status": status,
                        "finished_at": time.time(),
                    }
                )

            # Enqueue interrupt
            request.async_tool_state.pending_interrupts.append((call_id, payload))
            print(
                f"[InterruptManager] Enqueued interrupt for request {request_id}, call {call_id}"
            )

    def inject_interrupts(
        self,
        request: "Request",
        tokenizer,
    ) -> list[int]:
        """
        Build interrupt tokens and return them for append-prefill.

        Format: [INTR] {call_id} [HEAD] {payload} [END]

        Args:
            request: Request to inject interrupts into
            tokenizer: Tokenizer for encoding interrupt text

        Returns:
            List of interrupt token IDs to append
        """
        if not request.async_tool_state:
            return []

        interrupt_tokens = []

        while request.async_tool_state.pending_interrupts:
            call_id, payload = request.async_tool_state.pending_interrupts.popleft()

            # Build interrupt text - Use raw payload for Harmony protocol compatibility
            interrupt_text = payload
            tokens = tokenizer.encode(interrupt_text)

            # Mark as internal-only (don't show to user in output)
            request.async_tool_state.internal_only_token_ids.update(tokens)

            interrupt_tokens.extend(tokens)
            print(f"[InterruptManager] Injecting interrupt: {interrupt_text[:100]}...")

        return interrupt_tokens
