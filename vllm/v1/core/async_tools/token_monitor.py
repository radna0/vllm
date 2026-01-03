# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Token monitor for CML (Control Markup Language) protocol parsing.

This module implements incremental parsing of CML markers for async tool calling:
- [CALL] ... [END]: Tool call blocks
- [TRAP] [END]: Model signals waiting for tool results
- [INTR] ... [END]: Server-injected interrupts (model should never emit)
- [HEAD]: Optional separator between call ID and body
"""

from dataclasses import dataclass
from typing import Literal

from vllm.v1.request import AsyncToolState


@dataclass
class ToolEvent:
    """Event emitted by token monitor."""

    type: Literal["on_call_complete", "on_trap", "set_interruptible"]
    call_id: str | None = None
    call_body: str | None = None
    interruptible: bool | None = None


class CMLTokenMonitor:
    """Incremental parser for CML protocol markers."""

    # CML markers
    MARKERS = {
        "[CALL]": "call_start",
        "[END]": "block_end",
        "[TRAP]": "trap_start",
        "[INTR]": "interrupt_start",  # Server-only, block if model emits
        "[HEAD]": "header_sep",
    }

    def __init__(self):
        pass

    def process_token(
        self,
        token_id: int,
        token_text: str,
        state: AsyncToolState,
    ) -> tuple[list[ToolEvent], bool]:
        """
        Process one token, update FSM state, emit events.

        Returns:
            (events, is_internal)
        """
        events = []
        is_internal = False

        # Accumulate text for marker detection
        state.partial_marker_buffer += token_text

        # Check for complete markers
        marker_found = None
        marker_type = None

        for marker, mtype in self.MARKERS.items():
            if marker in state.partial_marker_buffer:
                marker_found = marker
                marker_type = mtype
                break

        if not marker_found:
            # No marker yet, accumulate in current state
            if state.fsm_state in ("IN_CALL_HEADER", "IN_CALL_BODY", "IN_TRAP"):
                state.call_buffer_tokens.append(token_id)
                state.call_buffer_text += token_text
                # For benchmark parity, we don't hide the contents
                is_internal = False

                # Throttled debug log
                import time

                if not hasattr(self, "_last_state_log"):
                    self._last_state_log = 0
                if time.time() - self._last_state_log > 2.0:
                    print(
                        f"[CMLMonitor] State: {state.fsm_state}, Buffer: {state.call_buffer_text[:100]}..."
                    )
                    self._last_state_log = time.time()

            return events, is_internal

        # Marker found - process FSM transition
        is_internal = False  # Keep markers in text stream for benchmark client

        # Clear buffer up to and including marker
        marker_idx = state.partial_marker_buffer.index(marker_found)
        before_marker = state.partial_marker_buffer[:marker_idx]

        # If there's text before the marker, accumulate it in the current state
        if before_marker and state.fsm_state in ("IN_CALL_HEADER", "IN_CALL_BODY"):
            state.call_buffer_text += before_marker

        # Clear the processed part of the buffer
        state.partial_marker_buffer = state.partial_marker_buffer[
            marker_idx + len(marker_found) :
        ]

        # FSM transitions
        if marker_type == "call_start":
            state.fsm_state = "IN_CALL_HEADER"
            state.in_critical_section = True
            state.call_buffer_tokens = []
            state.call_buffer_text = ""
            events.append(ToolEvent(type="set_interruptible", interruptible=False))
        elif marker_type == "trap_start":
            state.fsm_state = "IN_TRAP"
        elif marker_type == "header_sep":
            if state.fsm_state == "IN_CALL_HEADER":
                state.fsm_state = "IN_CALL_BODY"
        elif marker_type == "block_end":
            if state.fsm_state in ("IN_CALL_HEADER", "IN_CALL_BODY"):
                call_text = state.call_buffer_text.strip()
                call_id = "default"
                call_body = call_text
                if " " in call_text:
                    parts = call_text.split(maxsplit=1)
                    call_id = parts[0]
                    call_body = parts[1] if len(parts) > 1 else ""
                events.append(
                    ToolEvent(
                        type="on_call_complete", call_id=call_id, call_body=call_body
                    )
                )
                state.fsm_state = "TEXT"
                state.in_critical_section = False
                state.call_buffer_tokens = []
                state.call_buffer_text = ""
                events.append(ToolEvent(type="set_interruptible", interruptible=True))
            elif state.fsm_state == "IN_TRAP":
                events.append(ToolEvent(type="on_trap"))
                state.fsm_state = "TEXT"

        return events, is_internal

    def should_inject_interrupts(self, state: AsyncToolState) -> bool:
        """Check if it's safe to inject interrupts (not in critical section)."""
        return (
            not state.in_critical_section
            and len(state.pending_interrupts) > 0
            and state.fsm_state == "TEXT"
        )
