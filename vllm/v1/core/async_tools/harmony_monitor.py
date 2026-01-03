# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Simplified Harmony Token monitor for server-side async tool calling.

This module implements detection of python tool calls by looking for the
<|call|> token (200012) which marks the end of a tool call in Harmony format.
"""

from __future__ import annotations
import time
import re
from typing import TYPE_CHECKING, List, Literal
from dataclasses import dataclass

if TYPE_CHECKING:
    from vllm.v1.request import AsyncToolState


@dataclass
class ToolEvent:
    """Event emitted by token monitor."""

    type: Literal["on_call_complete", "on_trap", "set_interruptible"]
    call_id: str | None = None
    call_body: str | None = None
    interruptible: bool | None = None


class HarmonyTokenMonitor:
    """Simplified token monitor that detects python tool calls via <|call|> token."""

    # Key Harmony protocol tokens (discovered via sync benchmark logging)
    CHANNEL_TOKEN = 200005  # <|channel|>
    START_TOKEN = 200006  # <|start|>
    END_TOKEN = 200007  # <|end|>
    MESSAGE_TOKEN = 200008  # <|message|>
    CALL_TOKEN = 200012  # <|call|> - marks end of python tool call

    def __init__(self):
        # Tracking for tool call IDs (auto-generated)
        self.call_counter = 0

    def should_inject_interrupts(self, state: "AsyncToolState") -> bool:
        """Check if we should inject pending interrupts."""
        return bool(state.pending_interrupts)

    def process_token(
        self,
        token_id: int,
        token_text: str,
        token_index: int,
        state: "AsyncToolState",
    ) -> tuple[List[ToolEvent], bool]:
        """
        Process one token and detect python tool calls.

        Strategy:
        1. All tokens starting from <|start|> or <|channel|> are marked as internal
           (buffered) until the recipient is identified at <|message|>.
        2. If the recipient is 'python', the entire message is filtered.
        3. Otherwise, tokens are flushed by setting is_internal=False.
        """
        events = []
        is_internal = False

        # Extra: If we just finished a python call, filter the trailing <|end|>
        if getattr(state, "harmony_waiting_for_end_after_call", False):
            state.harmony_waiting_for_end_after_call = False
            if token_id == self.END_TOKEN:
                state.harmony_filtered_token_indices.add(token_index)
                # CRITICAL: Return False so it travels to API-level filtering in OutputProcessor.
                # If we return True, it gets buffered and might leak later.
                return events, False

        # 1) Start of any message or channel block - begin tracking and buffering
        if token_id in (self.START_TOKEN, self.CHANNEL_TOKEN):
            # If we are already in a potential call (e.g. multiple headers),
            # just keep going. But if not, start now.
            if not state.harmony_in_potential_call:
                state.harmony_in_potential_call = True
                state.harmony_accumulated_text = []
                state.harmony_filtering_body = False
                state.harmony_call_start_index = token_index
                state.harmony_is_buffering = True

            state.harmony_accumulated_text.append(token_text)
            return events, True  # Buffer the trigger token

        if state.harmony_in_potential_call:
            # 2) Accumulate text to identify role and channel
            state.harmony_accumulated_text.append(token_text)

            # 3) Check for end of tool call
            if token_id == self.CALL_TOKEN:
                accumulated = "".join(state.harmony_accumulated_text)

                # Check if this was a Python call
                is_python_call = state.harmony_filtering_body

                if is_python_call:
                    # Filter ALL tokens from start to call (inclusive)
                    start_idx = state.harmony_call_start_index
                    for idx in range(start_idx, token_index + 1):
                        state.harmony_filtered_token_indices.add(idx)

                    # Generate call ID and emit event
                    call_id = f"call_{state.harmony_call_counter}"
                    state.harmony_call_counter += 1

                    code = self._extract_code(accumulated)
                    if code:
                        events.append(
                            ToolEvent(
                                type="on_call_complete",
                                call_id=call_id,
                                call_body=code,
                            )
                        )
                        self._register_call(state, call_id, "python")
                        events.append(
                            ToolEvent(type="set_interruptible", interruptible=True)
                        )

                    # CRITICAL: Signal the scheduler to drop the entire buffer including current token
                    state.harmony_should_drop_buffer = True

                # Reset state
                state.harmony_in_potential_call = False
                state.harmony_filtering_body = False
                state.harmony_accumulated_text = []
                state.harmony_call_start_index = None
                state.harmony_is_buffering = False

                # If it was python, it stays hidden.
                # ALSO: We often see a trailing <|end|> after <|call|> in Harmony.
                # We need to filter that too.
                if is_python_call:
                    state.harmony_waiting_for_end_after_call = True

                return events, is_python_call

            # 4) Identify channel and recipient when <|message|> is hit
            if not state.harmony_filtering_body and token_id == self.MESSAGE_TOKEN:
                accumulated = "".join(state.harmony_accumulated_text)
                if "python" in accumulated.lower():
                    state.harmony_filtering_body = True
                    is_internal = True
                else:
                    # Non-python channel, stop buffering and flush
                    state.harmony_in_potential_call = False
                    state.harmony_is_buffering = False
                    is_internal = False  # Signals flushing of buffer in scheduler
                return events, is_internal

            # Continue buffering
            return events, True

        return events, is_internal

    def _extract_code(self, text: str) -> str:
        """Extract python code from accumulated text.

        The text contains Harmony markers and the code.
        Format: <|channel|>analysis to=python code<|message|>CODE_HERE<|call|>
        """
        # Remove markers to isolate the code
        # We look for content between <|message|> and <|call|>
        match = re.search(r"<\|message\|>(.*)", text, re.DOTALL)
        if match:
            code = match.group(1)
            # Remove any trailing markers that might be trapped (like <|call|> or others)
            code = re.sub(r"<\|.*", "", code)
            return code.strip()

        # Fallback to simple extraction if markers are missing
        lines = text.split("\n")
        code_lines = []
        for line in lines:
            # Skip empty lines and lines with only special chars
            if line.strip() and not line.strip().startswith("<|"):
                code_lines.append(line)

        return "\n".join(code_lines)

    def _register_call(self, state, call_id, recipient):
        """Register a tool call in the state."""
        state.call_registry[call_id] = {
            "status": "submitted",
            "submitted_at": time.time(),
            "recipient": recipient,
        }
