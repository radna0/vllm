from dataclasses import dataclass
from typing import Optional, Tuple
import re


@dataclass
class CMLEvent:
    """Structured event from CML parser."""

    event_type: str  # "call_start", "call_end", "trap", "intr_rejected"
    call_id: str | None = None
    tool_name: str | None = None
    tool_args: str | None = None


class CMLParser:
    """
    Incremental parser for AsyncLM Control Message Language (CML).
    Detects [CALL]...[END], [TRAP]...[END], and rejects model-emitted [INTR].

    Enhanced to extract structured tool call information.
    """

    CALL_START = "[CALL]"
    CALL_END = "[END]"
    TRAP_START = "[TRAP]"
    INTR_START = "[INTR]"  # Model should NOT emit this (P1.1)
    HEAD_SEP = "[HEAD]"

    def __init__(self):
        self.buffer = ""
        self.in_call = False
        self.in_trap = False
        self.call_start_idx = -1
        self._current_call_id: str | None = None
        self._call_content_buffer: str = ""

    def step(self, text: str) -> Tuple[bool, Optional[CMLEvent]]:
        """
        Process new text chunk.
        Returns:
            in_critical_section (bool): True if inside a [CALL] block.
            event (Optional[CMLEvent]): Structured event if a transition occurred.
        """
        self.buffer += text
        event = None

        # P1.1: Reject model-generated [INTR] blocks
        if self.INTR_START in self.buffer and not self.in_call:
            # Model is trying to generate an interrupt - this is a protocol violation
            intr_idx = self.buffer.find(self.INTR_START)
            event = CMLEvent(event_type="intr_rejected")
            # Consume the [INTR] marker to prevent infinite detection
            self.buffer = (
                self.buffer[:intr_idx] + self.buffer[intr_idx + len(self.INTR_START) :]
            )
            return self.in_call, event

        # Check for [CALL] start
        if not self.in_call:
            call_idx = self.buffer.find(self.CALL_START)
            if call_idx != -1:
                self.in_call = True
                self.call_start_idx = call_idx
                self._call_content_buffer = ""
                # Extract call_id: format is [CALL] call_id [HEAD] ...
                # We'll emit call_start event when we have the call_id
                post_call = self.buffer[call_idx + len(self.CALL_START) :]
                head_idx = post_call.find(self.HEAD_SEP)
                if head_idx != -1:
                    self._current_call_id = post_call[:head_idx].strip()
                    event = CMLEvent(
                        event_type="call_start", call_id=self._current_call_id
                    )

        # Check for [END] if in call
        if self.in_call:
            end_idx = self.buffer.find(
                self.CALL_END, self.call_start_idx + len(self.CALL_START)
            )
            if end_idx != -1:
                # Extract call content if we have [HEAD]
                call_block = self.buffer[
                    self.call_start_idx : end_idx + len(self.CALL_END)
                ]
                head_idx = call_block.find(self.HEAD_SEP)
                tool_name = None
                tool_args = None
                if head_idx != -1:
                    content = call_block[
                        head_idx + len(self.HEAD_SEP) : -(len(self.CALL_END))
                    ]
                    # Try to extract tool_name from common patterns like "python: <code>"
                    if ":" in content:
                        tool_name = content.split(":", 1)[0].strip()
                        tool_args = (
                            content.split(":", 1)[1].strip()
                            if ":" in content
                            else content
                        )
                    else:
                        tool_args = content.strip()

                # Call complete - emit call_end event
                event = CMLEvent(
                    event_type="call_end",
                    call_id=self._current_call_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                )

                self.in_call = False
                self.buffer = self.buffer[end_idx + len(self.CALL_END) :]
                self.call_start_idx = -1
                self._current_call_id = None

        # Check for [TRAP]
        trap_idx = self.buffer.find(self.TRAP_START)
        if trap_idx != -1 and event is None:
            event = CMLEvent(event_type="trap")
            # Consume trap to prevent repeated detection
            # Look for [END] after trap
            trap_end = self.buffer.find(self.CALL_END, trap_idx + len(self.TRAP_START))
            if trap_end != -1:
                self.buffer = self.buffer[trap_end + len(self.CALL_END) :]
            else:
                # No [END] yet, keep partial
                pass

        return self.in_call, event

    def reset(self):
        self.buffer = ""
        self.in_call = False
        self.in_trap = False
        self._current_call_id = None
