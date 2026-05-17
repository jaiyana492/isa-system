"""
agents/utils.py
CorePilora AI — Shared Agent Utilities

Single source of truth for logic shared across buyer, seller, investor graphs.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# APPOINTMENT CONFIRMATION DETECTOR
# Reads the LEAD's last message — not Jaiyana's response.
# Called inside node_qualify of every graph after each turn.
# ─────────────────────────────────────────────────────────────────────────────

_CONFIRM_SIGNALS: frozenset[str] = frozenset([
    "yes", "yeah", "yep", "sure", "ok", "okay",
    "that works", "sounds good", "perfect", "great",
    "confirmed", "book it", "let's do it", "lets do it",
    "i'll take", "ill take", "i'll do", "ill do",
    "works for me", "that's fine", "thats fine",
    "tuesday works", "wednesday works", "thursday works",
    "friday works", "monday works",
    "morning works", "afternoon works",
    "10am works", "11am works", "2pm works", "3pm works",
    "i'm in", "im in", "sign me up", "set it up",
])


def lead_confirmed_appointment(lead_message: str) -> bool:
    """
    Return True if the lead's message contains an appointment confirmation signal.

    Args:
        lead_message: Raw transcribed or typed message from the lead.

    Returns:
        True if the lead confirmed the appointment, False otherwise.
    """
    if not lead_message:
        return False

    lead_lower = lead_message.lower()
    return any(signal in lead_lower for signal in _CONFIRM_SIGNALS)
