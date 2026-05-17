"""
services/nurture.py
CorePilora AI — Nurture Sequence Engine

Handles:
- Creating nurture sequences for leads not ready to close
- Executing follow-up attempts by channel
- Re-engagement detection
- Sequence progression logic

Every follow-up contains value — not just "are you ready yet?"
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from config.database import AsyncSessionLocal
from config.settings import settings
from models.nurture import (
    NurtureSequence,
    NurtureAttempt,
    NurtureStatus,
    NurtureChannel,
    NurtureSequenceType,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NURTURE SCHEDULE — days between each attempt
# ─────────────────────────────────────────────────────────────────────────────

NURTURE_SCHEDULE = {
    1: 0,    # Immediate
    2: 0,    # Same day — 4 hours later
    3: 1,    # Day 1
    4: 3,    # Day 3
    5: 7,    # Day 7
    6: 14,   # Day 14
    7: 30,   # Day 30
    8: 60,   # Day 60
}

CHANNEL_ROTATION = {
    1: NurtureChannel.CALL,
    2: NurtureChannel.SMS,
    3: NurtureChannel.CALL,
    4: NurtureChannel.EMAIL,
    5: NurtureChannel.CALL,
    6: NurtureChannel.SMS,
    7: NurtureChannel.CALL,
    8: NurtureChannel.EMAIL,
}


# ─────────────────────────────────────────────────────────────────────────────
# CREATE NURTURE SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

async def create_nurture_sequence(
    lead_id:       str,
    sequence_type: str,
    market:        str,
    lead_type:     str,
    lead_source:   str,
    is_priority:   bool = False,
) -> Optional[str]:
    """
    Create a new nurture sequence for a lead.
    Returns sequence ID if successful. None if failed.
    """
    type_map = {
        "no_answer":     NurtureSequenceType.NO_ANSWER,
        "not_now":       NurtureSequenceType.NOT_NOW,
        "no_show":       NurtureSequenceType.NO_SHOW,
        "long_term":     NurtureSequenceType.LONG_TERM,
        "re_engagement": NurtureSequenceType.RE_ENGAGEMENT,
    }

    try:
        sequence = NurtureSequence(
            id              = uuid.uuid4(),
            lead_id         = uuid.UUID(lead_id),
            sequence_type   = type_map.get(sequence_type, NurtureSequenceType.NO_ANSWER),
            status          = NurtureStatus.ACTIVE,
            sequence_step   = 1,
            attempt_count   = 0,
            max_attempts    = 8,
            next_contact_at = datetime.now(timezone.utc),
            market          = market,
            lead_type       = lead_type,
            lead_source     = lead_source,
            is_priority     = is_priority,
        )

        async with AsyncSessionLocal() as db:
            async with db.begin():
                db.add(sequence)

        logger.info(
            "NURTURE | Sequence created | lead=%s | type=%s | market=%s",
            lead_id, sequence_type, market,
        )
        return str(sequence.id)

    except Exception as e:
        logger.error(
            "NURTURE | create_nurture_sequence FAILED | lead=%s | error=%s",
            lead_id, str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTE NEXT ATTEMPT
# ─────────────────────────────────────────────────────────────────────────────

async def execute_next_attempt(
    sequence_id: str,
    lead_id:     str,
    phone:       str,
    lead_name:   str,
    market:      str,
    lead_type:   str,
) -> bool:
    """
    Execute the next nurture attempt in sequence.
    Sends message. Logs attempt. Schedules next contact.
    Returns True if executed. False if failed or exhausted.
    """
    try:
        # ── READ: load sequence state in its own transaction ──────────────
        async with AsyncSessionLocal() as db:
            sequence = await db.get(NurtureSequence, uuid.UUID(sequence_id))

            if not sequence:
                logger.error("NURTURE | Sequence not found | id=%s", sequence_id)
                return False

            if not sequence.is_contactable():
                logger.info(
                    "NURTURE | Not contactable | id=%s | status=%s",
                    sequence_id, sequence.status,
                )
                return False

            # Capture state before closing read session
            step           = sequence.sequence_step
            attempt_count  = sequence.attempt_count
            max_attempts   = sequence.max_attempts

        channel = CHANNEL_ROTATION.get(step, NurtureChannel.SMS)
        message = _build_nurture_message(
            step=step,
            lead_name=lead_name,
            market=market,
            lead_type=lead_type,
        )

        # ── EXTERNAL I/O: send contact outside any DB transaction ─────────
        outcome = await _send_nurture_contact(
            channel=channel,
            phone=phone,
            message=message,
            lead_id=lead_id,
        )

        next_step       = step + 1
        days_until_next = NURTURE_SCHEDULE.get(next_step, 90)
        next_contact    = datetime.now(timezone.utc) + timedelta(days=days_until_next)
        new_count       = attempt_count + 1

        # ── WRITE: persist attempt + update sequence in fresh transaction ──
        async with AsyncSessionLocal() as db:
            async with db.begin():
                attempt = NurtureAttempt(
                    id           = uuid.uuid4(),
                    sequence_id  = uuid.UUID(sequence_id),
                    lead_id      = uuid.UUID(lead_id),
                    step_number  = step,
                    channel      = channel,
                    message_sent = message,
                    outcome      = outcome,
                )
                db.add(attempt)

                seq = await db.get(NurtureSequence, uuid.UUID(sequence_id))
                if seq:
                    seq.sequence_step   = next_step
                    seq.attempt_count   = new_count
                    seq.last_contact_at = datetime.now(timezone.utc)
                    seq.last_channel    = channel
                    seq.last_message    = message
                    seq.last_outcome    = outcome
                    seq.next_contact_at = next_contact

                    if new_count >= max_attempts:
                        seq.status = NurtureStatus.DEAD
                        logger.info(
                            "NURTURE | Sequence exhausted | id=%s | lead=%s",
                            sequence_id, lead_id,
                        )

        logger.info(
            "NURTURE | Attempt executed | lead=%s | step=%s | channel=%s | outcome=%s",
            lead_id, step, channel, outcome,
        )
        return True

    except Exception as e:
        logger.error(
            "NURTURE | execute_next_attempt FAILED | lead=%s | error=%s",
            lead_id, str(e),
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MARK RE-ENGAGED
# ─────────────────────────────────────────────────────────────────────────────

async def mark_re_engaged(
    sequence_id:           str,
    re_engagement_message: str,
    channel:               str,
) -> bool:
    """Mark lead as re-engaged. Pauses nurture — lead re-enters pipeline."""
    channel_map = {
        "call":      NurtureChannel.CALL,
        "sms":       NurtureChannel.SMS,
        "email":     NurtureChannel.EMAIL,
        "voicemail": NurtureChannel.VOICEMAIL,
    }

    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                sequence = await db.get(NurtureSequence, uuid.UUID(sequence_id))
                if sequence:
                    sequence.status                = NurtureStatus.RE_ENGAGED
                    sequence.re_engaged_at         = datetime.now(timezone.utc)
                    sequence.re_engagement_channel = channel_map.get(channel)
                    sequence.re_engagement_message = re_engagement_message

        logger.info("NURTURE | Re-engaged | sequence=%s", sequence_id)
        return True

    except Exception as e:
        logger.error(
            "NURTURE | mark_re_engaged FAILED | sequence=%s | error=%s",
            sequence_id, str(e),
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MARK CONVERTED
# ─────────────────────────────────────────────────────────────────────────────

async def mark_converted(
    sequence_id:    str,
    appointment_id: str,
) -> bool:
    """Mark nurture sequence as converted — appointment booked."""
    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                sequence = await db.get(NurtureSequence, uuid.UUID(sequence_id))
                if sequence:
                    sequence.status         = NurtureStatus.CONVERTED
                    sequence.converted_at   = datetime.now(timezone.utc)
                    sequence.appointment_id = uuid.UUID(appointment_id)

        logger.info(
            "NURTURE | Converted | sequence=%s | appointment=%s",
            sequence_id, appointment_id,
        )
        return True

    except Exception as e:
        logger.error(
            "NURTURE | mark_converted FAILED | sequence=%s | error=%s",
            sequence_id, str(e),
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SEND NURTURE CONTACT — internal
# ─────────────────────────────────────────────────────────────────────────────

async def _send_nurture_contact(
    channel:  NurtureChannel,
    phone:    str,
    message:  str,
    lead_id:  str,
) -> str:
    """Route nurture contact to correct channel. Returns outcome string."""
    from services.communication import send_sms, make_outbound_call

    try:
        if channel == NurtureChannel.SMS:
            result = await send_sms(
                to_phone=phone,
                message=message,
                lead_id=lead_id,
            )
            return "sent" if result else "failed"

        elif channel == NurtureChannel.CALL:
            result = await make_outbound_call(
                to_phone=phone,
                lead_id=lead_id,
                webhook_url=f"https://{settings.APP_DOMAIN}/api/v1/voice/incoming",
            )
            return "initiated" if result else "failed"

        elif channel == NurtureChannel.EMAIL:
            logger.info("NURTURE | Email channel pending | lead=%s", lead_id)
            return "email_pending"

        return "unknown_channel"

    except Exception as e:
        logger.error(
            "NURTURE | _send_nurture_contact FAILED | channel=%s | error=%s",
            channel, str(e),
        )
        return "error"


# ─────────────────────────────────────────────────────────────────────────────
# NURTURE MESSAGE BUILDER
# Every message must contain value.
# ─────────────────────────────────────────────────────────────────────────────

def _build_nurture_message(
    step:      int,
    lead_name: str,
    market:    str,
    lead_type: str,
) -> str:
    """Build value-driven nurture message per step."""
    market_display = market.replace("_", " ").title()
    name           = lead_name or "there"

    messages = {
        1: f"Hey {name} — Jaiyana here. Wanted to follow up on your {market_display} real estate inquiry. Do you have 2 minutes?",
        2: f"Hey {name} — Jaiyana. Just left you a voicemail. Has anything changed on your timeline?",
        3: f"Hey {name} — {market_display} market moved this week. Wanted to make sure you have the latest before making any decisions.",
        4: f"Hey {name} — a few {lead_type} opportunities in {market_display} just came up that match what you described. Worth a quick call?",
        5: f"Hey {name} — Jaiyana. Just want to make sure you have everything you need on the {market_display} market. Any questions?",
        6: f"Hey {name} — {market_display} inventory is shifting. Wanted to keep you in the loop in case your timeline has moved up.",
        7: f"Hey {name} — one month check-in. Still thinking about {market_display} real estate or has the plan changed?",
        8: f"Hey {name} — last follow-up from me. If timing isn't right no problem. When it changes — I'm here.",
    }

    return messages.get(step, messages[8])