"""
services/scheduler.py
CorePilora AI — Appointment Booking & Confirmation Engine

Handles:
- Appointment creation in database
- 24hr and 2hr confirmation messages
- No-show detection and re-routing
- HubSpot deal creation on booking
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from config.database import AsyncSessionLocal
from models.appointment import Appointment, AppointmentStatus, AppointmentType, AppointmentChannel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BOOK APPOINTMENT
# ─────────────────────────────────────────────────────────────────────────────

async def book_appointment(
    lead_id:          str,
    session_id:       str,
    scheduled_at:     datetime,
    appointment_type: str,
    market:           str,
    lead_type:        str,
    lead_source:      str,
    lpmama_state:     Optional[dict] = None,
    channel:          str = "phone_call",
) -> Optional[str]:
    """
    Create appointment record in database.
    Returns appointment ID if successful. None if failed.
    Called when lead confirms appointment in conversation.
    """
    try:
        type_map = {
            "buyer":    AppointmentType.BUYER_CONSULTATION,
            "seller":   AppointmentType.LISTING_CONSULTATION,
            "investor": AppointmentType.INVESTOR_STRATEGY,
        }
        channel_map = {
            "phone_call": AppointmentChannel.PHONE_CALL,
            "video_call": AppointmentChannel.VIDEO_CALL,
            "in_person":  AppointmentChannel.IN_PERSON,
        }

        appointment = Appointment(
            id               = uuid.uuid4(),
            lead_id          = uuid.UUID(lead_id),
            session_id       = uuid.UUID(session_id) if session_id else None,
            appointment_type = type_map.get(lead_type, AppointmentType.BUYER_CONSULTATION),
            channel          = channel_map.get(channel, AppointmentChannel.PHONE_CALL),
            status           = AppointmentStatus.SCHEDULED,
            scheduled_at     = scheduled_at,
            market           = market,
            lead_type        = lead_type,
            lead_source      = lead_source,
            lpmama_state     = lpmama_state,
            hubspot_synced   = False,
        )

        async with AsyncSessionLocal() as session:
            async with session.begin():
                session.add(appointment)

        logger.info(
            "SCHEDULER | Appointment booked | lead=%s | at=%s | type=%s",
            lead_id,
            scheduled_at.isoformat(),
            lead_type,
        )
        return str(appointment.id)

    except Exception as e:
        logger.error(
            "SCHEDULER | book_appointment FAILED | lead=%s | error=%s",
            lead_id, str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SEND CONFIRMATION
# 24hr and 2hr confirmation messages via SMS.
# ─────────────────────────────────────────────────────────────────────────────

async def send_confirmation(
    appointment_id: str,
    lead_id:        str,
    phone:          str,
    lead_name:      str,
    scheduled_at:   datetime,
    confirmation_type: str,  # "24hr" or "2hr"
) -> bool:
    """
    Send appointment confirmation SMS to lead.
    Updates confirmation flags in database.
    """
    from services.communication import send_sms

    time_str = scheduled_at.strftime("%A, %B %d at %I:%M %p")

    if confirmation_type == "24hr":
        message = (
            f"Hey {lead_name} — Jaiyana here. "
            f"Just confirming our appointment tomorrow, {time_str}. "
            f"Reply YES to confirm or call me to reschedule."
        )
    else:
        message = (
            f"Hey {lead_name} — Jaiyana. "
            f"See you in 2 hours at {time_str}. "
            f"Looking forward to it."
        )

    success = await send_sms(
        to_phone=phone,
        message=message,
        lead_id=lead_id,
    )

    if success:
        await _update_confirmation_flag(
            appointment_id=appointment_id,
            confirmation_type=confirmation_type,
        )

    return bool(success)


async def _update_confirmation_flag(
    appointment_id:    str,
    confirmation_type: str,
) -> None:
    """Update confirmation flag in database."""
    try:
        from sqlalchemy import update as sa_update

        async with AsyncSessionLocal() as session:
            async with session.begin():
                values = {}
                if confirmation_type == "24hr":
                    values["confirmed_24hr"] = True
                else:
                    values["confirmed_2hr"] = True
                    values["confirmed_at"]  = datetime.now(timezone.utc)

                await session.execute(
                    sa_update(Appointment)
                    .where(Appointment.id == uuid.UUID(appointment_id))
                    .values(**values)
                )

    except Exception as e:
        logger.error(
            "SCHEDULER | _update_confirmation_flag FAILED | appt=%s | error=%s",
            appointment_id, str(e),
        )


# ─────────────────────────────────────────────────────────────────────────────
# MARK NO SHOW
# ─────────────────────────────────────────────────────────────────────────────

async def mark_no_show(appointment_id: str) -> bool:
    """
    Mark appointment as no-show.
    Triggers immediate rescue nurture sequence.
    """
    try:
        from sqlalchemy import update as sa_update

        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    sa_update(Appointment)
                    .where(Appointment.id == uuid.UUID(appointment_id))
                    .values(
                        status=AppointmentStatus.NO_SHOW,
                        no_show_at=datetime.now(timezone.utc),
                    )
                )

        logger.warning(
            "SCHEDULER | No-show marked | appointment_id=%s",
            appointment_id,
        )
        return True

    except Exception as e:
        logger.error(
            "SCHEDULER | mark_no_show FAILED | appt=%s | error=%s",
            appointment_id, str(e),
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MARK COMPLETED
# ─────────────────────────────────────────────────────────────────────────────

async def mark_completed(appointment_id: str) -> bool:
    """Mark appointment as completed after meeting happens."""
    try:
        from sqlalchemy import update as sa_update

        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    sa_update(Appointment)
                    .where(Appointment.id == uuid.UUID(appointment_id))
                    .values(
                        status=AppointmentStatus.COMPLETED,
                        completed_at=datetime.now(timezone.utc),
                    )
                )

        logger.info(
            "SCHEDULER | Appointment completed | id=%s",
            appointment_id,
        )
        return True

    except Exception as e:
        logger.error(
            "SCHEDULER | mark_completed FAILED | appt=%s | error=%s",
            appointment_id, str(e),
        )
        return False