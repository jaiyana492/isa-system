"""
services/nurture_runner.py
CorePilora AI — Background Nurture Sequence Runner

Asyncio task started at server startup.
Runs every 10 minutes. Finds all ACTIVE nurture sequences
due for contact and fires execute_next_attempt().
Also sends pending appointment confirmations (24hr, 2hr).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, and_

from config.database import AsyncSessionLocal
from models.nurture import NurtureSequence, NurtureStatus
from models.appointment import Appointment, AppointmentStatus
from services.nurture import execute_next_attempt
from services.scheduler import send_confirmation

logger = logging.getLogger(__name__)

RUNNER_INTERVAL_SECONDS = 600   # 10 minutes
CONFIRMATION_WINDOW_MINUTES = 30  # send 24hr confirm within 30-min window


# ─────────────────────────────────────────────────────────────────────────────
# NURTURE SEQUENCES — RUN DUE ATTEMPTS
# ─────────────────────────────────────────────────────────────────────────────

async def _run_due_nurture_sequences() -> int:
    """Query ACTIVE sequences due now. Execute each attempt. Return count."""
    now = datetime.now(timezone.utc)
    executed = 0

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(NurtureSequence).where(
                    and_(
                        NurtureSequence.status == NurtureStatus.ACTIVE,
                        NurtureSequence.next_contact_at <= now,
                    )
                )
            )
            sequences = result.scalars().all()

        for seq in sequences:
            try:
                # Load lead phone + name from leads table
                from models.lead import LeadRecord
                async with AsyncSessionLocal() as db:
                    lead = await db.get(LeadRecord, seq.lead_id)

                if not lead:
                    logger.warning(
                        "NURTURE RUNNER | Lead not found | sequence=%s | lead=%s",
                        seq.id, seq.lead_id,
                    )
                    continue

                success = await execute_next_attempt(
                    sequence_id = str(seq.id),
                    lead_id     = str(seq.lead_id),
                    phone       = lead.phone,
                    lead_name   = lead.full_name,
                    market      = seq.market,
                    lead_type   = seq.lead_type,
                )
                if success:
                    executed += 1

            except Exception as e:
                logger.error(
                    "NURTURE RUNNER | Attempt failed | sequence=%s | error=%s",
                    seq.id, str(e),
                )

    except Exception as e:
        logger.error("NURTURE RUNNER | Query failed | error=%s", str(e))

    return executed


# ─────────────────────────────────────────────────────────────────────────────
# APPOINTMENT CONFIRMATIONS — 24HR AND 2HR
# ─────────────────────────────────────────────────────────────────────────────

async def _send_pending_confirmations() -> int:
    """Send 24hr and 2hr appointment confirmations as appointments come due."""
    now = datetime.now(timezone.utc)
    sent = 0

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Appointment).where(
                    Appointment.status == AppointmentStatus.SCHEDULED,
                )
            )
            appointments = result.scalars().all()

        for appt in appointments:
            try:
                from models.lead import LeadRecord
                async with AsyncSessionLocal() as db:
                    lead = await db.get(LeadRecord, appt.lead_id)

                if not lead:
                    continue

                hours_until = (appt.scheduled_at - now).total_seconds() / 3600

                # 24hr window: between 23 and 25 hours out, not yet sent
                if 23 <= hours_until <= 25 and not appt.confirmed_24hr:
                    success = await send_confirmation(
                        appointment_id    = str(appt.id),
                        lead_id           = str(appt.lead_id),
                        phone             = lead.phone,
                        lead_name         = lead.full_name,
                        scheduled_at      = appt.scheduled_at,
                        confirmation_type = "24hr",
                    )
                    if success:
                        sent += 1
                        logger.info(
                            "NURTURE RUNNER | 24hr confirm sent | appt=%s | lead=%s",
                            appt.id, appt.lead_id,
                        )

                # 2hr window: between 1.75 and 2.25 hours out, not yet sent
                elif 1.75 <= hours_until <= 2.25 and not appt.confirmed_2hr:
                    success = await send_confirmation(
                        appointment_id    = str(appt.id),
                        lead_id           = str(appt.lead_id),
                        phone             = lead.phone,
                        lead_name         = lead.full_name,
                        scheduled_at      = appt.scheduled_at,
                        confirmation_type = "2hr",
                    )
                    if success:
                        sent += 1
                        logger.info(
                            "NURTURE RUNNER | 2hr confirm sent | appt=%s | lead=%s",
                            appt.id, appt.lead_id,
                        )

            except Exception as e:
                logger.error(
                    "NURTURE RUNNER | Confirmation failed | appt=%s | error=%s",
                    appt.id, str(e),
                )

    except Exception as e:
        logger.error("NURTURE RUNNER | Confirmation query failed | error=%s", str(e))

    return sent


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def run_nurture_loop() -> None:
    """
    Background asyncio task. Runs every 10 minutes.
    Fires nurture attempts + appointment confirmations.
    Designed to run for the lifetime of the server process.
    """
    logger.info("NURTURE RUNNER | Started")

    while True:
        try:
            nurture_count = await _run_due_nurture_sequences()
            confirm_count = await _send_pending_confirmations()

            if nurture_count or confirm_count:
                logger.info(
                    "NURTURE RUNNER | Tick | nurture_attempts=%s | confirmations=%s",
                    nurture_count, confirm_count,
                )

        except asyncio.CancelledError:
            logger.info("NURTURE RUNNER | Cancelled — shutting down")
            break
        except Exception as e:
            logger.error("NURTURE RUNNER | Tick error | error=%s", str(e))

        await asyncio.sleep(RUNNER_INTERVAL_SECONDS)
