"""
agents/intake.py
CorePilora AI — Universal Lead Intake Gate

Every lead enters here. No exceptions.
Validates → Classifies → Scores → Persists → Caches → Routes → Records.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from config.database import AsyncSessionLocal
from config.settings import settings
from models.lead import LeadRecord, LeadStatus, LeadMarket
from services.classifier import classify_lead, ClassificationResult
from services.scorer import score_lead
from services.crm import sync_lead_to_crm
from services.communication import make_outbound_call, update_lead_call_status
from services.conversation import create_session, complete_session
from services.scheduler import book_appointment, send_confirmation
from services.nurture import create_nurture_sequence
from services.cache import set_lead_cache
from services.lead_type_router import route_lead
from services.validator import validate_lead_payload

logger = logging.getLogger(__name__)

_VALID_MARKETS = {m.value for m in LeadMarket}


# ─────────────────────────────────────────────────────────────────────────────
# SAVE LEAD TO DATABASE
# ─────────────────────────────────────────────────────────────────────────────

async def _save_lead_record(
    validated:      dict,
    classification: ClassificationResult,
    score:          int,
) -> Optional[str]:
    """
    Persist LeadRecord to PostgreSQL.
    Returns lead_id UUID string. None on failure.
    """
    market_val = classification.market.value
    market     = market_val if market_val in _VALID_MARKETS else LeadMarket.UNKNOWN.value

    try:
        lead = LeadRecord(
            id          = uuid.uuid4(),
            full_name   = validated["full_name"],
            phone       = validated["phone"],
            email       = validated.get("email"),
            source      = validated.get("source", "website"),
            market      = market,
            lead_type   = classification.lead_type.value,
            status      = LeadStatus.NEW.value,
            raw_message = validated.get("message") or None,
            score       = score,
        )

        async with AsyncSessionLocal() as db:
            async with db.begin():
                db.add(lead)

        lead_id = str(lead.id)
        logger.info(
            "INTAKE | Lead persisted | id=%s | name=%s | type=%s | score=%s | market=%s",
            lead_id,
            validated["full_name"],
            classification.lead_type.value,
            score,
            market,
        )
        return lead_id

    except Exception as e:
        logger.error("INTAKE | _save_lead_record FAILED | error=%s", str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# APPOINTMENT TIME PARSER
# Converts LPMAMA natural language appointment to UTC datetime.
# ─────────────────────────────────────────────────────────────────────────────

def _parse_appointment_time(lpmama: dict) -> datetime:
    """
    Parse LPMAMA appointment string ("Tuesday at 10am") into a UTC datetime.
    Falls back to next business day 10am UTC if unparseable.
    """
    appt_str   = (lpmama or {}).get("appointment", "") or ""
    appt_lower = appt_str.lower()
    now        = datetime.now(timezone.utc)

    day_offsets = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    }

    for day_name, day_num in day_offsets.items():
        if day_name in appt_lower:
            today_num  = now.weekday()
            days_ahead = (day_num - today_num) % 7 or 7
            target     = now + timedelta(days=days_ahead)

            hour = 10
            if   "10am" in appt_lower or "10 am" in appt_lower: hour = 10
            elif "11am" in appt_lower or "11 am" in appt_lower: hour = 11
            elif "12pm" in appt_lower or "noon"  in appt_lower: hour = 12
            elif "1pm"  in appt_lower or "1 pm"  in appt_lower: hour = 13
            elif "2pm"  in appt_lower or "2 pm"  in appt_lower: hour = 14
            elif "3pm"  in appt_lower or "3 pm"  in appt_lower: hour = 15
            elif "4pm"  in appt_lower or "4 pm"  in appt_lower: hour = 16
            elif "morning"   in appt_lower: hour = 10
            elif "afternoon" in appt_lower: hour = 14

            return target.replace(hour=hour, minute=0, second=0, microsecond=0)

    # Default: next business day at 10am
    days_to_add = 1
    candidate   = now + timedelta(days=days_to_add)
    while candidate.weekday() >= 5:  # skip weekend
        days_to_add += 1
        candidate = now + timedelta(days=days_to_add)
    return candidate.replace(hour=10, minute=0, second=0, microsecond=0)


# ─────────────────────────────────────────────────────────────────────────────
# INTAKE RESULT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_intake_result(
    status:          str,
    classification:  Optional[ClassificationResult],
    pipeline_result: Optional[dict],
    payload:         dict,
    lead_id:         Optional[str] = None,
    score_result:    Optional[dict] = None,
    error:           Optional[str]  = None,
) -> dict[str, Any]:
    return {
        "status":          status,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "lead_id":         lead_id,
        "classification":  classification.to_dict() if classification else None,
        "score":           score_result,
        "pipeline_result": pipeline_result,
        "source":          payload.get("source", "unknown"),
        "error":           error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INTAKE HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def process_incoming_lead(payload: dict) -> dict[str, Any]:
    """
    Universal entry point for every incoming lead.

    Flow:
      1.  Validate payload
      2.  Classify lead (type / market / timeline / finance)
      3.  Score lead 1-100
      4.  Persist LeadRecord to PostgreSQL
      5.  Cache lead context in Redis
      6.  Fire-and-forget CRM sync (never blocks pipeline)
      7.  Create conversation session in DB
      8.  Inject lead_id into routing payload
      9.  Route to correct LangGraph pipeline
      10. Book appointment if pipeline set one
      11. Create nurture sequence if pipeline flagged one
      12. Complete session with full outcome data
      13. Return structured result
    """

    logger.info(
        "INTAKE START | source=%s | phone=%s",
        payload.get("source", "unknown"),
        payload.get("phone", "unknown"),
    )

    # ── Step 1 — Validate ─────────────────────────────────────────────────
    try:
        validated = validate_lead_payload(payload)
    except Exception as e:
        logger.error("INTAKE VALIDATION FAILED | error=%s", str(e))
        return _build_intake_result(
            status="validation_failed",
            classification=None,
            pipeline_result=None,
            payload=payload,
            error=str(e),
        )

    # ── Step 2 — Classify ─────────────────────────────────────────────────
    raw_message    = validated.get("message", "")
    classification = classify_lead(
        raw_message  = raw_message,
        lead_source  = validated.get("source"),
        market_hint  = validated.get("market"),
    )

    logger.info(
        "INTAKE CLASSIFIED | type=%s | confidence=%s | market=%s | timeline=%s | finance=%s",
        classification.lead_type.value,
        classification.confidence,
        classification.market.value,
        classification.timeline.value,
        classification.finance_type.value,
    )

    # ── Step 3 — Score ────────────────────────────────────────────────────
    score_result = score_lead(classification)
    lead_score   = score_result["score"]

    logger.info(
        "INTAKE SCORED | score=%s | temperature=%s",
        lead_score,
        score_result["temperature"],
    )

    # ── Step 4 — Persist lead to DB ───────────────────────────────────────
    lead_id = await _save_lead_record(
        validated      = validated,
        classification = classification,
        score          = lead_score,
    )

    if not lead_id:
        return _build_intake_result(
            status          = "db_error",
            classification  = classification,
            pipeline_result = None,
            payload         = payload,
            score_result    = score_result,
            error           = "Failed to persist lead to database",
        )

    # ── Step 5 — Cache lead context in Redis ──────────────────────────────
    await set_lead_cache(lead_id, {
        "lead_id":     lead_id,
        "full_name":   validated["full_name"],
        "phone":       validated["phone"],
        "email":       validated.get("email"),
        "source":      validated.get("source", "website"),
        "market":      classification.market.value,
        "lead_type":   classification.lead_type.value,
        "score":       lead_score,
        "temperature": score_result["temperature"],
        "flags":       classification.flags,
    })

    # Index phone → lead_id for voice pipeline caller lookup (30-day TTL)
    try:
        from config.redis import get_redis
        _redis = await get_redis()
        await _redis.setex(
            f"phone:index:{validated['phone']}",
            86400 * 30,
            lead_id,
        )
    except Exception as _e:
        logger.warning("INTAKE | Phone index failed | error=%s", str(_e))

    # ── Step 5.5 — Trigger outbound call ─────────────────────────────────
    webhook_url = f"https://{settings.APP_DOMAIN}/api/v1/voice/incoming"
    call_sid = await make_outbound_call(
        to_phone    = validated["phone"],
        lead_id     = lead_id,
        webhook_url = webhook_url,
    )
    voice_call_active = bool(call_sid)
    if call_sid:
        asyncio.create_task(
            update_lead_call_status(lead_id, call_sid, "called")
        )
        logger.info("INTAKE | Outbound call triggered | lead=%s | sid=%s", lead_id, call_sid)

    # ── Step 6 — CRM sync (fire-and-forget — never block pipeline) ────────
    asyncio.create_task(
        sync_lead_to_crm(
            full_name   = validated["full_name"],
            phone       = validated["phone"],
            email       = validated.get("email"),
            market      = classification.market.value,
            lead_type   = classification.lead_type.value,
            lead_source = validated.get("source", "website"),
            notes       = (
                f"Score: {lead_score} | "
                f"Temp: {score_result['temperature']} | "
                f"Confidence: {classification.confidence}% | "
                f"Timeline: {classification.timeline.value} | "
                f"Flags: {', '.join(classification.flags) or 'none'} | "
                f"Message: {raw_message[:200]}"
            ),
        )
    )

    # ── Step 7 — Create conversation session ──────────────────────────────
    session_id = await create_session(
        lead_id     = lead_id,
        channel     = "webchat",
        pipeline    = classification.lead_type.value,
        market      = classification.market.value,
        lead_source = validated.get("source", "website"),
        lead_type   = classification.lead_type.value,
    )

    if not session_id:
        logger.warning("INTAKE | Session creation failed — pipeline continues without session tracking")

    # ── Step 8 — Inject lead_id into payload for graph nodes ─────────────
    validated["lead_id"] = lead_id

    # ── Step 9 — Route to LangGraph pipeline ─────────────────────────────
    # Skip text pipeline when voice call was triggered — Twilio handles conv.
    if voice_call_active:
        pipeline_result = {
            "pipeline":        "voice",
            "appointment_set": False,
            "needs_nurture":   False,
            "turn_count":      0,
            "objection_count": 0,
            "lpmama":          {},
            "messages":        [],
        }
        logger.info("INTAKE | Voice call active — skipping text pipeline | lead=%s", lead_id)
    else:
        try:
            pipeline_result = await route_lead(
                result  = classification,
                payload = validated,
            )
        except Exception as e:
            logger.error("INTAKE ROUTING FAILED | lead=%s | error=%s", lead_id, str(e))
            if session_id:
                await complete_session(
                    session_id      = session_id,
                    outcome         = "incomplete",
                    appointment_set = False,
                )
            return _build_intake_result(
                status          = "routing_failed",
                classification  = classification,
                pipeline_result = None,
                payload         = payload,
                lead_id         = lead_id,
                score_result    = score_result,
                error           = str(e),
            )

    appointment_set = pipeline_result.get("appointment_set", False)
    needs_nurture   = pipeline_result.get("needs_nurture",   False)
    lpmama          = pipeline_result.get("lpmama",          {})
    turn_count      = pipeline_result.get("turn_count",      0)
    objection_count = pipeline_result.get("objection_count", 0)

    logger.info(
        "INTAKE PIPELINE DONE | pipeline=%s | appt=%s | nurture=%s | turns=%s",
        pipeline_result.get("pipeline", "unknown"),
        appointment_set,
        needs_nurture,
        turn_count,
    )

    appointment_id = None

    # ── Step 10 — Book appointment ────────────────────────────────────────
    if appointment_set:
        scheduled_at   = _parse_appointment_time(lpmama)
        appointment_id = await book_appointment(
            lead_id          = lead_id,
            session_id       = session_id or "",
            scheduled_at     = scheduled_at,
            appointment_type = classification.lead_type.value,
            market           = classification.market.value,
            lead_type        = classification.lead_type.value,
            lead_source      = validated.get("source", "website"),
            lpmama_state     = lpmama,
        )
        if appointment_id:
            logger.info(
                "INTAKE | Appointment booked | id=%s | scheduled=%s | lead=%s",
                appointment_id,
                scheduled_at.isoformat(),
                lead_id,
            )
            asyncio.create_task(
                send_confirmation(
                    appointment_id    = appointment_id,
                    lead_id           = lead_id,
                    phone             = validated["phone"],
                    lead_name         = validated["full_name"],
                    scheduled_at      = scheduled_at,
                    confirmation_type = "24hr",
                )
            )

    # ── Step 11 — Create nurture sequence ─────────────────────────────────
    if needs_nurture:
        sequence_type = "no_answer" if turn_count == 0 else "not_now"
        await create_nurture_sequence(
            lead_id       = lead_id,
            sequence_type = sequence_type,
            market        = classification.market.value,
            lead_type     = classification.lead_type.value,
            lead_source   = validated.get("source", "website"),
            is_priority   = (lead_score >= 60),
        )
        logger.info(
            "INTAKE | Nurture sequence created | lead=%s | type=%s | priority=%s",
            lead_id,
            sequence_type,
            lead_score >= 60,
        )

    # ── Step 12 — Complete session ────────────────────────────────────────
    if session_id:
        outcome = (
            "appointment_set" if appointment_set
            else "nurture_queued" if needs_nurture
            else "incomplete"
        )
        await complete_session(
            session_id      = session_id,
            outcome         = outcome,
            lpmama_state    = lpmama,
            messages        = pipeline_result.get("messages"),
            appointment_set = appointment_set,
            objection_count = objection_count,
        )

    logger.info(
        "INTAKE COMPLETE | lead=%s | outcome=%s | score=%s | appt_id=%s",
        lead_id,
        "appointment_set" if appointment_set else "nurture_queued" if needs_nurture else "incomplete",
        lead_score,
        appointment_id,
    )

    return _build_intake_result(
        status          = "success",
        classification  = classification,
        pipeline_result = pipeline_result,
        payload         = payload,
        lead_id         = lead_id,
        score_result    = score_result,
    )
