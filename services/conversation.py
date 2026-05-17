"""
services/conversation.py
CorePilora AI — Conversation State Manager

Handles:
- Creating conversation sessions in database
- Loading prior conversation context for memory
- Updating session state after every turn
- Writing conversation turns for audit trail
- Session completion and outcome logging

This is what gives Jaiyana memory across calls.
Without this — every call starts from zero.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update as sa_update

from config.database import AsyncSessionLocal
from models.conversation import (
    ConversationSession,
    ConversationTurn,
    ConversationChannel,
    ConversationStatus,
    ConversationOutcome,
    PipelineType,
)
from services.cache import (
    set_conversation_state,
    get_conversation_state,
    set_lpmama,
    get_lpmama,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CREATE SESSION
# ─────────────────────────────────────────────────────────────────────────────

async def create_session(
    lead_id:     str,
    channel:     str,
    pipeline:    str,
    market:      str,
    lead_source: str,
    lead_type:   str,
    prior_session_id: Optional[str] = None,
) -> Optional[str]:
    """
    Create a new conversation session in database.
    Returns session ID if successful. None if failed.
    Called at start of every new conversation.
    """
    channel_map = {
        "voice":   ConversationChannel.VOICE,
        "sms":     ConversationChannel.SMS,
        "email":   ConversationChannel.EMAIL,
        "webchat": ConversationChannel.WEBCHAT,
    }
    pipeline_map = {
        "buyer":      PipelineType.BUYER,
        "seller":     PipelineType.SELLER,
        "investor":   PipelineType.INVESTOR,
        "probe":      PipelineType.PROBE,
        "escalation": PipelineType.ESCALATION,
        "nurture":    PipelineType.NURTURE,
    }

    try:
        session_obj = ConversationSession(
            id          = uuid.uuid4(),
            lead_id     = uuid.UUID(lead_id),
            channel     = channel_map.get(channel, ConversationChannel.VOICE),
            pipeline    = pipeline_map.get(pipeline, PipelineType.BUYER),
            status      = ConversationStatus.ACTIVE,
            market      = market,
            lead_source = lead_source,
            lead_type   = lead_type,
            turn_count  = 0,
            objection_count   = 0,
            appointment_set   = False,
            lpmama_complete   = False,
            prior_session_id  = uuid.UUID(prior_session_id) if prior_session_id else None,
        )

        # Load prior context if this is a returning lead
        prior_summary = None
        if prior_session_id:
            prior_summary = await _load_prior_context(prior_session_id)
            session_obj.prior_context_summary = prior_summary

        async with AsyncSessionLocal() as db:
            async with db.begin():
                db.add(session_obj)

        session_id = str(session_obj.id)

        logger.info(
            "CONVERSATION | Session created | id=%s | lead=%s | pipeline=%s",
            session_id, lead_id, pipeline,
        )
        return session_id

    except Exception as e:
        logger.error(
            "CONVERSATION | create_session FAILED | lead=%s | error=%s",
            lead_id, str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LOAD PRIOR CONTEXT
# Gives Jaiyana memory of previous conversations with this lead.
# ─────────────────────────────────────────────────────────────────────────────

async def _load_prior_context(session_id: str) -> Optional[str]:
    """
    Load summary from a prior session.
    Returns context string for injection into system prompt.
    """
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ConversationSession)
                .where(ConversationSession.id == uuid.UUID(session_id))
            )
            prior = result.scalar_one_or_none()

            if not prior:
                return None

            summary_parts = []

            if prior.lpmama_state:
                collected = {
                    k: v for k, v in prior.lpmama_state.items() if v
                }
                if collected:
                    summary_parts.append(
                        f"Previously collected: {', '.join(f'{k}={v}' for k, v in collected.items())}"
                    )

            if prior.objection_count and prior.objection_count > 0:
                summary_parts.append(
                    f"Raised {prior.objection_count} objections previously."
                )

            if prior.outcome:
                summary_parts.append(
                    f"Last outcome: {prior.outcome.value}"
                )

            return " | ".join(summary_parts) if summary_parts else None

    except Exception as e:
        logger.error(
            "CONVERSATION | _load_prior_context FAILED | session=%s | error=%s",
            session_id, str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MOST RECENT SESSION FOR LEAD
# Used when lead calls back — find their last session.
# ─────────────────────────────────────────────────────────────────────────────

async def get_last_session_id(lead_id: str) -> Optional[str]:
    """
    Find the most recent conversation session for a lead.
    Returns session ID. None if no prior sessions.
    """
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ConversationSession.id)
                .where(ConversationSession.lead_id == uuid.UUID(lead_id))
                .order_by(ConversationSession.started_at.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            return str(row) if row else None

    except Exception as e:
        logger.error(
            "CONVERSATION | get_last_session_id FAILED | lead=%s | error=%s",
            lead_id, str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RECORD TURN
# Called after every conversation exchange.
# ─────────────────────────────────────────────────────────────────────────────

async def record_turn(
    session_id:       str,
    lead_id:          str,
    turn_number:      int,
    graph_node:       str,
    lead_message:     Optional[str],
    jaiyana_response: str,
    lpmama_after:     Optional[dict] = None,
    response_ms:      Optional[float] = None,
    tokens_used:      Optional[int] = None,
    is_objection:     bool = False,
    is_close_attempt: bool = False,
    is_confirmed:     bool = False,
) -> bool:
    """
    Record a single conversation turn in database.
    Full audit trail for every word spoken.
    """
    try:
        turn = ConversationTurn(
            id                    = uuid.uuid4(),
            session_id            = uuid.UUID(session_id),
            lead_id               = uuid.UUID(lead_id),
            turn_number           = turn_number,
            graph_node            = graph_node,
            lead_message          = lead_message,
            jaiyana_response      = jaiyana_response,
            lpmama_after          = lpmama_after,
            response_ms           = response_ms,
            tokens_used           = tokens_used,
            is_objection          = is_objection,
            is_appointment_close  = is_close_attempt,
            appointment_confirmed = is_confirmed,
        )

        async with AsyncSessionLocal() as db:
            async with db.begin():
                db.add(turn)

        # Update session turn count
        await _increment_turn_count(session_id)

        # Cache LPMAMA state in Redis
        if lpmama_after:
            await set_lpmama(lead_id, lpmama_after)

        return True

    except Exception as e:
        logger.error(
            "CONVERSATION | record_turn FAILED | session=%s | turn=%s | error=%s",
            session_id, turn_number, str(e),
        )
        return False


async def _increment_turn_count(session_id: str) -> None:
    """Increment turn count on session record."""
    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                result = await db.execute(
                    select(ConversationSession)
                    .where(ConversationSession.id == uuid.UUID(session_id))
                )
                session_obj = result.scalar_one_or_none()
                if session_obj:
                    session_obj.turn_count = (session_obj.turn_count or 0) + 1

    except Exception as e:
        logger.error(
            "CONVERSATION | _increment_turn_count FAILED | session=%s | error=%s",
            session_id, str(e),
        )


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE SESSION
# Called when conversation ends — naturally or by timeout.
# ─────────────────────────────────────────────────────────────────────────────

async def complete_session(
    session_id:      str,
    outcome:         str,
    lpmama_state:    Optional[dict] = None,
    messages:        Optional[list] = None,
    appointment_set: bool = False,
    objection_count: int = 0,
    duration_seconds: Optional[int] = None,
    twilio_call_sid: Optional[str] = None,
    recording_url:   Optional[str] = None,
    transcription:   Optional[str] = None,
) -> bool:
    """
    Mark session as completed with full outcome data.
    Writes final state to database.
    """
    outcome_map = {
        "appointment_set":    ConversationOutcome.APPOINTMENT_SET,
        "nurture_queued":     ConversationOutcome.NURTURE_QUEUED,
        "not_qualified":      ConversationOutcome.NOT_QUALIFIED,
        "wrong_number":       ConversationOutcome.WRONG_NUMBER,
        "do_not_contact":     ConversationOutcome.DO_NOT_CONTACT,
        "callback_requested": ConversationOutcome.CALLBACK_REQUESTED,
        "no_answer":          ConversationOutcome.NO_ANSWER,
        "incomplete":         ConversationOutcome.INCOMPLETE,
    }

    lpmama_complete = False
    if lpmama_state:
        lpmama_complete = all(v is not None for v in lpmama_state.values())

    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                await db.execute(
                    sa_update(ConversationSession)
                    .where(ConversationSession.id == uuid.UUID(session_id))
                    .values(
                        status           = ConversationStatus.COMPLETED,
                        outcome          = outcome_map.get(outcome, ConversationOutcome.INCOMPLETE),
                        lpmama_state     = lpmama_state,
                        lpmama_complete  = lpmama_complete,
                        messages         = messages,
                        appointment_set  = appointment_set,
                        objection_count  = objection_count,
                        duration_seconds = duration_seconds,
                        twilio_call_sid  = twilio_call_sid,
                        recording_url    = recording_url,
                        transcription    = transcription,
                        ended_at         = datetime.now(timezone.utc),
                    )
                )

        logger.info(
            "CONVERSATION | Session completed | id=%s | outcome=%s | appointment=%s",
            session_id, outcome, appointment_set,
        )
        return True

    except Exception as e:
        logger.error(
            "CONVERSATION | complete_session FAILED | session=%s | error=%s",
            session_id, str(e),
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ABANDON SESSION
# Called when lead drops off mid-conversation.
# ─────────────────────────────────────────────────────────────────────────────

async def abandon_session(session_id: str) -> bool:
    """Mark session as abandoned when lead disconnects."""
    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                await db.execute(
                    sa_update(ConversationSession)
                    .where(ConversationSession.id == uuid.UUID(session_id))
                    .values(
                        status   = ConversationStatus.ABANDONED,
                        outcome  = ConversationOutcome.INCOMPLETE,
                        ended_at = datetime.now(timezone.utc),
                    )
                )

        logger.info(
            "CONVERSATION | Session abandoned | id=%s",
            session_id,
        )
        return True

    except Exception as e:
        logger.error(
            "CONVERSATION | abandon_session FAILED | session=%s | error=%s",
            session_id, str(e),
        )
        return False