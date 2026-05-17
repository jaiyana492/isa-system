"""
models/conversation.py
CorePilora AI — Conversation Session Database Schema

Persistent memory layer for every conversation Jaiyana has.
Every turn. Every channel. Every outcome.
This is what gives Jaiyana memory across sessions.
Without this — every call starts from zero.
"""

from __future__ import annotations

import enum
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Text,
    JSON,
    Float,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from config.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class ConversationChannel(str, enum.Enum):
    VOICE        = "voice"
    SMS          = "sms"
    EMAIL        = "email"
    WEBCHAT      = "webchat"


class ConversationStatus(str, enum.Enum):
    ACTIVE       = "active"        # Conversation in progress
    COMPLETED    = "completed"     # Natural end — appointment or nurture
    ABANDONED    = "abandoned"     # Lead dropped off mid conversation
    FAILED       = "failed"        # System error during conversation


class ConversationOutcome(str, enum.Enum):
    APPOINTMENT_SET   = "appointment_set"
    NURTURE_QUEUED    = "nurture_queued"
    NOT_QUALIFIED     = "not_qualified"
    WRONG_NUMBER      = "wrong_number"
    DO_NOT_CONTACT    = "do_not_contact"
    CALLBACK_REQUESTED = "callback_requested"
    NO_ANSWER         = "no_answer"
    INCOMPLETE        = "incomplete"


class PipelineType(str, enum.Enum):
    BUYER      = "buyer"
    SELLER     = "seller"
    INVESTOR   = "investor"
    PROBE      = "probe"
    ESCALATION = "escalation"
    NURTURE    = "nurture"


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION SESSION MODEL
# One record per conversation session.
# A lead can have multiple sessions over time.
# ─────────────────────────────────────────────────────────────────────────────

class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    # ── Identity ──────────────────────────────────────────────────────────
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    lead_id = Column(
        UUID(as_uuid=True),
        ForeignKey("leads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Session Classification ─────────────────────────────────────────────
    channel = Column(
        SAEnum(ConversationChannel),
        nullable=False,
        default=ConversationChannel.VOICE,
    )
    pipeline = Column(
        SAEnum(PipelineType),
        nullable=False,
        default=PipelineType.BUYER,
    )
    status = Column(
        SAEnum(ConversationStatus),
        nullable=False,
        default=ConversationStatus.ACTIVE,
        index=True,
    )
    outcome = Column(
        SAEnum(ConversationOutcome),
        nullable=True,
        index=True,
    )

    # ── Lead Context at Session Start ──────────────────────────────────────
    market = Column(
        String(50),
        nullable=True,
    )
    lead_source = Column(
        String(50),
        nullable=True,
    )
    lead_type = Column(
        String(20),
        nullable=True,
    )

    # ── LPMAMA State ───────────────────────────────────────────────────────
    lpmama_state = Column(
        JSON,
        nullable=True,
        comment="LPMAMA qualification fields at session end",
    )
    lpmama_complete = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="True when all 6 LPMAMA gates collected",
    )

    # ── Conversation Metrics ───────────────────────────────────────────────
    turn_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total conversation turns in this session",
    )
    objection_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total objections raised in this session",
    )
    duration_seconds = Column(
        Integer,
        nullable=True,
        comment="Total session duration in seconds",
    )

    # ── Full Conversation History ──────────────────────────────────────────
    messages = Column(
        JSON,
        nullable=True,
        comment="Full message history [{role, content, timestamp}]",
    )

    # ── Appointment Tracking ───────────────────────────────────────────────
    appointment_set = Column(
        Boolean,
        nullable=False,
        default=False,
    )
    appointment_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Linked appointment record if booked",
    )

    # ── Voice Call Specifics ───────────────────────────────────────────────
    twilio_call_sid = Column(
        String(100),
        nullable=True,
        unique=True,
        comment="Twilio call SID for voice sessions",
    )
    recording_url = Column(
        String(500),
        nullable=True,
        comment="Twilio call recording URL",
    )
    transcription = Column(
        Text,
        nullable=True,
        comment="Full Deepgram transcription of voice call",
    )

    # ── AI Performance ─────────────────────────────────────────────────────
    groq_tokens_used = Column(
        Integer,
        nullable=True,
        comment="Total Groq tokens consumed in this session",
    )
    avg_response_ms = Column(
        Float,
        nullable=True,
        comment="Average Jaiyana response latency in milliseconds",
    )

    # ── Prior Context ──────────────────────────────────────────────────────
    prior_session_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Previous session ID — enables memory chain across calls",
    )
    prior_context_summary = Column(
        Text,
        nullable=True,
        comment="Summary of prior sessions injected into this session",
    )

    # ── Flags ──────────────────────────────────────────────────────────────
    escalation_flags = Column(
        JSON,
        nullable=True,
        comment="Escalation flags detected during session",
    )
    is_escalated = Column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ── Timestamps ────────────────────────────────────────────────────────
    started_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    ended_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────
    lead = relationship("LeadRecord", back_populates="conversation_sessions")
    turns = relationship(
        "ConversationTurn",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ConversationTurn.turn_number",
    )

    def __repr__(self) -> str:
        return (
            f"<ConversationSession id={self.id} "
            f"lead_id={self.lead_id} "
            f"pipeline={self.pipeline} "
            f"status={self.status} "
            f"outcome={self.outcome}>"
        )

    def is_complete(self) -> bool:
        return self.status in [
            ConversationStatus.COMPLETED,
            ConversationStatus.ABANDONED,
            ConversationStatus.FAILED,
        ]

    def duration_minutes(self) -> float | None:
        if self.duration_seconds:
            return round(self.duration_seconds / 60, 2)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION TURN MODEL
# One record per individual message exchange.
# Full audit trail. Every word Jaiyana says. Every word the lead says.
# This is the raw data that trains better conversations over time.
# ─────────────────────────────────────────────────────────────────────────────

class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    # ── Identity ──────────────────────────────────────────────────────────
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    lead_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    # ── Turn Detail ───────────────────────────────────────────────────────
    turn_number = Column(
        Integer,
        nullable=False,
        comment="Sequential turn number within session",
    )
    graph_node = Column(
        String(50),
        nullable=True,
        comment="LangGraph node that generated this turn",
    )

    # ── Messages ──────────────────────────────────────────────────────────
    lead_message = Column(
        Text,
        nullable=True,
        comment="What the lead said in this turn",
    )
    jaiyana_response = Column(
        Text,
        nullable=False,
        comment="What Jaiyana said in this turn",
    )

    # ── LPMAMA Snapshot ───────────────────────────────────────────────────
    lpmama_after = Column(
        JSON,
        nullable=True,
        comment="LPMAMA state after this turn",
    )

    # ── Performance ───────────────────────────────────────────────────────
    response_ms = Column(
        Float,
        nullable=True,
        comment="Jaiyana response generation time in milliseconds",
    )
    tokens_used = Column(
        Integer,
        nullable=True,
        comment="Groq tokens used for this turn",
    )

    # ── Flags ─────────────────────────────────────────────────────────────
    is_objection = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if lead raised an objection in this turn",
    )
    is_appointment_close = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if this turn contained an appointment close attempt",
    )
    appointment_confirmed = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if lead confirmed appointment in this turn",
    )

    # ── Timestamp ─────────────────────────────────────────────────────────
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # ── Relationships ─────────────────────────────────────────────────────
    session = relationship("ConversationSession", back_populates="turns")

    def __repr__(self) -> str:
        return (
            f"<ConversationTurn id={self.id} "
            f"turn={self.turn_number} "
            f"node={self.graph_node} "
            f"confirmed={self.appointment_confirmed}>"
        )