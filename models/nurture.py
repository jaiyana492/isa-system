"""
models/nurture.py
CorePilora AI — Nurture Sequence Database Schema

Persistent memory layer for every lead in nurture.
Every attempt. Every channel. Every re-engagement.
Nothing forgotten. Nothing lost.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from config.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class NurtureStatus(str, enum.Enum):
    ACTIVE      = "active"
    PAUSED      = "paused"
    COMPLETED   = "completed"
    RE_ENGAGED  = "re_engaged"
    CONVERTED   = "converted"
    DEAD        = "dead"


class NurtureChannel(str, enum.Enum):
    CALL      = "call"
    SMS       = "sms"
    EMAIL     = "email"
    VOICEMAIL = "voicemail"


class NurtureSequenceType(str, enum.Enum):
    NO_ANSWER     = "no_answer"
    NOT_NOW       = "not_now"
    NO_SHOW       = "no_show"
    LONG_TERM     = "long_term"
    RE_ENGAGEMENT = "re_engagement"


# ─────────────────────────────────────────────────────────────────────────────
# NURTURE SEQUENCE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class NurtureSequence(Base):
    __tablename__ = "nurture_sequences"

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
    sequence_type = Column(
        SAEnum(NurtureSequenceType),
        nullable=False,
        default=NurtureSequenceType.NO_ANSWER,
    )
    status = Column(
        SAEnum(NurtureStatus),
        nullable=False,
        default=NurtureStatus.ACTIVE,
        index=True,
    )
    sequence_step = Column(
        Integer,
        nullable=False,
        default=1,
    )
    attempt_count = Column(
        Integer,
        nullable=False,
        default=0,
    )
    max_attempts = Column(
        Integer,
        nullable=False,
        default=8,
    )
    next_contact_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    last_contact_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_channel = Column(
        SAEnum(NurtureChannel),
        nullable=True,
    )
    last_message = Column(
        Text,
        nullable=True,
    )
    last_outcome = Column(
        String(100),
        nullable=True,
    )
    re_engaged_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )
    re_engagement_channel = Column(
        SAEnum(NurtureChannel),
        nullable=True,
    )
    re_engagement_message = Column(
        Text,
        nullable=True,
    )
    converted_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )
    appointment_id = Column(
        UUID(as_uuid=True),
        nullable=True,
    )
    market = Column(
        String(50),
        nullable=True,
    )
    lead_type = Column(
        String(20),
        nullable=True,
    )
    lead_source = Column(
        String(50),
        nullable=True,
    )
    sequence_data = Column(
        JSON,
        nullable=True,
    )
    notes = Column(
        Text,
        nullable=True,
    )
    is_priority = Column(
        Boolean,
        nullable=False,
        default=False,
    )
    do_not_contact = Column(
        Boolean,
        nullable=False,
        default=False,
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
    lead = relationship("LeadRecord", back_populates="nurture_sequences")
    attempts = relationship(
        "NurtureAttempt",
        back_populates="sequence",
        cascade="all, delete-orphan",
        order_by="NurtureAttempt.attempted_at",
    )

    def __repr__(self) -> str:
        return (
            f"<NurtureSequence id={self.id} "
            f"lead_id={self.lead_id} "
            f"status={self.status} "
            f"step={self.sequence_step} "
            f"attempts={self.attempt_count}>"
        )

    def is_exhausted(self) -> bool:
        return self.attempt_count >= self.max_attempts

    def is_contactable(self) -> bool:
        return (
            not self.do_not_contact
            and self.status == NurtureStatus.ACTIVE
            and not self.is_exhausted()
        )


# ─────────────────────────────────────────────────────────────────────────────
# NURTURE ATTEMPT MODEL
# ─────────────────────────────────────────────────────────────────────────────

class NurtureAttempt(Base):
    __tablename__ = "nurture_attempts"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    sequence_id = Column(
        UUID(as_uuid=True),
        ForeignKey("nurture_sequences.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    lead_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    step_number = Column(
        Integer,
        nullable=False,
    )
    channel = Column(
        SAEnum(NurtureChannel),
        nullable=False,
    )
    message_sent = Column(
        Text,
        nullable=True,
    )
    outcome = Column(
        String(100),
        nullable=True,
    )
    lead_response = Column(
        Text,
        nullable=True,
    )
    attempted_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    duration_seconds = Column(
        Integer,
        nullable=True,
    )

    # ── Relationships ─────────────────────────────────────────────────────
    sequence = relationship("NurtureSequence", back_populates="attempts")

    def __repr__(self) -> str:
        return (
            f"<NurtureAttempt id={self.id} "
            f"step={self.step_number} "
            f"channel={self.channel} "
            f"outcome={self.outcome}>"
        )