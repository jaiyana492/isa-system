"""
models/lead.py
CorePilora AI — Lead Database Schema

LeadMarket values match classifier.py Market enum exactly.
Relationships added for conversation sessions, nurture sequences,
and appointments.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

import phonenumbers
from pydantic import BaseModel, field_validator
from sqlalchemy import String, DateTime, Text, Integer, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from config.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class LeadType(str, Enum):
    BUYER    = "buyer"
    SELLER   = "seller"
    INVESTOR = "investor"


class LeadStatus(str, Enum):
    NEW          = "new"
    QUALIFIED    = "qualified"
    CONTACTED    = "contacted"
    CALLED       = "called"
    VOICEMAIL    = "voicemail"
    BOOKED       = "booked"
    DISQUALIFIED = "disqualified"


class LeadMarket(str, Enum):
    # FIX: Values now match classifier.py Market enum exactly.
    # Old: DALLAS = "dallas" — did not match "dallas_fort_worth"
    # New: matches classifier output — zero mismatch on routing
    DALLAS_FORT_WORTH = "dallas_fort_worth"
    HOUSTON           = "houston"
    ORLANDO           = "orlando"
    TAMPA             = "tampa"
    MIAMI             = "miami"
    DUBAI             = "dubai"
    UNKNOWN           = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# LEAD RECORD
# ─────────────────────────────────────────────────────────────────────────────

class LeadRecord(Base):
    __tablename__ = "leads"

    id:          Mapped[uuid.UUID]     = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name:   Mapped[str]           = mapped_column(String(255))
    phone:       Mapped[str]           = mapped_column(String(20))
    email:       Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source:      Mapped[str]           = mapped_column(String(100))
    market:      Mapped[str]           = mapped_column(String(50), default=LeadMarket.DALLAS_FORT_WORTH)
    lead_type:   Mapped[str]           = mapped_column(String(50), default=LeadType.BUYER)
    status:      Mapped[str]           = mapped_column(String(50), default=LeadStatus.NEW)
    raw_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    score:       Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    notes:       Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    action:      Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    hubspot_id:  Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    call_sid:    Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at:  Mapped[datetime]      = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:  Mapped[datetime]      = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # ── Relationships ─────────────────────────────────────────────────────
    conversation_sessions = relationship(
        "ConversationSession",
        back_populates="lead",
        cascade="all, delete-orphan",
    )
    nurture_sequences = relationship(
        "NurtureSequence",
        back_populates="lead",
        cascade="all, delete-orphan",
    )
    appointments = relationship(
        "Appointment",
        back_populates="lead",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<LeadRecord id={self.id} "
            f"name={self.full_name} "
            f"type={self.lead_type} "
            f"status={self.status} "
            f"market={self.market}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# LEAD PAYLOAD — PYDANTIC VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class LeadPayload(BaseModel):
    full_name:   str
    phone:       str
    email:       Optional[str] = None
    source:      str
    market:      str = "dallas_fort_worth"
    raw_message: Optional[str] = None

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        try:
            parsed = phonenumbers.parse(v, "US")
            if not phonenumbers.is_valid_number(parsed):
                raise ValueError("Invalid phone number")
            return phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.E164
            )
        except phonenumbers.NumberParseException as e:
            raise ValueError(f"Phone parse error: {e}")