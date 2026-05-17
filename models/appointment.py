"""
models/appointment.py
CorePilora AI — Appointment Database Schema

Every booked appointment lives here.
This is the final outcome Jaiyana works toward.
Without this model — appointments exist nowhere.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    Text,
    Enum as SAEnum,
    ForeignKey,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from config.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class AppointmentStatus(str, enum.Enum):
    SCHEDULED   = "scheduled"    # Booked — not yet happened
    CONFIRMED   = "confirmed"    # Lead confirmed 24hr before
    COMPLETED   = "completed"    # Meeting happened
    NO_SHOW     = "no_show"      # Lead did not show
    CANCELLED   = "cancelled"    # Lead cancelled
    RESCHEDULED = "rescheduled"  # Moved to new time


class AppointmentType(str, enum.Enum):
    BUYER_CONSULTATION    = "buyer_consultation"
    LISTING_CONSULTATION  = "listing_consultation"
    INVESTOR_STRATEGY     = "investor_strategy"
    FOLLOW_UP             = "follow_up"


class AppointmentChannel(str, enum.Enum):
    IN_PERSON  = "in_person"
    VIDEO_CALL = "video_call"
    PHONE_CALL = "phone_call"


# ─────────────────────────────────────────────────────────────────────────────
# APPOINTMENT MODEL
# ─────────────────────────────────────────────────────────────────────────────

class Appointment(Base):
    __tablename__ = "appointments"

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
    session_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Conversation session that generated this appointment",
    )

    # ── Appointment Classification ─────────────────────────────────────────
    appointment_type = Column(
        SAEnum(AppointmentType),
        nullable=False,
        default=AppointmentType.BUYER_CONSULTATION,
    )
    channel = Column(
        SAEnum(AppointmentChannel),
        nullable=False,
        default=AppointmentChannel.PHONE_CALL,
    )
    status = Column(
        SAEnum(AppointmentStatus),
        nullable=False,
        default=AppointmentStatus.SCHEDULED,
        index=True,
    )

    # ── Scheduling ────────────────────────────────────────────────────────
    scheduled_at = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Confirmed appointment datetime",
    )
    duration_minutes = Column(
        String(10),
        nullable=True,
        default="30",
        comment="Expected duration in minutes",
    )
    timezone = Column(
        String(50),
        nullable=True,
        default="America/Chicago",
        comment="Lead timezone for appointment",
    )

    # ── Lead Context ──────────────────────────────────────────────────────
    market = Column(
        String(50),
        nullable=True,
    )
    lead_type = Column(
        String(20),
        nullable=True,
        comment="buyer / seller / investor",
    )
    lead_source = Column(
        String(50),
        nullable=True,
    )

    # ── LPMAMA at Booking ─────────────────────────────────────────────────
    lpmama_state = Column(
        JSON,
        nullable=True,
        comment="Full LPMAMA qualification state at time of booking",
    )

    # ── Confirmation Tracking ─────────────────────────────────────────────
    confirmed_24hr = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="True when 24hr confirmation sent and acknowledged",
    )
    confirmed_2hr = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="True when 2hr confirmation sent and acknowledged",
    )
    confirmed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When lead last confirmed the appointment",
    )

    # ── Outcome ───────────────────────────────────────────────────────────
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )
    no_show_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )
    reschedule_reason = Column(
        Text,
        nullable=True,
    )
    outcome_notes = Column(
        Text,
        nullable=True,
        comment="Agent notes after appointment completion",
    )

    # ── CRM Sync ──────────────────────────────────────────────────────────
    hubspot_deal_id = Column(
        String(100),
        nullable=True,
        comment="HubSpot deal ID linked to this appointment",
    )
    hubspot_synced = Column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ── Timestamps ────────────────────────────────────────────────────────
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
    lead = relationship("LeadRecord", back_populates="appointments")

    def __repr__(self) -> str:
        return (
            f"<Appointment id={self.id} "
            f"lead_id={self.lead_id} "
            f"type={self.appointment_type} "
            f"status={self.status} "
            f"scheduled_at={self.scheduled_at}>"
        )

    def is_upcoming(self) -> bool:
        from datetime import timezone
        return (
            self.status == AppointmentStatus.SCHEDULED
            and self.scheduled_at > datetime.now(timezone.utc)
        )

    def is_no_show(self) -> bool:
        return self.status == AppointmentStatus.NO_SHOW