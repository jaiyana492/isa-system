"""
services/communication.py
CorePilora AI — Voice & Messaging Layer

Handles:
- Telnyx inbound/outbound calls
- Deepgram STT — speech to text
- ElevenLabs TTS with Deepgram fallback
- Lead last message injection into graph state
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import httpx

from config.settings import settings
from config.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TELNYX OUTBOUND CALL
# ─────────────────────────────────────────────────────────────────────────────

async def make_outbound_call(
    to_phone:    str,
    lead_id:     str,
    webhook_url: str,
) -> Optional[str]:
    """
    Initiate outbound call to lead via Telnyx.
    Returns call control ID if successful. None if failed.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                "https://api.telnyx.com/v2/calls",
                headers={
                    "Authorization": f"Bearer {settings.TELNYX_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "connection_id":    settings.TELNYX_API_KEY,
                    "to":               to_phone,
                    "from":             settings.TELNYX_PHONE_NUMBER,
                    "webhook_url":      f"{webhook_url}?lead_id={lead_id}",
                    "webhook_url_method": "POST",
                },
            )
            r.raise_for_status()
            call_id = r.json().get("data", {}).get("call_control_id", "")
            logger.info(
                "COMMUNICATION | Outbound call initiated | lead=%s | call_id=%s",
                lead_id, call_id,
            )
            return call_id

    except Exception as e:
        logger.error(
            "COMMUNICATION | Outbound call FAILED | lead=%s | error=%s",
            lead_id, str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TELNYX SEND SMS
# ─────────────────────────────────────────────────────────────────────────────

async def send_sms(
    to_phone: str,
    message:  str,
    lead_id:  str,
) -> Optional[str]:
    """Send SMS to lead via Telnyx Messaging."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                "https://api.telnyx.com/v2/messages",
                headers={
                    "Authorization": f"Bearer {settings.TELNYX_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "from": settings.TELNYX_PHONE_NUMBER,
                    "to":   to_phone,
                    "text": message[:1600],
                },
            )
            r.raise_for_status()
            msg_id = r.json().get("data", {}).get("id", "")
            logger.info(
                "COMMUNICATION | SMS sent | lead=%s | id=%s",
                lead_id, msg_id,
            )
            return msg_id

    except Exception as e:
        logger.error(
            "COMMUNICATION | SMS FAILED | lead=%s | error=%s",
            lead_id, str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DEEPGRAM STT — SPEECH TO TEXT
# ─────────────────────────────────────────────────────────────────────────────

async def transcribe_audio(audio_url: str) -> Optional[str]:
    """Send audio URL to Deepgram for transcription. Returns transcript text."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                headers={
                    "Authorization": f"Token {settings.DEEPGRAM_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "url":          audio_url,
                    "model":        "nova-2",
                    "smart_format": True,
                    "punctuate":    True,
                    "language":     "en-US",
                },
            )
            response.raise_for_status()
            data = response.json()
            transcript = (
                data
                .get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            )
            logger.info(
                "COMMUNICATION | Transcription complete | length=%s",
                len(transcript),
            )
            return transcript if transcript else None

    except Exception as e:
        logger.error("COMMUNICATION | Transcription FAILED | error=%s", str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TTS — ELEVENLABS PRIMARY, DEEPGRAM FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

async def synthesize_speech(text: str) -> Optional[bytes]:
    """
    TTS with automatic fallback:
      1. ElevenLabs (primary) — ulaw_8000 for Telnyx compatibility
      2. Deepgram Aura (fallback) — free, mulaw 8000Hz
    Returns audio bytes. None if both fail.
    """
    # ── Primary: ElevenLabs ───────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech"
                f"/{settings.ELEVENLABS_VOICE_ID}"
                f"/stream?output_format=ulaw_8000&optimize_streaming_latency=3",
                headers={
                    "xi-api-key":   settings.ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text":     text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {
                        "stability":        0.5,
                        "similarity_boost": 0.75,
                        "style":            0.0,
                        "use_speaker_boost": True,
                    },
                },
            )
            response.raise_for_status()
            logger.info("COMMUNICATION | TTS via ElevenLabs | text_length=%s", len(text))
            return response.content
    except Exception as e:
        logger.warning("COMMUNICATION | ElevenLabs TTS failed — trying Deepgram | error=%s", str(e))

    # ── Fallback: Deepgram Aura TTS ───────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.deepgram.com/v1/speak"
                "?model=aura-asteria-en&encoding=mulaw&sample_rate=8000&container=none",
                headers={
                    "Authorization": f"Token {settings.DEEPGRAM_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={"text": text},
            )
            response.raise_for_status()
            logger.info("COMMUNICATION | TTS via Deepgram fallback | text_length=%s", len(text))
            return response.content
    except Exception as e:
        logger.error("COMMUNICATION | Both TTS providers failed | error=%s", str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LEAD LAST MESSAGE INJECTOR
# ─────────────────────────────────────────────────────────────────────────────

def inject_lead_message(graph_state: dict, lead_message: str) -> dict:
    if not lead_message:
        return graph_state
    messages = graph_state.get("messages", [])
    messages = messages + [{"role": "user", "content": lead_message}]
    return {
        **graph_state,
        "lead_last_message": lead_message,
        "messages": messages,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CALL STATUS UPDATE
# ─────────────────────────────────────────────────────────────────────────────

async def update_lead_call_status(
    lead_id:  str,
    call_sid: str,
    status:   str,
) -> bool:
    try:
        from sqlalchemy import update as sa_update
        from models.lead import LeadRecord

        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    sa_update(LeadRecord)
                    .where(LeadRecord.id == uuid.UUID(lead_id))
                    .values(call_sid=call_sid, status=status)
                )
        logger.info(
            "COMMUNICATION | Lead status updated | lead=%s | status=%s",
            lead_id, status,
        )
        return True

    except Exception as e:
        logger.error(
            "COMMUNICATION | Lead status update FAILED | lead=%s | error=%s",
            lead_id, str(e),
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# VOICEMAIL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def is_voicemail(answered_by: str) -> bool:
    machine_values = {
        "machine_start", "machine_end_beep",
        "machine_end_silence", "machine_end_other", "fax",
    }
    return answered_by in machine_values
