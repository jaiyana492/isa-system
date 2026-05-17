"""
services/communication.py
CorePilora AI — Voice & Messaging Layer

Handles:
- Twilio inbound/outbound calls
- Deepgram STT — speech to text
- ElevenLabs TTS — text to speech
- Lead last message injection into graph state
- Call record updates to database

FIX: Twilio REST client is synchronous.
All blocking Twilio operations wrapped in asyncio.to_thread()
to prevent event loop blocking.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import httpx
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException

from config.settings import settings
from config.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TWILIO CLIENT — SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_twilio_client: Optional[TwilioClient] = None


def get_twilio() -> TwilioClient:
    global _twilio_client
    if _twilio_client is None:
        _twilio_client = TwilioClient(
            settings.TWILIO_ACCOUNT_SID,
            settings.TWILIO_AUTH_TOKEN,
        )
        logger.info("COMMUNICATION | Twilio client initialized")
    return _twilio_client


# ─────────────────────────────────────────────────────────────────────────────
# OUTBOUND CALL
# FIX: Twilio client.calls.create() is synchronous.
# Wrapped in asyncio.to_thread() to prevent event loop blocking.
# ─────────────────────────────────────────────────────────────────────────────

def _sync_make_call(
    to_phone:    str,
    lead_id:     str,
    webhook_url: str,
) -> str:
    """Synchronous Twilio call creation. Runs in thread."""
    client = get_twilio()
    call = client.calls.create(
        to=to_phone,
        from_=settings.TWILIO_PHONE_NUMBER,
        url=f"{webhook_url}?lead_id={lead_id}",
        method="POST",
        timeout=30,
        machine_detection="DetectMessageEnd",
        machine_detection_timeout=5,
    )
    return call.sid


async def make_outbound_call(
    to_phone:    str,
    lead_id:     str,
    webhook_url: str,
) -> Optional[str]:
    """
    Initiate outbound call to lead via Twilio.
    Returns call SID if successful. None if failed.
    Non-blocking — runs Twilio sync call in thread.
    """
    try:
        call_sid = await asyncio.to_thread(
            _sync_make_call,
            to_phone,
            lead_id,
            webhook_url,
        )
        logger.info(
            "COMMUNICATION | Outbound call initiated | lead=%s | sid=%s",
            lead_id, call_sid,
        )
        return call_sid

    except TwilioRestException as e:
        logger.error(
            "COMMUNICATION | Outbound call FAILED | lead=%s | error=%s",
            lead_id, str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SEND SMS
# FIX: Same thread wrapping for SMS.
# ─────────────────────────────────────────────────────────────────────────────

def _sync_send_sms(to_phone: str, message: str) -> str:
    """Synchronous Twilio SMS send. Runs in thread."""
    client = get_twilio()
    msg = client.messages.create(
        to=to_phone,
        from_=settings.TWILIO_PHONE_NUMBER,
        body=message[:1600],
    )
    return msg.sid


async def send_sms(
    to_phone: str,
    message:  str,
    lead_id:  str,
) -> Optional[str]:
    """
    Send SMS to lead via Twilio.
    Returns message SID if successful. None if failed.
    Non-blocking — runs Twilio sync call in thread.
    """
    try:
        msg_sid = await asyncio.to_thread(
            _sync_send_sms,
            to_phone,
            message,
        )
        logger.info(
            "COMMUNICATION | SMS sent | lead=%s | sid=%s",
            lead_id, msg_sid,
        )
        return msg_sid

    except TwilioRestException as e:
        logger.error(
            "COMMUNICATION | SMS FAILED | lead=%s | error=%s",
            lead_id, str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DEEPGRAM STT — SPEECH TO TEXT
# ─────────────────────────────────────────────────────────────────────────────

async def transcribe_audio(audio_url: str) -> Optional[str]:
    """
    Send audio URL to Deepgram for transcription.
    Returns transcript text. None if failed.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                headers={
                    "Authorization": f"Token {settings.DEEPGRAM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "url": audio_url,
                    "model": "nova-2",
                    "smart_format": True,
                    "punctuate": True,
                    "language": "en-US",
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
        logger.error(
            "COMMUNICATION | Transcription FAILED | error=%s",
            str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ELEVENLABS TTS — TEXT TO SPEECH
# ─────────────────────────────────────────────────────────────────────────────

async def synthesize_speech(text: str) -> Optional[bytes]:
    """
    Send Jaiyana's text response to ElevenLabs for synthesis.
    Returns audio bytes for playback via Twilio.
    None if failed — caller uses fallback TTS.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{settings.ELEVENLABS_VOICE_ID}",
                headers={
                    "xi-api-key": settings.ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True,
                    },
                },
            )
            response.raise_for_status()

            logger.info(
                "COMMUNICATION | Speech synthesized | text_length=%s",
                len(text),
            )
            return response.content

    except Exception as e:
        logger.error(
            "COMMUNICATION | TTS FAILED | error=%s",
            str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LEAD LAST MESSAGE INJECTOR
# Solves the graph state problem.
# Populates lead_last_message for appointment detector.
# ─────────────────────────────────────────────────────────────────────────────

def inject_lead_message(
    graph_state:  dict,
    lead_message: str,
) -> dict:
    """
    Inject lead's transcribed message into graph state.
    Called after every Deepgram transcription on live call.
    """
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
    """Update lead call_sid and status in database after call."""
    try:
        from sqlalchemy import update as sa_update
        from models.lead import LeadRecord

        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    sa_update(LeadRecord)
                    .where(LeadRecord.id == uuid.UUID(lead_id))
                    .values(
                        call_sid=call_sid,
                        status=status,
                    )
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
    """
    Determine if call was answered by voicemail/machine.
    Uses Twilio AnsweredBy field.
    """
    machine_values = {
        "machine_start",
        "machine_end_beep",
        "machine_end_silence",
        "machine_end_other",
        "fax",
    }
    return answered_by in machine_values


def _sync_leave_voicemail(call_sid: str, voicemail_url: str) -> None:
    """Synchronous Twilio voicemail redirect. Runs in thread."""
    client = get_twilio()
    client.calls(call_sid).update(
        url=voicemail_url,
        method="POST",
    )


async def leave_voicemail(
    call_sid:          str,
    voicemail_audio_url: str,
) -> bool:
    """
    Play voicemail message when machine detected.
    Non-blocking — runs Twilio sync call in thread.
    """
    try:
        await asyncio.to_thread(
            _sync_leave_voicemail,
            call_sid,
            voicemail_audio_url,
        )
        logger.info(
            "COMMUNICATION | Voicemail left | call_sid=%s",
            call_sid,
        )
        return True

    except TwilioRestException as e:
        logger.error(
            "COMMUNICATION | Voicemail FAILED | call_sid=%s | error=%s",
            call_sid, str(e),
        )
        return False