"""
services/audio_cache.py
CorePilora AI — Pre-Synthesized Audio Response Cache

FIX: Uses dedicated Redis client with decode_responses=False
for binary audio storage. Main Redis client decodes as UTF-8
which corrupts binary MP3 data.
"""

from __future__ import annotations

import hashlib
import logging
import ssl as _ssl_mod
from typing import Optional
from urllib.parse import urlparse

import redis.asyncio as aioredis
from redis.asyncio import Redis

from config.settings import settings

logger = logging.getLogger(__name__)

TTL_AUDIO_CACHE = 86400 * 7  # 7 days


# ─────────────────────────────────────────────────────────────────────────────
# DEDICATED BINARY REDIS CLIENT
# Main Redis client has decode_responses=True — corrupts binary data.
# This client keeps raw bytes intact.
# ─────────────────────────────────────────────────────────────────────────────

_binary_redis: Optional[Redis] = None


async def _get_binary_redis() -> Redis:
    """Redis client for binary data. decode_responses=False."""
    global _binary_redis
    if _binary_redis is None:
        _u = urlparse(settings.REDIS_URL.strip())
        _binary_redis = aioredis.Redis(
            host=_u.hostname,
            port=_u.port or 6379,
            password=_u.password,
            username=_u.username or "default",
            ssl=True,
            ssl_cert_reqs="none",
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )
    return _binary_redis


async def close_binary_redis() -> None:
    """Close binary Redis client on shutdown."""
    global _binary_redis
    if _binary_redis:
        await _binary_redis.aclose()
        _binary_redis = None


# ─────────────────────────────────────────────────────────────────────────────
# COMMON RESPONSES — PRE-SYNTHESIZE THESE
# ─────────────────────────────────────────────────────────────────────────────

CACHEABLE_RESPONSES = [
    "What made you reach out today?",
    "You came to the right place. Tell me more.",
    "What's driving the move right now?",
    "What area are you focused on?",
    "What range are we working within?",
    "Are you pre-approved right now?",
    "Have you connected with anyone else on this yet?",
    "I have Tuesday at 10am or Thursday at 2pm. Which works better for you?",
    "You are locked in. I will send a confirmation shortly.",
    "Perfect. Looking forward to it.",
    "No pressure at all. I will follow up with something useful for you.",
    "Completely understand. I will keep you posted.",
    "I hear you. What would need to change for this to make sense?",
    "Let me ask you this — what does ready look like for you?",
]


# ─────────────────────────────────────────────────────────────────────────────
# KEY BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _audio_key(text: str) -> str:
    text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()
    return f"audio:cache:{text_hash}"


# ─────────────────────────────────────────────────────────────────────────────
# STORE AUDIO
# ─────────────────────────────────────────────────────────────────────────────

async def store_audio(text: str, audio_bytes: bytes) -> bool:
    """Store pre-synthesized audio bytes in Redis."""
    try:
        client = await _get_binary_redis()
        key = _audio_key(text)
        await client.setex(key.encode(), TTL_AUDIO_CACHE, audio_bytes)

        logger.info(
            "AUDIO CACHE | Stored | text_length=%s | audio_bytes=%s",
            len(text), len(audio_bytes),
        )
        return True

    except Exception as e:
        logger.error("AUDIO CACHE | store FAILED | error=%s", str(e))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVE AUDIO
# ─────────────────────────────────────────────────────────────────────────────

async def get_cached_audio(text: str) -> Optional[bytes]:
    """
    Check if audio for this text exists in cache.
    Returns raw audio bytes if found. None if not cached.
    """
    try:
        client = await _get_binary_redis()
        key = _audio_key(text)
        data = await client.get(key.encode())

        if data:
            logger.info("AUDIO CACHE | HIT | key=%s", key)
            return data

        return None

    except Exception as e:
        logger.error("AUDIO CACHE | get FAILED | error=%s", str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN MATCH
# ─────────────────────────────────────────────────────────────────────────────

def find_matching_cached_text(response: str) -> Optional[str]:
    """
    Check if Jaiyana's response matches any cacheable response.
    Exact match for Phase 1.
    """
    response_clean = response.lower().strip()

    for cached_text in CACHEABLE_RESPONSES:
        if cached_text.lower().strip() == response_clean:
            return cached_text

    return None


# ─────────────────────────────────────────────────────────────────────────────
# GET AUDIO — WITH FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

async def get_audio_for_response(response: str) -> Optional[bytes]:
    """
    Get audio for Jaiyana's response.
    Checks cache first. Returns None if not cached — caller
    falls back to ElevenLabs TTS and stores result.
    """
    matched = find_matching_cached_text(response)

    if matched:
        audio = await get_cached_audio(matched)
        if audio:
            return audio

    return None


# ─────────────────────────────────────────────────────────────────────────────
# WARM UP CACHE
# ─────────────────────────────────────────────────────────────────────────────

async def warm_up_cache() -> int:
    """
    Pre-synthesize all cacheable responses using ElevenLabs.
    Called once on system startup.
    Fast-fails on first 401 — skips remaining attempts instead of
    burning 20 seconds on 14 blocked calls.
    Returns count of successfully cached responses.
    """
    from services.communication import synthesize_speech

    cached_count = 0
    tts_available = True

    for text in CACHEABLE_RESPONSES:
        existing = await get_cached_audio(text)
        if existing:
            cached_count += 1
            continue

        if not tts_available:
            continue

        audio = await synthesize_speech(text)
        if audio:
            stored = await store_audio(text, audio)
            if stored:
                cached_count += 1
        else:
            # First failure means TTS is blocked — skip remaining
            logger.warning("AUDIO CACHE | TTS unavailable — skipping warm-up")
            tts_available = False

    logger.info(
        "AUDIO CACHE | Warm up complete | cached=%s/%s",
        cached_count,
        len(CACHEABLE_RESPONSES),
    )
    return cached_count