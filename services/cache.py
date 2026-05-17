"""
services/cache.py
CorePilora AI — Redis Cache Operations Layer

All Redis read/write operations live here.
No other file touches Redis directly.
Every operation has error handling and logging.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from config.redis import (
    get_redis,
    key_conversation_state,
    key_lead_cache,
    key_lead_lock,
    key_lpmama,
    key_nurture_queue,
    key_rate_limit,
    key_session,
    TTL_SESSION,
    TTL_LEAD_CACHE,
    TTL_LEAD_LOCK,
    TTL_LPMAMA,
    TTL_NURTURE_JOB,
    TTL_RATE_LIMIT,
    TTL_CONV_STATE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

async def set_session(lead_id: str, state: dict) -> bool:
    """Write full session state to Redis after every turn."""
    try:
        client = await get_redis()
        await client.setex(
            key_session(lead_id),
            TTL_SESSION,
            json.dumps(state),
        )
        return True
    except Exception as e:
        logger.error("CACHE | set_session FAILED | lead=%s | error=%s", lead_id, str(e))
        return False


async def get_session(lead_id: str) -> Optional[dict]:
    """Read session state. Returns None if expired or not found."""
    try:
        client = await get_redis()
        raw = await client.get(key_session(lead_id))
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.error("CACHE | get_session FAILED | lead=%s | error=%s", lead_id, str(e))
        return None


async def delete_session(lead_id: str) -> bool:
    """Clear session state after conversation ends."""
    try:
        client = await get_redis()
        await client.delete(key_session(lead_id))
        return True
    except Exception as e:
        logger.error("CACHE | delete_session FAILED | lead=%s | error=%s", lead_id, str(e))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LEAD CONTEXT CACHE
# Zero DB reads during live call — everything from Redis.
# ─────────────────────────────────────────────────────────────────────────────

async def set_lead_cache(lead_id: str, lead_data: dict) -> bool:
    """Cache lead record when lead first enters system."""
    try:
        client = await get_redis()
        await client.setex(
            key_lead_cache(lead_id),
            TTL_LEAD_CACHE,
            json.dumps(lead_data),
        )
        return True
    except Exception as e:
        logger.error("CACHE | set_lead_cache FAILED | lead=%s | error=%s", lead_id, str(e))
        return False


async def get_lead_cache(lead_id: str) -> Optional[dict]:
    """Read cached lead data. Returns None if expired."""
    try:
        client = await get_redis()
        raw = await client.get(key_lead_cache(lead_id))
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.error("CACHE | get_lead_cache FAILED | lead=%s | error=%s", lead_id, str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LEAD LOCK
# Prevents two processes hitting same lead simultaneously.
# Critical when Twilio webhook and inbound call fire at same time.
# ─────────────────────────────────────────────────────────────────────────────

async def acquire_lead_lock(phone: str) -> bool:
    """
    Attempt to acquire lock on lead by phone number.
    Returns True if lock acquired.
    Returns False if lead already being processed.
    SET NX — atomic. No race condition.
    """
    try:
        client = await get_redis()
        result = await client.set(
            key_lead_lock(phone),
            "locked",
            nx=True,
            ex=TTL_LEAD_LOCK,
        )
        acquired = result is True
        if not acquired:
            logger.warning("CACHE | Lead already locked | phone=%s", phone)
        return acquired
    except Exception as e:
        logger.error("CACHE | acquire_lead_lock FAILED | phone=%s | error=%s", phone, str(e))
        return False


async def release_lead_lock(phone: str) -> bool:
    """Release lead lock after processing complete."""
    try:
        client = await get_redis()
        await client.delete(key_lead_lock(phone))
        return True
    except Exception as e:
        logger.error("CACHE | release_lead_lock FAILED | phone=%s | error=%s", phone, str(e))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LPMAMA STATE
# ─────────────────────────────────────────────────────────────────────────────

async def set_lpmama(lead_id: str, lpmama: dict) -> bool:
    """Write LPMAMA state to Redis."""
    try:
        client = await get_redis()
        await client.setex(
            key_lpmama(lead_id),
            TTL_LPMAMA,
            json.dumps(lpmama),
        )
        return True
    except Exception as e:
        logger.error("CACHE | set_lpmama FAILED | lead=%s | error=%s", lead_id, str(e))
        return False


async def get_lpmama(lead_id: str) -> Optional[dict]:
    """Read LPMAMA state from Redis."""
    try:
        client = await get_redis()
        raw = await client.get(key_lpmama(lead_id))
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.error("CACHE | get_lpmama FAILED | lead=%s | error=%s", lead_id, str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION STATE
# Full conversation state including message history.
# ─────────────────────────────────────────────────────────────────────────────

async def set_conversation_state(session_id: str, state: dict) -> bool:
    """Write full conversation state including message history."""
    try:
        client = await get_redis()
        await client.setex(
            key_conversation_state(session_id),
            TTL_CONV_STATE,
            json.dumps(state),
        )
        return True
    except Exception as e:
        logger.error("CACHE | set_conversation_state FAILED | session=%s | error=%s", session_id, str(e))
        return False


async def get_conversation_state(session_id: str) -> Optional[dict]:
    """Read full conversation state."""
    try:
        client = await get_redis()
        raw = await client.get(key_conversation_state(session_id))
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.error("CACHE | get_conversation_state FAILED | session=%s | error=%s", session_id, str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITING
# Prevents calling same lead too frequently.
# ─────────────────────────────────────────────────────────────────────────────

async def check_rate_limit(phone: str, max_calls: int = 3) -> bool:
    """
    Check if phone number is within call rate limit.
    Returns True if call is allowed.
    Returns False if rate limit exceeded.
    Max 3 calls per minute per number by default.
    """
    try:
        client  = await get_redis()
        rk      = key_rate_limit(phone)
        current = await client.get(rk)

        if current is None:
            await client.setex(rk, TTL_RATE_LIMIT, 1)
            return True

        count = int(current)
        if count >= max_calls:
            logger.warning(
                "CACHE | Rate limit exceeded | phone=%s | count=%s",
                phone, count,
            )
            return False

        await client.incr(rk)
        return True

    except Exception as e:
        logger.error("CACHE | check_rate_limit FAILED | phone=%s | error=%s", phone, str(e))
        # Fail open — never block calls on Redis error
        return True


# ─────────────────────────────────────────────────────────────────────────────
# NURTURE QUEUE
# ─────────────────────────────────────────────────────────────────────────────

async def queue_nurture(lead_id: str, nurture_data: dict) -> bool:
    """
    Add lead to nurture queue.
    nurture_data: sequence_type, next_contact_at, channel, message.
    """
    try:
        client = await get_redis()
        await client.setex(
            key_nurture_queue(lead_id),
            TTL_NURTURE_JOB,
            json.dumps(nurture_data),
        )
        logger.info("CACHE | Nurture queued | lead=%s", lead_id)
        return True
    except Exception as e:
        logger.error("CACHE | queue_nurture FAILED | lead=%s | error=%s", lead_id, str(e))
        return False


async def get_nurture_job(lead_id: str) -> Optional[dict]:
    """Read nurture job for a lead."""
    try:
        client = await get_redis()
        raw = await client.get(key_nurture_queue(lead_id))
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.error("CACHE | get_nurture_job FAILED | lead=%s | error=%s", lead_id, str(e))
        return None


async def remove_nurture_job(lead_id: str) -> bool:
    """Remove nurture job after processed."""
    try:
        client = await get_redis()
        await client.delete(key_nurture_queue(lead_id))
        return True
    except Exception as e:
        logger.error("CACHE | remove_nurture_job FAILED | lead=%s | error=%s", lead_id, str(e))
        return False