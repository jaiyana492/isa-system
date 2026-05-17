"""
api/v1/webhook.py
CorePilora AI — Webhook Handlers

Receives leads from:
- Zillow
- Facebook Ads
- Instagram Ads
- Website Forms

Each source sends different payload structure.
Normalizer converts every source into a single clean format
before passing to intake.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request, HTTPException

from core.security import verify_webhook_signature
from services.validator import extract_and_validate_raw
from agents.intake import process_incoming_lead
from services.cache import acquire_lead_lock, release_lead_lock, check_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhooks"])


# ─────────────────────────────────────────────────────────────────────────────
# PAYLOAD NORMALIZER
# Every lead source sends different format.
# This converts all into one consistent structure.
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_zillow(data: dict) -> dict:
    """Normalize Zillow webhook payload."""
    return {
        "full_name":   data.get("name", data.get("full_name", "")),
        "phone":       data.get("phone", ""),
        "email":       data.get("email"),
        "source":      "zillow",
        "market":      data.get("market", ""),
        "message":     data.get("message", data.get("comments", "")),
        "property_url": data.get("property_url", data.get("listing_url", "")),
        "raw":         data,
    }


def _normalize_facebook(data: dict) -> dict:
    """Normalize Facebook Lead Ads webhook payload."""
    field_data = {}
    for field in data.get("field_data", []):
        field_data[field.get("name", "")] = field.get("values", [""])[0]

    return {
        "full_name":   field_data.get("full_name", data.get("full_name", "")),
        "phone":       field_data.get("phone_number", data.get("phone", "")),
        "email":       field_data.get("email", data.get("email")),
        "source":      "facebook",
        "market":      field_data.get("city", data.get("market", "")),
        "message":     field_data.get("comments", data.get("message", "")),
        "raw":         data,
    }


def _normalize_instagram(data: dict) -> dict:
    """Normalize Instagram Lead Ads webhook payload."""
    field_data = {}
    for field in data.get("field_data", []):
        field_data[field.get("name", "")] = field.get("values", [""])[0]

    return {
        "full_name":   field_data.get("full_name", data.get("full_name", "")),
        "phone":       field_data.get("phone_number", data.get("phone", "")),
        "email":       field_data.get("email", data.get("email")),
        "source":      "instagram",
        "market":      field_data.get("city", data.get("market", "")),
        "message":     field_data.get("comments", data.get("message", "")),
        "raw":         data,
    }


def _normalize_website(data: dict) -> dict:
    """Normalize website contact form payload."""
    return {
        "full_name":   data.get("full_name", data.get("name", "")),
        "phone":       data.get("phone", ""),
        "email":       data.get("email"),
        "source":      "website",
        "market":      data.get("market", ""),
        "message":     data.get("message", data.get("comments", data.get("notes", ""))),
        "raw":         data,
    }


NORMALIZERS = {
    "zillow":    _normalize_zillow,
    "facebook":  _normalize_facebook,
    "instagram": _normalize_instagram,
    "website":   _normalize_website,
}


def normalize_payload(source: str, data: dict) -> dict:
    """Route to correct normalizer by source."""
    normalizer = NORMALIZERS.get(source.lower(), _normalize_website)
    return normalizer(data)


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED WEBHOOK ENDPOINT
# Single endpoint receives all sources. Source identified by query param.
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/lead")
async def receive_lead(
    request: Request,
    source: str = "website",
):
    """
    Universal lead webhook endpoint.
    Receives leads from all sources.

    Query params:
        source: zillow | facebook | instagram | website

    Flow:
        1. Validate raw body
        2. Verify webhook signature
        3. Normalize payload
        4. Acquire lead lock
        5. Process through intake
        6. Release lock
        7. Return result
    """
    # Step 1 — Validate raw body
    raw_body = await extract_and_validate_raw(request)

    # Step 2 — Verify signature
    signature = request.headers.get("X-Webhook-Signature", "")
    if not verify_webhook_signature(raw_body, signature):
        logger.warning("WEBHOOK | Invalid signature | source=%s", source)
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # Step 3 — Parse and normalize
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    normalized = normalize_payload(source, data)

    phone = normalized.get("phone", "")
    if not phone:
        raise HTTPException(status_code=400, detail="Phone number required")

    logger.info(
        "WEBHOOK | Lead received | source=%s | phone=%s | name=%s",
        source,
        phone,
        normalized.get("full_name", "unknown"),
    )

    # Step 4 — Rate limit check (max 3 webhook calls per phone per minute)
    if not await check_rate_limit(phone):
        logger.warning("WEBHOOK | Rate limit exceeded | phone=%s", phone)
        raise HTTPException(status_code=429, detail="Too many requests for this number")

    # Step 5 — Acquire lead lock (prevents duplicate parallel processing)
    locked = await acquire_lead_lock(phone)
    if not locked:
        logger.warning("WEBHOOK | Lead already processing | phone=%s", phone)
        return {
            "status": "duplicate",
            "message": "Lead already being processed",
        }

    # Step 6 — Process through intake
    try:
        result = await process_incoming_lead(normalized)
    except Exception as e:
        logger.error(
            "WEBHOOK | Processing failed | source=%s | error=%s",
            source, str(e),
        )
        await release_lead_lock(phone)
        raise HTTPException(status_code=500, detail="Lead processing failed")

    # Step 7 — Release lock
    await release_lead_lock(phone)

    # Step 8 — Return result
    logger.info(
        "WEBHOOK | Lead processed | source=%s | status=%s",
        source,
        result.get("status"),
    )

    return {
        "status": result.get("status"),
        "lead_type": result.get("classification", {}).get("lead_type") if result.get("classification") else None,
        "pipeline": result.get("pipeline_result", {}).get("pipeline") if result.get("pipeline_result") else None,
    }