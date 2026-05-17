"""
services/validator.py
CorePilora AI — Request & Payload Validation

Two responsibilities:
1. extract_and_validate_raw  — FastAPI raw body extraction (size + empty guard)
2. validate_lead_payload     — Dict payload normalization + Pydantic validation
"""

from __future__ import annotations

from fastapi import HTTPException, Request


async def extract_and_validate_raw(request: Request) -> bytes:
    """
    Extract raw body from incoming webhook request.
    Raises 400 if body is empty or malformed.
    Raises 413 if payload exceeds 1MB.
    """
    body = await request.body()

    if not body:
        raise HTTPException(
            status_code=400,
            detail="Empty request body — no lead data received.",
        )

    if len(body) > 1_000_000:
        raise HTTPException(
            status_code=413,
            detail="Payload too large.",
        )

    return body


def validate_lead_payload(payload: dict) -> dict:
    """
    Validate and normalize the webhook payload dict.

    Uses LeadPayload Pydantic model for phone validation (E.164 format).
    Raises ValueError if required fields are missing or invalid.
    Returns clean, normalized dict ready for classification and routing.

    Args:
        payload: Normalized dict from webhook normalizer.

    Returns:
        Validated dict with consistent field names.
    """
    from models.lead import LeadPayload

    full_name = (payload.get("full_name") or payload.get("name") or "").strip()
    phone     = (payload.get("phone") or "").strip()
    email     = payload.get("email")
    source    = (payload.get("source") or "website").strip().lower()
    market    = (payload.get("market") or "dallas_fort_worth").strip().lower()
    message   = (
        payload.get("message")
        or payload.get("notes")
        or payload.get("comments")
        or ""
    )

    if not full_name:
        raise ValueError("full_name is required")

    if not phone:
        raise ValueError("phone is required")

    # Pydantic validates phone → E.164 format
    lead = LeadPayload(
        full_name=full_name,
        phone=phone,
        email=email,
        source=source,
        market=market,
        raw_message=message or None,
    )

    return {
        "full_name":    lead.full_name,
        "phone":        lead.phone,          # E.164 validated
        "email":        lead.email,
        "source":       lead.source,
        "market":       lead.market,
        "message":      lead.raw_message or "",
        "raw":          payload.get("raw", {}),
        "property_url": payload.get("property_url", ""),
    }
