"""
services/crm.py
CorePilora AI — HubSpot CRM Integration Layer

Handles:
- Contact creation on lead capture
- Deal stage updates on qualification
- Appointment logging on booking
- Note creation for conversation summaries

FIX: Contact upsert now searches by phone first.
Updates if found. Creates if not.
No fragile error message parsing.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

HUBSPOT_BASE_URL = "https://api.hubapi.com"


def _headers() -> dict:
    """Fresh headers on every request — token never stale."""
    return {
        "Authorization": f"Bearer {settings.HUBSPOT_ACCESS_TOKEN}",
        "Content-Type":  "application/json",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONTACT — SEARCH BY PHONE
# ─────────────────────────────────────────────────────────────────────────────

async def _search_contact_by_phone(phone: str) -> Optional[str]:
    """
    Search HubSpot for existing contact by phone number.
    Returns contact ID if found. None if not found.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{HUBSPOT_BASE_URL}/crm/v3/objects/contacts/search",
                headers=_headers(),
                json={
                    "filterGroups": [{
                        "filters": [{
                            "propertyName": "phone",
                            "operator": "EQ",
                            "value": phone,
                        }]
                    }],
                    "properties": ["phone", "firstname", "lastname"],
                    "limit": 1,
                },
            )
            response.raise_for_status()
            results = response.json().get("results", [])

            if results:
                return results[0].get("id")
            return None

    except Exception as e:
        logger.error(
            "CRM | _search_contact_by_phone FAILED | phone=%s | error=%s",
            phone, str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CONTACT — CREATE OR UPDATE
# FIX: Search first. Update if found. Create if not.
# No fragile 409 error message parsing.
# ─────────────────────────────────────────────────────────────────────────────

async def upsert_contact(
    full_name:   str,
    phone:       str,
    email:       Optional[str],
    market:      str,
    lead_type:   str,
    lead_source: str,
) -> Optional[str]:
    """
    Create or update HubSpot contact.
    Returns HubSpot contact ID if successful. None if failed.
    """
    name_parts = full_name.strip().split(" ", 1)
    first_name = name_parts[0]
    last_name  = name_parts[1] if len(name_parts) > 1 else ""

    properties = {
        "firstname":      first_name,
        "lastname":       last_name,
        "phone":          phone,
        "hs_lead_status": "NEW",
    }
    if email:
        properties["email"] = email

    try:
        # Search first — does this contact already exist?
        existing_id = await _search_contact_by_phone(phone)

        async with httpx.AsyncClient(timeout=10.0) as client:
            if existing_id:
                # Update existing contact
                response = await client.patch(
                    f"{HUBSPOT_BASE_URL}/crm/v3/objects/contacts/{existing_id}",
                    headers=_headers(),
                    json={"properties": properties},
                )
                response.raise_for_status()
                logger.info(
                    "CRM | Contact updated | id=%s | phone=%s",
                    existing_id, phone,
                )
                return existing_id
            else:
                # Create new contact
                response = await client.post(
                    f"{HUBSPOT_BASE_URL}/crm/v3/objects/contacts",
                    headers=_headers(),
                    json={"properties": properties},
                )
                response.raise_for_status()
                contact_id = response.json().get("id")
                logger.info(
                    "CRM | Contact created | id=%s | phone=%s",
                    contact_id, phone,
                )
                return contact_id

    except Exception as e:
        logger.error(
            "CRM | upsert_contact FAILED | phone=%s | error=%s",
            phone, str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DEAL — CREATE
# ─────────────────────────────────────────────────────────────────────────────

async def create_deal(
    contact_id: str,
    deal_name:  str,
    market:     str,
    lead_type:  str,
    pipeline:   str = "default",
    stage:      str = "appointmentscheduled",
) -> Optional[str]:
    """
    Create HubSpot deal and associate with contact.
    Returns deal ID if successful. None if failed.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{HUBSPOT_BASE_URL}/crm/v3/objects/deals",
                headers=_headers(),
                json={
                    "properties": {
                        "dealname":  deal_name,
                        "pipeline":  pipeline,
                        "dealstage": stage,
                    }
                },
            )
            response.raise_for_status()
            deal_id = response.json().get("id")

            # Associate deal with contact
            await client.put(
                f"{HUBSPOT_BASE_URL}/crm/v3/objects/deals/{deal_id}"
                f"/associations/contacts/{contact_id}/3",
                headers=_headers(),
            )

            logger.info(
                "CRM | Deal created | deal_id=%s | contact_id=%s",
                deal_id, contact_id,
            )
            return deal_id

    except Exception as e:
        logger.error(
            "CRM | create_deal FAILED | contact=%s | error=%s",
            contact_id, str(e),
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DEAL — UPDATE STAGE
# ─────────────────────────────────────────────────────────────────────────────

async def update_deal_stage(
    deal_id: str,
    stage:   str,
) -> bool:
    """Update HubSpot deal stage."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.patch(
                f"{HUBSPOT_BASE_URL}/crm/v3/objects/deals/{deal_id}",
                headers=_headers(),
                json={"properties": {"dealstage": stage}},
            )
            response.raise_for_status()

            logger.info(
                "CRM | Deal stage updated | deal_id=%s | stage=%s",
                deal_id, stage,
            )
            return True

    except Exception as e:
        logger.error(
            "CRM | update_deal_stage FAILED | deal_id=%s | error=%s",
            deal_id, str(e),
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# NOTE — CREATE
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# CONTACT STATUS UPDATE
# ─────────────────────────────────────────────────────────────────────────────

async def update_contact_status(
    contact_id: str,
    status:     str,
) -> bool:
    """Update lead status on HubSpot contact."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.patch(
                f"{HUBSPOT_BASE_URL}/crm/v3/objects/contacts/{contact_id}",
                headers=_headers(),
                json={"properties": {"hs_lead_status": status.upper()}},
            )
            response.raise_for_status()

            logger.info(
                "CRM | Contact status updated | id=%s | status=%s",
                contact_id, status,
            )
            return True

    except Exception as e:
        logger.error(
            "CRM | update_contact_status FAILED | id=%s | error=%s",
            contact_id, str(e),
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# FULL LEAD SYNC
# ─────────────────────────────────────────────────────────────────────────────

async def sync_lead_to_crm(
    full_name:   str,
    phone:       str,
    email:       Optional[str],
    market:      str,
    lead_type:   str,
    lead_source: str,
    lpmama:      Optional[dict] = None,
    notes:       Optional[str]  = None,
) -> Optional[str]:
    """
    Full CRM sync for incoming lead.
    Creates/updates contact + creates deal in HubSpot.
    LPMAMA qualification data lives in PostgreSQL (conversation_sessions).
    Returns HubSpot contact ID. None if failed.
    """
    contact_id = await upsert_contact(
        full_name=full_name,
        phone=phone,
        email=email,
        market=market,
        lead_type=lead_type,
        lead_source=lead_source,
    )

    if contact_id:
        market_display = market.replace("_", " ").title()
        deal_name      = f"{full_name} — {market_display} {lead_type.title()}"
        await create_deal(
            contact_id = contact_id,
            deal_name  = deal_name,
            market     = market,
            lead_type  = lead_type,
        )

    return contact_id