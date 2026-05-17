"""
services/lead_type_router.py
CorePilora AI — Real Estate Lead Intelligent System (ISA)

Routes a fully classified lead to the correct LangGraph pipeline.
Carries the full ClassificationResult — not just confidence.
Every pipeline receives: lead type, market, timeline, finance,
source, flags, and raw scores. No data left behind at the gate.
"""

from __future__ import annotations

import logging
from typing import Any

from services.classifier import ClassificationResult, LeadType, Timeline, Market

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING CONTEXT
# Passed into every pipeline. Single source of truth for the graph.
# ─────────────────────────────────────────────────────────────────────────────

def _build_routing_context(
    result: ClassificationResult,
    payload: dict,
) -> dict[str, Any]:
    """
    Merge webhook payload with full classification intelligence.
    Every downstream graph node reads from this — nothing reconstructed.
    """
    return {
        # Raw lead payload (webhook body from Zillow, FB, IG, website)
        "payload": payload,

        # Classification core
        "lead_type":    result.lead_type.value,
        "confidence":   result.confidence,
        "is_qualified": result.is_qualified(),

        # Market intelligence
        "market":       result.market.value,

        # Urgency
        "timeline":     result.timeline.value,
        "is_hot":       result.is_hot(),

        # Finance position
        "finance_type": result.finance_type.value,
        "needs_lender": result.needs_lender(),

        # Lead origin
        "source":       result.source.value,

        # Escalation flags — consumed by graph nodes for special routing
        "flags":        result.flags,

        # Raw scores — available for LLM prompt injection and audit logs
        "score_breakdown": result.score_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRE-ROUTE ESCALATION CHECK
# Some leads must be intercepted before the standard pipeline runs.
# Distressed, high-motivation, or hard-deadline leads need immediate handling.
# ─────────────────────────────────────────────────────────────────────────────

def _requires_escalation(result: ClassificationResult) -> bool:
    escalation_flags = {
        "DISTRESSED_SELLER",
        "HIGH_MOTIVATION",
        "HOT_INVESTOR_CLOSE",
    }
    return bool(escalation_flags.intersection(set(result.flags)))


async def _escalate(
    context: dict,
    reason: str,
) -> dict:
    """
    Route flagged lead to the escalation pipeline.
    Logs the flag reason. Graph handles the actual override behavior.
    """
    logger.warning(
        "ESCALATION TRIGGERED | lead_type=%s | flags=%s | reason=%s",
        context["lead_type"],
        context["flags"],
        reason,
    )
    from agents.escalation_graph import run_escalation_pipeline
    return await run_escalation_pipeline(context=context)


# ─────────────────────────────────────────────────────────────────────────────
# LOW CONFIDENCE HANDLER
# If the classifier isn't sure, the ISA probes before committing to a track.
# ─────────────────────────────────────────────────────────────────────────────

_LOW_CONFIDENCE_THRESHOLD = 40


async def _handle_low_confidence(context: dict) -> dict:
    """
    Confidence below threshold → probe graph asks clarifying questions
    before committing the lead to buyer / seller / investor track.
    """
    logger.info(
        "LOW CONFIDENCE ROUTE | confidence=%s | lead_type=%s",
        context["confidence"],
        context["lead_type"],
    )
    from agents.probe_graph import run_probe_pipeline
    return await run_probe_pipeline(context=context)


# ─────────────────────────────────────────────────────────────────────────────
# MARKET GUARD
# Dubai is future expansion only. Block it from live pipelines.
# ─────────────────────────────────────────────────────────────────────────────

def _is_active_market(market: str) -> bool:
    active = {
        Market.DFW.value,
        Market.HOUSTON.value,
        Market.ORLANDO.value,
        Market.TAMPA.value,
        Market.MIAMI.value,
    }
    return market in active


async def _handle_inactive_market(context: dict) -> dict:
    """
    Lead detected in Dubai or unknown market.
    Log, tag, and route to nurture — do not burn with live pipeline.
    """
    logger.info(
        "INACTIVE MARKET | market=%s | lead_type=%s",
        context["market"],
        context["lead_type"],
    )
    from agents.nurture_graph import run_nurture_pipeline
    return await run_nurture_pipeline(context=context)


# ─────────────────────────────────────────────────────────────────────────────
# NURTURE CHECK
# Long-timeline leads don't get the full close sequence.
# They enter a drip nurture pipeline and get re-evaluated on re-engagement.
# ─────────────────────────────────────────────────────────────────────────────

def _is_nurture_only(result: ClassificationResult) -> bool:
    return (
        result.timeline == Timeline.NURTURE
        and result.confidence < 60
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────

async def route_lead(
    result: ClassificationResult,
    payload: dict,
) -> dict:
    """
    Route a classified lead to the correct LangGraph pipeline.

    Decision order (strict — do not reorder):
      1. Escalation flags          → escalation_graph
      2. Inactive market (Dubai)   → nurture_graph
      3. Low confidence            → probe_graph  (clarify before committing)
      4. Nurture-only timeline     → nurture_graph
      5. BUYER                     → buyer_graph
      6. SELLER                    → seller_graph
      7. INVESTOR                  → investor_graph
      8. Fallback (UNKNOWN)        → buyer_graph  (highest volume default)

    Every pipeline receives the full routing context — not a stripped dict.

    Args:
        result:   Full ClassificationResult from classify_lead().
        payload:  Raw webhook payload dict (Zillow, FB, IG, website form).

    Returns:
        Pipeline execution result dict.
    """

    context = _build_routing_context(result, payload)

    logger.info(
        "ROUTING LEAD | type=%s | confidence=%s | market=%s | "
        "timeline=%s | finance=%s | source=%s | flags=%s",
        context["lead_type"],
        context["confidence"],
        context["market"],
        context["timeline"],
        context["finance_type"],
        context["source"],
        context["flags"],
    )

    # ── 1. Escalation ──────────────────────────────────────────────────────
    if _requires_escalation(result):
        return await _escalate(
            context=context,
            reason=", ".join(result.flags),
        )

    # ── 2. Inactive market ─────────────────────────────────────────────────
    if not _is_active_market(context["market"]):
        return await _handle_inactive_market(context)

    # ── 3. Low confidence — probe before committing ────────────────────────
    if result.confidence < _LOW_CONFIDENCE_THRESHOLD:
        return await _handle_low_confidence(context)

    # ── 4. Nurture-only ────────────────────────────────────────────────────
    if _is_nurture_only(result):
        logger.info(
            "NURTURE ROUTE | timeline=%s | confidence=%s",
            context["timeline"],
            context["confidence"],
        )
        from agents.nurture_graph import run_nurture_pipeline
        return await run_nurture_pipeline(context=context)

    # ── 5. BUYER ───────────────────────────────────────────────────────────
    if result.lead_type == LeadType.BUYER:
        from agents.buyer_graph import run_buyer_pipeline
        return await run_buyer_pipeline(context=context)

    # ── 6. SELLER ──────────────────────────────────────────────────────────
    if result.lead_type == LeadType.SELLER:
        from agents.seller_graph import run_seller_pipeline
        return await run_seller_pipeline(context=context)

    # ── 7. INVESTOR ────────────────────────────────────────────────────────
    if result.lead_type == LeadType.INVESTOR:
        from agents.investor_graph import run_investor_pipeline
        return await run_investor_pipeline(context=context)

    # ── 8. Fallback — UNKNOWN or unhandled ────────────────────────────────
    logger.warning(
        "FALLBACK ROUTE | lead_type=%s | confidence=%s | payload_keys=%s",
        context["lead_type"],
        context["confidence"],
        list(payload.keys()),
    )
    from agents.buyer_graph import run_buyer_pipeline
    return await run_buyer_pipeline(context=context)