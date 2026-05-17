"""
agents/graph.py
CorePilora AI — Master Graph Orchestrator

Receives routed context. Dispatches to correct sub-graph.
Single source of truth for all graph execution.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def run_master_pipeline(context: dict) -> dict[str, Any]:
    """
    Master orchestrator. Reads lead_type from context.
    Dispatches to correct sub-graph.
    """

    lead_type = context.get("lead_type", "buyer")

    logger.info(
        "MASTER GRAPH | lead_type=%s | market=%s | confidence=%s",
        lead_type,
        context.get("market"),
        context.get("confidence"),
    )

    if lead_type == "buyer":
        from agents.buyer_graph import run_buyer_pipeline
        return await run_buyer_pipeline(context=context)

    if lead_type == "seller":
        from agents.seller_graph import run_seller_pipeline
        return await run_seller_pipeline(context=context)

    if lead_type == "investor":
        from agents.investor_graph import run_investor_pipeline
        return await run_investor_pipeline(context=context)

    # Fallback
    logger.warning("MASTER GRAPH FALLBACK | lead_type=%s", lead_type)
    from agents.buyer_graph import run_buyer_pipeline
    return await run_buyer_pipeline(context=context)