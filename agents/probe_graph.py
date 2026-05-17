"""
agents/probe_graph.py
CorePilora AI — Probe Pipeline
Handles: Low confidence leads — confidence below 40

When the classifier cannot determine lead type with confidence,
Jaiyana runs a reclassification pass, generates a neutral clarifying opening,
then dispatches to the correct sub-pipeline with enriched context.

Flow: initialize → reclassify → probe_open → dispatch → END
The dispatch node calls the real buyer/seller/investor pipeline
and stores the result — no data is lost.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.persona import build_system_prompt
from services.groq_client import call_groq

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class ProbeState(TypedDict):
    lead_id:           str
    lead_name:         str
    market:            str
    source:            str
    original_type:     str   # classifier's best guess (low confidence)
    detected_type:     str   # probe's reclassification
    confidence:        int
    flags:             list[str]
    timeline:          str
    finance_type:      str
    messages:          list[dict]
    turn_count:        int
    pipeline_result:   dict
    current_node:      str
    system_prompt:     str
    last_response:     str
    lead_last_message: str
    lead_payload:      dict


# ─────────────────────────────────────────────────────────────────────────────
# NODE — INITIALIZE
# ─────────────────────────────────────────────────────────────────────────────

async def node_initialize(state: ProbeState) -> ProbeState:
    system_prompt = build_system_prompt(
        lead_type=state.get("original_type", "buyer"),
        market=state.get("market", "Dallas-Fort Worth"),
        lead_source=state.get("source", "website"),
        lead_name=state.get("lead_name", ""),
    )

    logger.info(
        "PROBE | INIT | lead=%s | original_type=%s | confidence=%s | market=%s",
        state.get("lead_name"),
        state.get("original_type"),
        state.get("confidence"),
        state.get("market"),
    )

    return {
        **state,
        "system_prompt":   system_prompt,
        "current_node":    "initialize",
        "turn_count":      0,
        "detected_type":   state.get("original_type", "buyer"),
        "pipeline_result": {},
        "last_response":   "",
        "lead_last_message": state.get("lead_last_message", ""),
        "messages":        state.get("messages", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — RECLASSIFY
# Uses Groq to make a sharper classification than the keyword classifier.
# Produces detected_type: buyer | seller | investor
# ─────────────────────────────────────────────────────────────────────────────

async def node_reclassify(state: ProbeState) -> ProbeState:
    lead_message  = state.get("lead_last_message", "")
    original_type = state.get("original_type", "buyer")
    market        = state.get("market", "unknown")
    source        = state.get("source", "website")

    if not lead_message:
        logger.info("PROBE | RECLASSIFY | No message — defaulting to original_type=%s", original_type)
        return {**state, "detected_type": original_type, "current_node": "reclassify"}

    classification_prompt = """
You are a real estate lead classification engine.
Read the lead's message and determine the most likely lead type.

Respond with ONLY one word — exactly one of:
buyer
seller
investor

Rules:
- buyer: wants to purchase a home to live in
- seller: wants to sell a property they own
- investor: wants to buy properties for rental income, flipping, or portfolio growth

If ambiguous but leaning toward any one type — pick that type. Never respond with anything other than buyer, seller, or investor.
"""

    context_message = (
        f"Lead message: \"{lead_message}\" | "
        f"Source: {source} | Market: {market} | "
        f"Initial classifier guess: {original_type} (low confidence)"
    )

    try:
        raw          = await call_groq(
            system_prompt=classification_prompt,
            messages=[{"role": "user", "content": context_message}],
            temperature=0.1,
            max_tokens=10,
        )
        detected = raw.strip().lower()
        if detected not in ("buyer", "seller", "investor"):
            detected = original_type
    except Exception as e:
        logger.warning("PROBE | RECLASSIFY FAILED | error=%s | fallback=%s", str(e), original_type)
        detected = original_type

    logger.info(
        "PROBE | RECLASSIFIED | original=%s → detected=%s | lead=%s",
        original_type,
        detected,
        state.get("lead_name"),
    )

    return {
        **state,
        "detected_type": detected,
        "current_node":  "reclassify",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — PROBE OPEN
# Neutral opening that works regardless of lead type.
# Clarifies intent without committing to a qualification track prematurely.
# ─────────────────────────────────────────────────────────────────────────────

async def node_probe_open(state: ProbeState) -> ProbeState:
    name              = state.get("lead_name", "")
    market            = state.get("market", "Dallas-Fort Worth")
    source            = state.get("source", "website")
    detected_type     = state.get("detected_type", "buyer")
    lead_last_message = state.get("lead_last_message", "")
    messages          = state.get("messages", [])

    lead_context = (
        f'Their message: "{lead_last_message}". Speak directly to what they shared.'
        if lead_last_message else ""
    )

    instruction = (
        f"Make first contact with {name or 'this lead'} from {source} about {market} real estate. "
        f"{lead_context} "
        f"PROBE OPENING — their intent is not fully clear yet. "
        f"Lean toward {detected_type} angle but keep the door open. "
        f"Open with a pattern interrupt. Do NOT assume their goal. "
        f"Ask one open-ended What or How question that reveals whether they are buying, selling, or investing. "
        f"Under 2 sentences. Human. Direct. No script-reader energy."
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.8,
            max_tokens=100,
        )
    except Exception as e:
        logger.error("PROBE | OPEN FAILED | error=%s", str(e))
        response = (
            f"Hey {name} — Jaiyana here. "
            f"You reached out about {market} real estate — "
            f"what are you trying to accomplish?"
        )

    messages = messages + [{"role": "assistant", "content": response}]

    logger.info(
        "PROBE | OPEN | lead=%s | detected_type=%s",
        state.get("lead_name"),
        detected_type,
    )

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "current_node":  "probe_open",
        "turn_count":    state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — DISPATCH
# Calls the real buyer / seller / investor pipeline with enriched context.
# Stores the full pipeline result in state for the entry point to return.
# ─────────────────────────────────────────────────────────────────────────────

async def node_dispatch(state: ProbeState) -> ProbeState:
    detected_type = state.get("detected_type", "buyer")
    payload       = state.get("lead_payload", {})
    market        = state.get("market", "dallas_fort_worth")

    enriched_context = {
        "lead_type":    detected_type,
        "market":       market,
        "source":       state.get("source", "website"),
        "confidence":   state.get("confidence", 0),
        "timeline":     state.get("timeline", "unknown"),
        "finance_type": state.get("finance_type", "unknown"),
        "flags":        state.get("flags", []),
        "payload":      {
            **payload,
            "probe_opening": state.get("last_response", ""),
        },
    }

    logger.info(
        "PROBE | DISPATCH | lead=%s | type=%s | market=%s",
        state.get("lead_name"),
        detected_type,
        market,
    )

    try:
        if detected_type == "seller":
            from agents.seller_graph import run_seller_pipeline
            result = await run_seller_pipeline(context=enriched_context)
        elif detected_type == "investor":
            from agents.investor_graph import run_investor_pipeline
            result = await run_investor_pipeline(context=enriched_context)
        else:
            from agents.buyer_graph import run_buyer_pipeline
            result = await run_buyer_pipeline(context=enriched_context)

        result["probed_from"] = state.get("original_type", "unknown")
        result["probe_reclassified_to"] = detected_type

    except Exception as e:
        logger.error("PROBE | DISPATCH FAILED | type=%s | error=%s", detected_type, str(e))
        result = {
            "status":          "completed",
            "pipeline":        "probe_fallback",
            "appointment_set": False,
            "needs_nurture":   True,
            "last_response":   state.get("last_response", ""),
            "probed_from":     state.get("original_type", "unknown"),
            "probe_reclassified_to": detected_type,
        }

    return {
        **state,
        "pipeline_result": result,
        "current_node":    "dispatch",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_probe_graph() -> StateGraph:
    graph = StateGraph(ProbeState)

    graph.add_node("initialize",  node_initialize)
    graph.add_node("reclassify",  node_reclassify)
    graph.add_node("probe_open",  node_probe_open)
    graph.add_node("dispatch",    node_dispatch)

    graph.set_entry_point("initialize")

    graph.add_edge("initialize", "reclassify")
    graph.add_edge("reclassify", "probe_open")
    graph.add_edge("probe_open", "dispatch")
    graph.add_edge("dispatch",   END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_probe_graph = None


def get_probe_graph():
    global _probe_graph
    if _probe_graph is None:
        _probe_graph = build_probe_graph()
    return _probe_graph


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT — signature locked, called by lead_type_router
# ─────────────────────────────────────────────────────────────────────────────

async def run_probe_pipeline(context: dict) -> dict:
    payload   = context.get("payload", {})
    full_name = payload.get("full_name", "")
    lead_name = full_name.split()[0] if full_name.strip() else ""

    initial_state: ProbeState = {
        "lead_id":           payload.get("lead_id", ""),
        "lead_name":         lead_name,
        "market":            context.get("market", "dallas_fort_worth"),
        "source":            context.get("source", "website"),
        "original_type":     context.get("lead_type", "buyer"),
        "detected_type":     context.get("lead_type", "buyer"),
        "confidence":        context.get("confidence", 0),
        "flags":             context.get("flags", []),
        "timeline":          context.get("timeline", "unknown"),
        "finance_type":      context.get("finance_type", "unknown"),
        "messages":          [],
        "turn_count":        0,
        "pipeline_result":   {},
        "current_node":      "",
        "system_prompt":     "",
        "last_response":     "",
        "lead_last_message": payload.get("message", "") or "",
        "lead_payload":      payload,
    }

    graph  = get_probe_graph()
    result = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": 30},
    )

    pipeline_result = result.get("pipeline_result", {})

    logger.info(
        "PROBE COMPLETE | original=%s | reclassified=%s | appointment=%s",
        result.get("original_type"),
        result.get("detected_type"),
        pipeline_result.get("appointment_set", False),
    )

    # Merge probe metadata into the sub-pipeline result and return
    return {
        **pipeline_result,
        "pipeline":              "probe",
        "probe_original_type":   result.get("original_type", ""),
        "probe_detected_type":   result.get("detected_type", ""),
        "probe_opening":         result.get("last_response", ""),
        "context":               context,
    }
