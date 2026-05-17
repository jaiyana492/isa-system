"""
agents/seller_graph.py
CorePilora AI — Seller Closing Pipeline

Full LangGraph implementation.
Groq LLM via shared services/groq_client.py — model driven by settings.GROQ_MODEL.
Seller psychology: trust first, value second, urgency third.
Appointment detection reads LEAD input, not Jaiyana output.
LPMAMA extraction runs every 3 turns — not every turn.
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.persona import build_system_prompt
from agents.utils import lead_confirmed_appointment
from services.groq_client import call_groq

logger = logging.getLogger(__name__)

DEFAULT_SELLER_LPMAMA = {
    "location":          None,
    "price_expectation": None,
    "motivation":        None,
    "agent":             None,
    "mortgage_balance":  None,
    "appointment":       None,
}


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class SellerState(TypedDict):
    lead_id:           str
    lead_name:         str
    market:            str
    source:            str
    timeline:          str
    confidence:        int
    flags:             list[str]
    messages:          list[dict]
    turn_count:        int
    lpmama:            dict
    is_distressed:     bool
    equity_position:   str
    objection_count:   int
    appointment_set:   bool
    needs_nurture:     bool
    current_node:      str
    system_prompt:     str
    last_response:     str
    lead_last_message: str


# ─────────────────────────────────────────────────────────────────────────────
# LPMAMA EXTRACTOR — runs every 3 turns
# ─────────────────────────────────────────────────────────────────────────────

async def _extract_seller_lpmama(
    conversation:   list[dict],
    current_lpmama: dict,
    turn_count:     int,
) -> dict:
    if turn_count % 3 != 0:
        return current_lpmama

    extraction_prompt = """
You are a data extraction engine for a real estate ISA system.
Read the conversation and extract confirmed seller qualification data.

Return ONLY valid JSON with these exact keys:
{
  "location": "confirmed property address or area or null",
  "price_expectation": "confirmed price expectation or null",
  "motivation": "confirmed reason for selling or null",
  "agent": "confirmed yes/no working with agent or null",
  "mortgage_balance": "confirmed mortgage balance or equity position or null",
  "appointment": "confirmed appointment time or null"
}

Only include values the seller explicitly stated.
Null if not confirmed. JSON only. No explanation. No markdown.
"""
    try:
        raw       = await call_groq(
            system_prompt=extraction_prompt,
            messages=conversation,
            temperature=0.1,
            max_tokens=200,
        )
        clean     = raw.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(clean)
        updated   = current_lpmama.copy()
        for key in updated:
            if updated[key] is None and extracted.get(key):
                updated[key] = extracted[key]
        return updated
    except Exception as e:
        logger.warning("SELLER | LPMAMA extraction failed | error=%s", str(e))
        return current_lpmama


# ─────────────────────────────────────────────────────────────────────────────
# NODE — INITIALIZE
# ─────────────────────────────────────────────────────────────────────────────

async def node_initialize(state: SellerState) -> SellerState:
    is_distressed = any(f in state.get("flags", []) for f in [
        "DISTRESSED_SELLER", "HIGH_MOTIVATION"
    ])

    system_prompt = build_system_prompt(
        lead_type="seller",
        market=state.get("market", "Dallas-Fort Worth"),
        lead_source=state.get("source", "website"),
        lead_name=state.get("lead_name", ""),
        lpmama_state=state.get("lpmama", DEFAULT_SELLER_LPMAMA.copy()),
    )

    logger.info(
        "SELLER | INIT | lead=%s | market=%s | distressed=%s",
        state.get("lead_name"),
        state.get("market"),
        is_distressed,
    )

    return {
        **state,
        "system_prompt":     system_prompt,
        "current_node":      "initialize",
        "is_distressed":     is_distressed,
        "equity_position":   state.get("equity_position", "unknown"),
        "turn_count":        0,
        "objection_count":   0,
        "appointment_set":   False,
        "needs_nurture":     False,
        "last_response":     "",
        "lead_last_message": "",
        "messages":          state.get("messages", []),
        "lpmama":            state.get("lpmama", DEFAULT_SELLER_LPMAMA.copy()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — OPENING
# ─────────────────────────────────────────────────────────────────────────────

async def node_opening(state: SellerState) -> SellerState:
    source            = state.get("source", "website")
    name              = state.get("lead_name", "")
    market            = state.get("market", "Dallas-Fort Worth")
    is_distressed     = state.get("is_distressed", False)
    messages          = state.get("messages", [])
    lead_last_message = state.get("lead_last_message", "")

    lead_context = (
        f'Their message: "{lead_last_message}". Speak directly to what they revealed.'
        if lead_last_message else ""
    )
    instruction = (
        f"Make first contact with {name or 'this seller'}. "
        f"Source: {source}. Market: {market}. Distressed: {is_distressed}. {lead_context} "
        f"If distressed: label the emotion first — empathy then certainty. "
        f"If standard: curiosity hook around what their home is worth right now in {market}. "
        f"Under 2 sentences. Human. Direct. End with one question."
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.8,
            max_tokens=80,
        )
    except Exception as e:
        logger.error("SELLER | OPENING FAILED | error=%s", str(e))
        response = (
            f"Hey {name} — Jaiyana here. "
            f"You reached out about selling in {market} — what prompted the decision?"
        )

    messages = messages + [{"role": "assistant", "content": response}]

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "current_node":  "opening",
        "turn_count":    state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — MARKET VALUE
# ─────────────────────────────────────────────────────────────────────────────

async def node_market_value(state: SellerState) -> SellerState:
    market            = state.get("market", "Dallas-Fort Worth")
    messages          = state.get("messages", [])
    lead_last_message = state.get("lead_last_message", "")

    lead_said = f'The seller said: "{lead_last_message}".' if lead_last_message else ""
    instruction = (
        f"{lead_said} "
        f"ACKNOWLEDGE what they revealed first — mirror a key word they used, or label what you hear underneath (wanting certainty, fear of leaving money, urgency). "
        f"Then build authority around {market} current conditions — reference specific market dynamics, not vague language. "
        f"Do not give a number yet — you need more information to do it right. "
        f"Position as the expert who gets maximum net proceeds. "
        f"Under 4 sentences. End with one natural How/What question advancing to motivation."
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.7,
            max_tokens=180,
        )
    except Exception as e:
        logger.error("SELLER | MARKET VALUE FAILED | error=%s", str(e))
        response = (
            f"In {market} right now values are moving — "
            f"but what your home is worth specifically depends on a few key factors. "
            f"Tell me about the property and what is driving the decision to sell."
        )

    messages = messages + [{"role": "assistant", "content": response}]

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "current_node":  "market_value",
        "turn_count":    state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — QUALIFY
# ─────────────────────────────────────────────────────────────────────────────

async def node_qualify(state: SellerState) -> SellerState:
    lpmama            = state.get("lpmama", {})
    missing           = [k for k, v in lpmama.items() if v is None]
    messages          = state.get("messages", [])
    lead_last_message = state.get("lead_last_message", "")

    if lead_confirmed_appointment(lead_last_message):
        logger.info("SELLER | APPOINTMENT CONFIRMED BY LEAD")
        return {
            **state,
            "appointment_set": True,
            "current_node":    "qualify",
            "turn_count":      state.get("turn_count", 0) + 1,
        }

    lead_said = f'The seller just said: "{lead_last_message}"' if lead_last_message else ""
    next_gate = missing[0].upper() if missing else "APPOINTMENT CLOSE"
    instruction = (
        f"{lead_said} "
        f"TURN ARCHITECTURE — EXECUTE THIS EXACTLY: "
        f"(1) ACKNOWLEDGE first — mirror their last 2-3 words as a soft question, OR label what you sense underneath ('It sounds like...', 'It seems like...'). Never skip this. "
        f"(2) DEEPEN if they revealed a life trigger, urgency driver, or emotional charge — one follow-up before advancing. Motivation is the master key for sellers. "
        f"(3) ONE focused How/What question advancing toward: {next_gate}. "
        f"STOP. 2 sentences max."
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )
    except Exception as e:
        logger.error("SELLER | QUALIFY FAILED | error=%s", str(e))
        response = "What is the main reason you are thinking about selling right now?"

    messages = messages + [{"role": "assistant", "content": response}]

    turn_count     = state.get("turn_count", 0)
    updated_lpmama = await _extract_seller_lpmama(
        conversation=messages,
        current_lpmama=lpmama,
        turn_count=turn_count,
    )

    logger.info(
        "SELLER | QUALIFY | turn=%s | collected=%s | missing=%s",
        turn_count,
        [k for k, v in updated_lpmama.items() if v],
        [k for k, v in updated_lpmama.items() if not v],
    )

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "lpmama":        updated_lpmama,
        "current_node":  "qualify",
        "turn_count":    turn_count + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — HANDLE OBJECTION
# ─────────────────────────────────────────────────────────────────────────────

async def node_handle_objection(state: SellerState) -> SellerState:
    count             = state.get("objection_count", 0)
    market            = state.get("market", "Dallas-Fort Worth")
    messages          = state.get("messages", [])
    lead_last_message = state.get("lead_last_message", "")

    lead_said = f'Their exact pushback: "{lead_last_message}"' if lead_last_message else "Seller pushed back."
    instruction = (
        f"{lead_said} Objection #{count + 1}. Market: {market}. "
        f"TURN ARCHITECTURE: "
        f"(1) LABEL the real fear underneath this objection first — 'It sounds like...' or 'It seems like...'. Seller fears: wrong price, wrong agent, wrong timing. "
        f"(2) Counter with a completely different angle — price objection = net proceeds math; agent objection = positioning; not ready = cost of delay. "
        f"(3) End with one calibrated How/What question. "
        f"Under 3 sentences. Certainty. Zero desperation."
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.75,
            max_tokens=180,
        )
    except Exception as e:
        logger.error("SELLER | OBJECTION FAILED | error=%s", str(e))
        response = "I completely understand. What would make this feel like the right time for you?"

    messages = messages + [{"role": "assistant", "content": response}]

    return {
        **state,
        "messages":        messages,
        "last_response":   response,
        "objection_count": count + 1,
        "current_node":    "handle_objection",
        "turn_count":      state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — CLOSE APPOINTMENT
# ─────────────────────────────────────────────────────────────────────────────

async def node_close_appointment(state: SellerState) -> SellerState:
    name     = state.get("lead_name", "")
    market   = state.get("market", "Dallas-Fort Worth")
    lpmama   = state.get("lpmama", {})
    messages = state.get("messages", [])

    instruction = (
        f"Qualification complete for {name or 'this seller'} in {market}. "
        f"Data: {json.dumps(lpmama)}. "
        f"Offer the listing consultation now. "
        f"Frame as: property walkthrough + market analysis + net proceeds sheet. "
        f"Two specific time options. Do NOT confirm the appointment yourself."
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.6,
            max_tokens=180,
        )
    except Exception as e:
        logger.error("SELLER | CLOSE FAILED | error=%s", str(e))
        response = (
            f"Here is what I want to do {name} — "
            f"I want to walk through your home, run a full market analysis, "
            f"and show you exactly what you will net. "
            f"I have Tuesday at 10am or Thursday at 2pm. Which works better?"
        )

    messages = messages + [{"role": "assistant", "content": response}]
    logger.info("SELLER | CLOSE OFFERED | lead=%s", state.get("lead_name"))

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "current_node":  "close_appointment",
        "turn_count":    state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — CONFIRM APPOINTMENT
# ─────────────────────────────────────────────────────────────────────────────

async def node_confirm_appointment(state: SellerState) -> SellerState:
    name     = state.get("lead_name", "")
    messages = state.get("messages", [])

    instruction = (
        f"{name or 'Seller'} just confirmed the listing consultation. "
        f"Send brief warm confirmation. Confirm time and what you will bring — "
        f"market analysis and net sheet. Under 3 sentences. Professional. Human."
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.6,
            max_tokens=120,
        )
    except Exception as e:
        logger.error("SELLER | CONFIRM FAILED | error=%s", str(e))
        response = (
            f"You are locked in {name}. "
            f"I will bring the full market analysis and your net proceeds sheet. See you then."
        )

    messages = messages + [{"role": "assistant", "content": response}]
    logger.info("SELLER | CONFIRMED | lead=%s", state.get("lead_name"))

    return {
        **state,
        "messages":        messages,
        "last_response":   response,
        "appointment_set": True,
        "current_node":    "confirm_appointment",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — ROUTE TO NURTURE
# ─────────────────────────────────────────────────────────────────────────────

async def node_route_nurture(state: SellerState) -> SellerState:
    name     = state.get("lead_name", "")
    market   = state.get("market", "Dallas-Fort Worth")
    messages = state.get("messages", [])

    instruction = (
        f"{name or 'This seller'} is not ready right now in {market}. "
        f"Exit gracefully. Reference market update follow-up. "
        f"Under 2 sentences. Warm. Zero desperation."
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.7,
            max_tokens=80,
        )
    except Exception as e:
        logger.error("SELLER | NURTURE EXIT FAILED | error=%s", str(e))
        response = (
            f"Completely understand {name}. "
            f"I will keep you posted on what is happening in {market} — "
            f"you will want to know when the right moment hits."
        )

    messages = messages + [{"role": "assistant", "content": response}]
    logger.info(
        "SELLER | NURTURE | lead=%s | objections=%s",
        state.get("lead_name"),
        state.get("objection_count"),
    )

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "needs_nurture": True,
        "current_node":  "route_nurture",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────

def route_after_opening(state: SellerState) -> str:
    if state.get("is_distressed"):
        return "qualify"
    return "market_value"


def route_after_qualify(state: SellerState) -> str:
    if state.get("appointment_set"):
        return "confirm_appointment"

    lpmama        = state.get("lpmama", {})
    all_collected = all(v is not None for v in lpmama.values())

    if all_collected:
        return "close_appointment"
    if state.get("objection_count", 0) >= 3 and state.get("timeline", "") == "nurture":
        return "route_nurture"
    if state.get("turn_count", 0) >= 20:
        return "route_nurture"
    return "qualify"


def route_after_objection(state: SellerState) -> str:
    if state.get("objection_count", 0) >= 5:
        return "route_nurture"
    return "qualify"


def route_after_close(state: SellerState) -> str:
    if state.get("appointment_set"):
        return "confirm_appointment"
    if state.get("objection_count", 0) >= 3:
        return "route_nurture"
    return "handle_objection"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_seller_graph() -> StateGraph:
    graph = StateGraph(SellerState)

    graph.add_node("initialize",          node_initialize)
    graph.add_node("opening",             node_opening)
    graph.add_node("market_value",        node_market_value)
    graph.add_node("qualify",             node_qualify)
    graph.add_node("handle_objection",    node_handle_objection)
    graph.add_node("close_appointment",   node_close_appointment)
    graph.add_node("confirm_appointment", node_confirm_appointment)
    graph.add_node("route_nurture",       node_route_nurture)

    graph.set_entry_point("initialize")

    graph.add_edge("initialize",   "opening")
    graph.add_edge("market_value", "qualify")

    graph.add_conditional_edges(
        "opening",
        route_after_opening,
        {
            "qualify":      "qualify",
            "market_value": "market_value",
        },
    )
    graph.add_conditional_edges(
        "qualify",
        route_after_qualify,
        {
            "qualify":             "qualify",
            "close_appointment":   "close_appointment",
            "confirm_appointment": "confirm_appointment",
            "route_nurture":       "route_nurture",
        },
    )
    graph.add_conditional_edges(
        "handle_objection",
        route_after_objection,
        {
            "qualify":       "qualify",
            "route_nurture": "route_nurture",
        },
    )
    graph.add_conditional_edges(
        "close_appointment",
        route_after_close,
        {
            "confirm_appointment": "confirm_appointment",
            "handle_objection":    "handle_objection",
            "route_nurture":       "route_nurture",
        },
    )

    graph.add_edge("confirm_appointment", END)
    graph.add_edge("route_nurture",       END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_seller_graph = None


def get_seller_graph():
    global _seller_graph
    if _seller_graph is None:
        _seller_graph = build_seller_graph()
    return _seller_graph


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def run_seller_pipeline(context: dict) -> dict:
    payload   = context.get("payload", {})
    full_name = payload.get("full_name", "")
    lead_name = full_name.split()[0] if full_name.strip() else ""

    initial_state: SellerState = {
        "lead_id":           payload.get("lead_id", ""),
        "lead_name":         lead_name,
        "market":            context.get("market", "dallas_fort_worth"),
        "source":            context.get("source", "website"),
        "timeline":          context.get("timeline", "unknown"),
        "confidence":        context.get("confidence", 0),
        "flags":             context.get("flags", []),
        "messages":          [],
        "turn_count":        0,
        "lpmama":            DEFAULT_SELLER_LPMAMA.copy(),
        "is_distressed":     False,
        "equity_position":   "unknown",
        "objection_count":   0,
        "appointment_set":   False,
        "needs_nurture":     False,
        "current_node":      "",
        "system_prompt":     "",
        "last_response":     "",
        "lead_last_message": payload.get("message", "") or "",
    }

    graph  = get_seller_graph()
    result = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": 30},
    )

    logger.info(
        "SELLER COMPLETE | appointment=%s | nurture=%s | turns=%s",
        result.get("appointment_set"),
        result.get("needs_nurture"),
        result.get("turn_count"),
    )

    return {
        "status":          "completed",
        "pipeline":        "seller",
        "appointment_set": result.get("appointment_set", False),
        "needs_nurture":   result.get("needs_nurture", False),
        "turn_count":      result.get("turn_count", 0),
        "lpmama":          result.get("lpmama", {}),
        "last_response":   result.get("last_response", ""),
        "market":          result.get("market", ""),
        "context":         context,
    }
