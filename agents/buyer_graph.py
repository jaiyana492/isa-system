"""
agents/buyer_graph.py
CorePilora AI — Buyer Closing Pipeline

Full LangGraph implementation.
Groq LLM via shared services/groq_client.py — model driven by settings.GROQ_MODEL.
LPMAMA enforced at graph level.
Appointment detection reads LEAD input, not Jaiyana output.
LPMAMA extraction runs every 3 turns — not every turn.
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.persona import build_system_prompt, DEFAULT_LPMAMA_STATE
from agents.utils import lead_confirmed_appointment
from services.groq_client import call_groq

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class BuyerState(TypedDict):
    lead_id:           str
    lead_name:         str
    market:            str
    source:            str
    finance_type:      str
    timeline:          str
    confidence:        int
    flags:             list[str]
    messages:          list[dict]
    turn_count:        int
    lpmama:            dict
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

async def _extract_lpmama(
    conversation:   list[dict],
    current_lpmama: dict,
    turn_count:     int,
) -> dict:
    if turn_count % 3 != 0:
        return current_lpmama

    extraction_prompt = """
You are a data extraction engine for a real estate ISA system.
Read the conversation and extract confirmed buyer qualification data.

Return ONLY valid JSON with these exact keys:
{
  "location": "confirmed area or null",
  "price": "confirmed budget range or null",
  "motivation": "confirmed reason for moving or null",
  "agent": "confirmed yes/no working with agent or null",
  "mortgage": "confirmed pre-approved/cash/needs lender or null",
  "appointment": "confirmed appointment time or null"
}

Only include values the lead explicitly stated.
Null if not confirmed. JSON only. No explanation. No markdown.
"""
    try:
        raw   = await call_groq(
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
        logger.warning("BUYER | LPMAMA extraction failed | error=%s", str(e))
        return current_lpmama


# ─────────────────────────────────────────────────────────────────────────────
# NODE — INITIALIZE
# ─────────────────────────────────────────────────────────────────────────────

async def node_initialize(state: BuyerState) -> BuyerState:
    system_prompt = build_system_prompt(
        lead_type="buyer",
        market=state.get("market", "Dallas-Fort Worth"),
        lead_source=state.get("source", "website"),
        lead_name=state.get("lead_name", ""),
        lpmama_state=state.get("lpmama", DEFAULT_LPMAMA_STATE.copy()),
    )

    logger.info(
        "BUYER | INIT | lead=%s | market=%s | timeline=%s",
        state.get("lead_name"),
        state.get("market"),
        state.get("timeline"),
    )

    return {
        **state,
        "system_prompt":     system_prompt,
        "current_node":      "initialize",
        "turn_count":        0,
        "objection_count":   0,
        "appointment_set":   False,
        "needs_nurture":     False,
        "last_response":     "",
        "lead_last_message": "",
        "messages":          state.get("messages", []),
        "lpmama":            state.get("lpmama", DEFAULT_LPMAMA_STATE.copy()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — OPENING
# ─────────────────────────────────────────────────────────────────────────────

async def node_opening(state: BuyerState) -> BuyerState:
    source            = state.get("source", "website")
    name              = state.get("lead_name", "")
    market            = state.get("market", "Dallas-Fort Worth")
    messages          = state.get("messages", [])
    lead_last_message = state.get("lead_last_message", "")

    lead_context = (
        f'Their message when they reached out: "{lead_last_message}". Speak to it.'
        if lead_last_message else ""
    )
    instruction = (
        f"Make first contact with {name or 'this lead'}. "
        f"Source: {source}. Market: {market}. {lead_context} "
        f"Deliver your opening line only. Under 2 sentences. "
        f"Pattern interrupt. Create curiosity. Do not qualify. Do not explain. End with one question."
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
        logger.error("BUYER | OPENING FAILED | error=%s", str(e))
        response = (
            f"Hey {name} — Jaiyana here. "
            f"You reached out about {market} real estate — "
            f"what made you reach out today?"
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
# NODE — QUALIFY
# ─────────────────────────────────────────────────────────────────────────────

async def node_qualify(state: BuyerState) -> BuyerState:
    lpmama            = state.get("lpmama", {})
    missing           = [k for k, v in lpmama.items() if v is None]
    messages          = state.get("messages", [])
    lead_last_message = state.get("lead_last_message", "")

    if lead_confirmed_appointment(lead_last_message):
        logger.info("BUYER | APPOINTMENT CONFIRMED BY LEAD")
        return {
            **state,
            "appointment_set": True,
            "current_node":    "qualify",
            "turn_count":      state.get("turn_count", 0) + 1,
        }

    lead_said = f'The lead just said: "{lead_last_message}"' if lead_last_message else ""
    next_gate = missing[0].upper() if missing else "APPOINTMENT CLOSE"
    instruction = (
        f"{lead_said} "
        f"TURN ARCHITECTURE — EXECUTE THIS EXACTLY: "
        f"(1) ACKNOWLEDGE first — mirror their last 2-3 words as a soft question, OR label the emotion underneath ('It sounds like...', 'It seems like...'). Never skip this. "
        f"(2) DEEPEN if they revealed a life trigger, fear, or strong emotion — one follow-up on that before advancing. "
        f"(3) ONE focused How/What question advancing toward: {next_gate}. "
        f"STOP after the question. 2 sentences max. Never 3."
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
        logger.error("BUYER | QUALIFY FAILED | error=%s", str(e))
        response = "What is driving the move right now — what changed that made this the moment?"

    messages = messages + [{"role": "assistant", "content": response}]

    turn_count     = state.get("turn_count", 0)
    updated_lpmama = await _extract_lpmama(
        conversation=messages,
        current_lpmama=lpmama,
        turn_count=turn_count,
    )

    logger.info(
        "BUYER | QUALIFY | turn=%s | collected=%s | missing=%s",
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

async def node_handle_objection(state: BuyerState) -> BuyerState:
    count             = state.get("objection_count", 0)
    market            = state.get("market", "Dallas-Fort Worth")
    messages          = state.get("messages", [])
    lead_last_message = state.get("lead_last_message", "")

    lead_said = f'Their exact pushback: "{lead_last_message}"' if lead_last_message else "Lead pushed back."
    instruction = (
        f"{lead_said} Objection #{count + 1}. Market: {market}. "
        f"TURN ARCHITECTURE: "
        f"(1) LABEL the real emotion underneath this objection first — 'It sounds like...' or 'It seems like...'. Name what is actually driving the resistance. "
        f"(2) Redirect using a completely different angle than any previous response — Voss reframe, Serhant FOMO, Caballero efficiency, or Ullrich data. "
        f"(3) End with one calibrated How/What question. "
        f"Under 3 sentences. Zero weak language."
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
        logger.error("BUYER | OBJECTION FAILED | error=%s", str(e))
        response = "I hear you. What would need to change for this to make sense for you right now?"

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
# Offers time options. Does NOT set appointment_set.
# Lead must confirm — detected in node_qualify via lead_last_message.
# ─────────────────────────────────────────────────────────────────────────────

async def node_close_appointment(state: BuyerState) -> BuyerState:
    name     = state.get("lead_name", "")
    market   = state.get("market", "Dallas-Fort Worth")
    lpmama   = state.get("lpmama", {})
    messages = state.get("messages", [])

    instruction = (
        f"Qualification complete for {name or 'this lead'} in {market}. "
        f"Data collected: {json.dumps(lpmama)}. "
        f"Offer the appointment now. Give two specific time options this week — morning and afternoon. "
        f"Frame as a real estate strategy session. Direct. Confident. No fluff. "
        f"Do NOT confirm the appointment yourself. Wait for their response."
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
        logger.error("BUYER | CLOSE FAILED | error=%s", str(e))
        response = (
            f"Here is what I want to do {name} — "
            f"I have Tuesday at 10am or Thursday at 2pm this week. "
            f"Which one works better for you?"
        )

    messages = messages + [{"role": "assistant", "content": response}]
    logger.info("BUYER | CLOSE OFFERED | lead=%s", state.get("lead_name"))

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

async def node_confirm_appointment(state: BuyerState) -> BuyerState:
    name     = state.get("lead_name", "")
    messages = state.get("messages", [])

    instruction = (
        f"{name or 'The lead'} just confirmed the appointment. "
        f"Send a brief warm confirmation. Confirm the time. State what to expect. "
        f"Under 3 sentences. Professional. Human."
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
        logger.error("BUYER | CONFIRM FAILED | error=%s", str(e))
        response = f"You are locked in {name}. I will send a confirmation shortly. Looking forward to it."

    messages = messages + [{"role": "assistant", "content": response}]
    logger.info("BUYER | CONFIRMED | lead=%s", state.get("lead_name"))

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

async def node_route_nurture(state: BuyerState) -> BuyerState:
    name     = state.get("lead_name", "")
    messages = state.get("messages", [])

    instruction = (
        f"{name or 'This lead'} is not ready right now. "
        f"Exit the conversation gracefully. Leave the door open. "
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
        logger.error("BUYER | NURTURE EXIT FAILED | error=%s", str(e))
        response = f"No pressure at all {name}. I will follow up with something useful for you. Talk soon."

    messages = messages + [{"role": "assistant", "content": response}]
    logger.info(
        "BUYER | NURTURE | lead=%s | objections=%s",
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

def route_after_qualify(state: BuyerState) -> str:
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


def route_after_objection(state: BuyerState) -> str:
    if state.get("objection_count", 0) >= 5:
        return "route_nurture"
    return "qualify"


def route_after_close(state: BuyerState) -> str:
    if state.get("appointment_set"):
        return "confirm_appointment"
    if state.get("objection_count", 0) >= 3:
        return "route_nurture"
    return "handle_objection"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_buyer_graph() -> StateGraph:
    graph = StateGraph(BuyerState)

    graph.add_node("initialize",          node_initialize)
    graph.add_node("opening",             node_opening)
    graph.add_node("qualify",             node_qualify)
    graph.add_node("handle_objection",    node_handle_objection)
    graph.add_node("close_appointment",   node_close_appointment)
    graph.add_node("confirm_appointment", node_confirm_appointment)
    graph.add_node("route_nurture",       node_route_nurture)

    graph.set_entry_point("initialize")

    graph.add_edge("initialize", "opening")
    graph.add_edge("opening",    "qualify")

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

_buyer_graph = None


def get_buyer_graph():
    global _buyer_graph
    if _buyer_graph is None:
        _buyer_graph = build_buyer_graph()
    return _buyer_graph


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def run_buyer_pipeline(context: dict) -> dict:
    payload   = context.get("payload", {})
    full_name = payload.get("full_name", "")
    # Use first word of full_name for personalization, fallback to full
    lead_name = full_name.split()[0] if full_name.strip() else ""

    initial_state: BuyerState = {
        "lead_id":           payload.get("lead_id", ""),
        "lead_name":         lead_name,
        "market":            context.get("market", "dallas_fort_worth"),
        "source":            context.get("source", "website"),
        "finance_type":      context.get("finance_type", "unknown"),
        "timeline":          context.get("timeline", "unknown"),
        "confidence":        context.get("confidence", 0),
        "flags":             context.get("flags", []),
        "messages":          [],
        "turn_count":        0,
        "lpmama":            DEFAULT_LPMAMA_STATE.copy(),
        "objection_count":   0,
        "appointment_set":   False,
        "needs_nurture":     False,
        "current_node":      "",
        "system_prompt":     "",
        "last_response":     "",
        "lead_last_message": payload.get("message", "") or "",
    }

    graph  = get_buyer_graph()
    result = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": 30},
    )

    logger.info(
        "BUYER COMPLETE | appointment=%s | nurture=%s | turns=%s",
        result.get("appointment_set"),
        result.get("needs_nurture"),
        result.get("turn_count"),
    )

    return {
        "status":          "completed",
        "pipeline":        "buyer",
        "appointment_set": result.get("appointment_set", False),
        "needs_nurture":   result.get("needs_nurture", False),
        "turn_count":      result.get("turn_count", 0),
        "lpmama":          result.get("lpmama", {}),
        "last_response":   result.get("last_response", ""),
        "market":          result.get("market", ""),
        "context":         context,
    }
