"""
agents/nurture_graph.py
CorePilora AI — Nurture Pipeline
Handles: Long-timeline leads | Inactive markets | Not-ready leads

Serhant's law: every contact must contain something worth having.
This pipeline does not try to close an appointment today.
It delivers genuine value, earns the re-engagement anchor,
and exits in a way they remember when they ARE ready.

Flow: initialize → nurture_open → value_deliver → re_engagement → graceful_exit → END
Linear. No loops. Quality of exit over volume of turns.
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

class NurtureState(TypedDict):
    lead_id:              str
    lead_name:            str
    market:               str
    source:               str
    lead_type:            str
    timeline:             str
    confidence:           int
    flags:                list[str]
    nurture_reason:       str
    messages:             list[dict]
    turn_count:           int
    re_engagement_anchor: str
    current_node:         str
    system_prompt:        str
    last_response:        str
    lead_last_message:    str


# ─────────────────────────────────────────────────────────────────────────────
# NURTURE REASON DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def _detect_nurture_reason(context: dict) -> str:
    timeline   = context.get("timeline", "unknown")
    market     = context.get("market", "unknown")
    confidence = context.get("confidence", 0)

    if market in ("dubai", "unknown"):
        return "inactive_market"
    if timeline == "nurture" or "6" in timeline or "year" in timeline.lower():
        return "long_timeline"
    if confidence < 40:
        return "low_confidence"
    return "not_ready"


# ─────────────────────────────────────────────────────────────────────────────
# NODE — INITIALIZE
# ─────────────────────────────────────────────────────────────────────────────

async def node_initialize(state: NurtureState) -> NurtureState:
    system_prompt = build_system_prompt(
        lead_type=state.get("lead_type", "buyer"),
        market=state.get("market", "Dallas-Fort Worth"),
        lead_source=state.get("source", "website"),
        lead_name=state.get("lead_name", ""),
    )

    logger.info(
        "NURTURE | INIT | lead=%s | reason=%s | market=%s | timeline=%s",
        state.get("lead_name"),
        state.get("nurture_reason"),
        state.get("market"),
        state.get("timeline"),
    )

    return {
        **state,
        "system_prompt":        system_prompt,
        "current_node":         "initialize",
        "turn_count":           0,
        "re_engagement_anchor": "",
        "last_response":        "",
        "lead_last_message":    state.get("lead_last_message", ""),
        "messages":             state.get("messages", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — NURTURE OPEN
# No pressure. Acknowledge timing directly. Earn the right to deliver value.
# Serhant: "Following up is service they didn't know they needed."
# ─────────────────────────────────────────────────────────────────────────────

async def node_nurture_open(state: NurtureState) -> NurtureState:
    name              = state.get("lead_name", "")
    market            = state.get("market", "Dallas-Fort Worth")
    lead_type         = state.get("lead_type", "buyer")
    nurture_reason    = state.get("nurture_reason", "not_ready")
    lead_last_message = state.get("lead_last_message", "")
    messages          = state.get("messages", [])

    lead_context = (
        f'Their message: "{lead_last_message}". Acknowledge what they revealed.'
        if lead_last_message else ""
    )

    reason_instructions = {
        "long_timeline": (
            f"This {lead_type} has a long timeline — they are not moving soon. "
            f"Acknowledge timing directly without embarrassment. "
            f"'I completely understand — timing is the whole game in real estate.' "
            f"Make them feel heard, not pushed. "
            f"Tell them you want to give them one thing worth having before you go."
        ),
        "inactive_market": (
            f"This lead is interested in a market outside primary coverage. "
            f"Acknowledge with honesty — you will serve them best when timing aligns. "
            f"Deliver value about {market} market conditions they can use to plan."
        ),
        "low_confidence": (
            f"This lead's intent is not fully clear yet — they may still be exploring. "
            f"No pressure. Open by giving them permission to be wherever they are. "
            f"'You reached out — which means something is on your mind even if it is not fully formed yet.' "
            f"Make them comfortable sharing without feeling committed."
        ),
        "not_ready": (
            f"This {lead_type} is not ready to move right now. "
            f"Acknowledge their position with zero pressure. "
            f"'Timing is everything in real estate — and there is no wrong answer today.' "
            f"Earn the right to stay in contact."
        ),
    }

    instruction = (
        f"Make first contact with {name or 'this lead'}. {lead_context} "
        f"NURTURE OPENING — zero pressure, maximum value: "
        f"{reason_instructions.get(nurture_reason, reason_instructions['not_ready'])} "
        f"Under 3 sentences. Warm. Human. No desperation. End with one open question."
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.75,
            max_tokens=120,
        )
    except Exception as e:
        logger.error("NURTURE | OPEN FAILED | error=%s", str(e))
        response = (
            f"Hey {name} — Jaiyana here. "
            f"No pressure at all — I just wanted to give you something useful "
            f"about {market} before you go. What is the most important thing "
            f"you want to figure out before making any move?"
        )

    messages = messages + [{"role": "assistant", "content": response}]

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "current_node":  "nurture_open",
        "turn_count":    state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — VALUE DELIVER
# Specific market intelligence calibrated to their lead type and market.
# Ullrich: exact numbers. Not vague market commentary.
# ─────────────────────────────────────────────────────────────────────────────

async def node_value_deliver(state: NurtureState) -> NurtureState:
    market    = state.get("market", "Dallas-Fort Worth")
    lead_type = state.get("lead_type", "buyer")
    timeline  = state.get("timeline", "unknown")
    messages  = state.get("messages", [])

    type_value = {
        "buyer": (
            f"Deliver buyer-specific market intelligence for {market}. "
            f"Use exact data points — median price movement, days on market, inventory levels. "
            f"Frame around: what does waiting cost vs. moving now? "
            f"Reference: 'The buyers who are positioned well in 6 months started the conversation today.' "
            f"Not pushy — educational. Give them data that makes them look smart."
        ),
        "seller": (
            f"Deliver seller-specific intelligence for {market}. "
            f"Use exact data — current absorption rate, average days on market, price per square foot trend. "
            f"Frame around: what does the market look like for sellers right now, and what would change it? "
            f"Reference: 'The sellers who maximized proceeds this year listed in [window].' "
            f"Position as someone who protects their net proceeds, not just closes deals."
        ),
        "investor": (
            f"Deliver investor-specific market data for {market}. "
            f"Cap rates, rental yield trends, vacancy rates, appreciation over 12 months. "
            f"Frame around: what is the opportunity cost of waiting on this market? "
            f"Reference specific submarkets or corridors with the best current risk-adjusted returns. "
            f"Numbers only. This lead thinks in ROI."
        ),
    }

    instruction = (
        f"This {lead_type} lead in {market} has a timeline of '{timeline}'. "
        f"DELIVER GENUINE VALUE — something specific they can use: "
        f"{type_value.get(lead_type, type_value['buyer'])} "
        f"Under 4 sentences. Exact numbers. No fluff. No ask. "
        f"This is the thing worth having — make it worth having."
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.65,
            max_tokens=200,
        )
    except Exception as e:
        logger.error("NURTURE | VALUE DELIVER FAILED | error=%s", str(e))
        market_intel = {
            "buyer":    f"In {market} right now, median days on market is hovering around 21 days — down from 35 last year. The window for choice is narrowing as inventory tightens.",
            "seller":   f"In {market}, sellers who listed in Q1 averaged 98.4% of asking price. That number typically softens 3-4% by Q3. The data favors moving sooner.",
            "investor": f"Core {market} submarkets are running 5.2-7.1% cap rates right now. Historically, those windows compress when the next wave of corporate relocation hits.",
        }
        response = market_intel.get(lead_type, market_intel["buyer"])

    messages = messages + [{"role": "assistant", "content": response}]

    logger.info(
        "NURTURE | VALUE DELIVERED | lead=%s | type=%s | market=%s",
        state.get("lead_name"),
        lead_type,
        market,
    )

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "current_node":  "value_deliver",
        "turn_count":    state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — RE-ENGAGEMENT ANCHOR
# Get a soft commitment on timing. Not a hard close — a real question.
# "When does this become real for you? Give me a month."
# That answer becomes the follow-up trigger.
# ─────────────────────────────────────────────────────────────────────────────

async def node_re_engagement(state: NurtureState) -> NurtureState:
    name      = state.get("lead_name", "")
    market    = state.get("market", "Dallas-Fort Worth")
    lead_type = state.get("lead_type", "buyer")
    messages  = state.get("messages", [])

    type_anchor = {
        "buyer": (
            f"Ask one soft commitment question: when does buying in {market} become real for them — "
            f"give me a month, not a maybe. "
            f"Frame as: 'I want to make sure I reach back when it actually matters for you — "
            f"not just to check in. When do you think this becomes serious?'"
        ),
        "seller": (
            f"Ask one soft commitment question: is there a month this year where selling "
            f"becomes the right move? "
            f"Frame as: 'I want to be tracking the market for you specifically — "
            f"what month should I flag as your window?'"
        ),
        "investor": (
            f"Ask one direct question: what does their acquisition timeline look like — "
            f"Q3, Q4, next year? "
            f"Frame as: 'I will keep your criteria on my radar and reach out when the "
            f"right deal comes through — when are you looking to move?'"
        ),
    }

    instruction = (
        f"RE-ENGAGEMENT ANCHOR for {name or 'this lead'} in {market}. "
        f"{type_anchor.get(lead_type, type_anchor['buyer'])} "
        f"Under 2 sentences. No pressure. Direct question. "
        f"This is not a close — it is a calendar question."
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.7,
            max_tokens=100,
        )
    except Exception as e:
        logger.error("NURTURE | RE-ENGAGEMENT FAILED | error=%s", str(e))
        response = (
            f"I want to reach back when it actually matters for you — "
            f"not just to check in. When do you think this becomes serious for you, "
            f"{name}? Give me a rough month."
        )

    messages = messages + [{"role": "assistant", "content": response}]

    return {
        **state,
        "messages":             messages,
        "last_response":        response,
        "re_engagement_anchor": response,
        "current_node":         "re_engagement",
        "turn_count":           state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — GRACEFUL EXIT
# Serhant exit: leave something. Zero desperation. High conviction.
# They should walk away thinking: that was the most useful 3 minutes I spent today.
# ─────────────────────────────────────────────────────────────────────────────

async def node_graceful_exit(state: NurtureState) -> NurtureState:
    name       = state.get("lead_name", "")
    market     = state.get("market", "Dallas-Fort Worth")
    lead_type  = state.get("lead_type", "buyer")
    messages   = state.get("messages", [])

    type_exit = {
        "buyer": (
            f"Exit warm. Tell them you will stay on top of {market} on their behalf. "
            f"Reference: 'When the market moves in a way that affects what you are looking at, "
            f"you will hear from me before anyone else does.' "
            f"Zero desperation. High conviction that your follow-up is service."
        ),
        "seller": (
            f"Exit with a forward commitment specific to {market} seller conditions. "
            f"'When something shifts that changes the net proceeds picture for your property, "
            f"I will reach out with the actual number — not a generic update.' "
            f"Make the future contact feel like something worth receiving."
        ),
        "investor": (
            f"Exit efficiently. Tell them you will send first-look deals matching their criteria "
            f"before they hit the market. "
            f"No fluff. That one sentence is worth more than three paragraphs of rapport-building."
        ),
    }

    instruction = (
        f"GRACEFUL EXIT for {name or 'this lead'}. "
        f"{type_exit.get(lead_type, type_exit['buyer'])} "
        f"Under 2 sentences. The exit should feel like a beginning, not an ending. "
        f"Jaiyana's exits are memorable — make this one count."
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.7,
            max_tokens=100,
        )
    except Exception as e:
        logger.error("NURTURE | EXIT FAILED | error=%s", str(e))
        response = (
            f"No pressure at all {name}. "
            f"I will keep an eye on {market} for you — "
            f"when something moves that matters for your situation, "
            f"you will hear from me directly."
        )

    messages = messages + [{"role": "assistant", "content": response}]

    logger.info(
        "NURTURE | EXIT | lead=%s | reason=%s | market=%s | turns=%s",
        state.get("lead_name"),
        state.get("nurture_reason"),
        market,
        state.get("turn_count"),
    )

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "current_node":  "graceful_exit",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY — Linear. No loops. Quality > volume.
# ─────────────────────────────────────────────────────────────────────────────

def build_nurture_graph() -> StateGraph:
    graph = StateGraph(NurtureState)

    graph.add_node("initialize",   node_initialize)
    graph.add_node("nurture_open", node_nurture_open)
    graph.add_node("value_deliver", node_value_deliver)
    graph.add_node("re_engagement", node_re_engagement)
    graph.add_node("graceful_exit", node_graceful_exit)

    graph.set_entry_point("initialize")

    graph.add_edge("initialize",    "nurture_open")
    graph.add_edge("nurture_open",  "value_deliver")
    graph.add_edge("value_deliver", "re_engagement")
    graph.add_edge("re_engagement", "graceful_exit")
    graph.add_edge("graceful_exit", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_nurture_graph = None


def get_nurture_graph():
    global _nurture_graph
    if _nurture_graph is None:
        _nurture_graph = build_nurture_graph()
    return _nurture_graph


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT — signature locked, called by lead_type_router
# ─────────────────────────────────────────────────────────────────────────────

async def run_nurture_pipeline(context: dict) -> dict:
    payload   = context.get("payload", {})
    full_name = payload.get("full_name", "")
    lead_name = full_name.split()[0] if full_name.strip() else ""

    nurture_reason = _detect_nurture_reason(context)

    initial_state: NurtureState = {
        "lead_id":              payload.get("lead_id", ""),
        "lead_name":            lead_name,
        "market":               context.get("market", "dallas_fort_worth"),
        "source":               context.get("source", "website"),
        "lead_type":            context.get("lead_type", "buyer"),
        "timeline":             context.get("timeline", "unknown"),
        "confidence":           context.get("confidence", 0),
        "flags":                context.get("flags", []),
        "nurture_reason":       nurture_reason,
        "messages":             [],
        "turn_count":           0,
        "re_engagement_anchor": "",
        "current_node":         "",
        "system_prompt":        "",
        "last_response":        "",
        "lead_last_message":    payload.get("message", "") or "",
    }

    graph  = get_nurture_graph()
    result = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": 20},
    )

    logger.info(
        "NURTURE COMPLETE | lead=%s | reason=%s | turns=%s",
        result.get("lead_name"),
        result.get("nurture_reason"),
        result.get("turn_count"),
    )

    return {
        "status":               "completed",
        "pipeline":             "nurture",
        "nurture_reason":       result.get("nurture_reason", ""),
        "re_engagement_anchor": result.get("re_engagement_anchor", ""),
        "appointment_set":      False,
        "needs_nurture":        True,
        "turn_count":           result.get("turn_count", 0),
        "last_response":        result.get("last_response", ""),
        "market":               result.get("market", ""),
        "lead_type":            result.get("lead_type", ""),
        "context":              context,
    }
