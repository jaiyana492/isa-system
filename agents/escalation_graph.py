"""
agents/escalation_graph.py
CorePilora AI — Escalation Pipeline
Handles: DISTRESSED_SELLER | HIGH_MOTIVATION | HOT_INVESTOR_CLOSE

Fast-track qualification. Urgency-first open. Hard close within 8 turns.
These leads have already decided — they need certainty and speed, not education.
Standard 20-turn qualification loop is replaced with a compressed 8-turn sprint.
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

DEFAULT_ESCALATION_LPMAMA = {
    "location":     None,
    "budget":       None,
    "motivation":   None,
    "urgency":      None,   # specific deadline or timeframe — critical for distressed
    "money_source": None,
    "appointment":  None,
}

ESCALATION_TURN_LIMIT = 8   # vs 20 for standard — hot leads close fast or not at all


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class EscalationState(TypedDict):
    lead_id:           str
    lead_name:         str
    market:            str
    source:            str
    lead_type:         str
    timeline:          str
    confidence:        int
    flags:             list[str]
    escalation_type:   str
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
# ESCALATION TYPE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def _detect_escalation_type(flags: list[str]) -> str:
    if "DISTRESSED_SELLER" in flags:
        return "distressed_seller"
    if "HOT_INVESTOR_CLOSE" in flags:
        return "hot_investor"
    return "high_motivation"


# ─────────────────────────────────────────────────────────────────────────────
# LPMAMA EXTRACTOR — every 2 turns (vs 3 for standard — speed matters here)
# ─────────────────────────────────────────────────────────────────────────────

async def _extract_escalation_lpmama(
    conversation:   list[dict],
    current_lpmama: dict,
    turn_count:     int,
) -> dict:
    if turn_count % 2 != 0:
        return current_lpmama

    extraction_prompt = """
You are a data extraction engine for a real estate ISA system.
Read the conversation and extract confirmed escalation lead data.

Return ONLY valid JSON with these exact keys:
{
  "location": "confirmed property address or target area or null",
  "budget": "confirmed budget or price expectation or null",
  "motivation": "confirmed urgent reason for moving or selling or null",
  "urgency": "confirmed specific deadline or timeframe or null",
  "money_source": "confirmed cash/pre-approved/mortgage balance or null",
  "appointment": "confirmed appointment time or null"
}

Only include values the lead explicitly stated.
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
        logger.warning("ESCALATION | LPMAMA extraction failed | error=%s", str(e))
        return current_lpmama


# ─────────────────────────────────────────────────────────────────────────────
# NODE — INITIALIZE
# ─────────────────────────────────────────────────────────────────────────────

async def node_initialize(state: EscalationState) -> EscalationState:
    flags           = state.get("flags", [])
    escalation_type = _detect_escalation_type(flags)

    system_prompt = build_system_prompt(
        lead_type=state.get("lead_type", "buyer"),
        market=state.get("market", "Dallas-Fort Worth"),
        lead_source=state.get("source", "website"),
        lead_name=state.get("lead_name", ""),
        lpmama_state=state.get("lpmama", DEFAULT_ESCALATION_LPMAMA.copy()),
    )

    logger.info(
        "ESCALATION | INIT | lead=%s | type=%s | flags=%s | market=%s",
        state.get("lead_name"),
        escalation_type,
        flags,
        state.get("market"),
    )

    return {
        **state,
        "system_prompt":     system_prompt,
        "current_node":      "initialize",
        "escalation_type":   escalation_type,
        "turn_count":        0,
        "objection_count":   0,
        "appointment_set":   False,
        "needs_nurture":     False,
        "last_response":     "",
        "lead_last_message": "",
        "messages":          state.get("messages", []),
        "lpmama":            state.get("lpmama", DEFAULT_ESCALATION_LPMAMA.copy()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — URGENCY OPEN
# No slow build. Label the situation. Establish authority in one sentence.
# ─────────────────────────────────────────────────────────────────────────────

async def node_urgency_open(state: EscalationState) -> EscalationState:
    name              = state.get("lead_name", "")
    market            = state.get("market", "Dallas-Fort Worth")
    escalation_type   = state.get("escalation_type", "high_motivation")
    lead_last_message = state.get("lead_last_message", "")
    lead_type         = state.get("lead_type", "buyer")
    messages          = state.get("messages", [])

    lead_context = (
        f'Their message: "{lead_last_message}". Respond directly to what they revealed.'
        if lead_last_message else ""
    )

    type_instructions = {
        "distressed_seller": (
            f"DISTRESSED SELLER — label the weight of their situation with precision. "
            f"Sound: 'It sounds like you are under real time pressure and you need someone who can actually move fast.' "
            f"Do NOT downplay it. Acknowledge the urgency. Establish certainty. "
            f"Voss tone: Late night FM DJ calm — you have done this before. "
            f"ONE question: what is the timeline you are working with?"
        ),
        "hot_investor": (
            f"HOT INVESTOR ready to close — skip warm-up entirely. "
            f"Open with a specific {market} data point about deal flow or cap rates. "
            f"Position as their unfair advantage — you have the deal access they need. "
            f"ONE question: what is the price point per door and how are they capitalizing?"
        ),
        "high_motivation": (
            f"HIGH MOTIVATION {lead_type} — acknowledge the momentum they already have. "
            f"'It sounds like you are at the point where this actually needs to happen — not someday, now.' "
            f"ONE focused question: what is the specific thing that made this the moment?"
        ),
    }

    instruction = (
        f"Make first contact with {name or 'this lead'}. {lead_context} "
        f"ESCALATION PROTOCOL — THIS IS A HOT LEAD: "
        f"{type_instructions.get(escalation_type, type_instructions['high_motivation'])} "
        f"Under 2 sentences. Zero wasted words."
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
        logger.error("ESCALATION | URGENCY OPEN FAILED | error=%s", str(e))
        response = (
            f"Hey {name} — Jaiyana. "
            f"It sounds like you need to move on this fast — "
            f"what is the actual timeline you are working with?"
        )

    messages = messages + [{"role": "assistant", "content": response}]

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "current_node":  "urgency_open",
        "turn_count":    state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — FAST QUALIFY
# Compressed LPMAMA. Priority: urgency → location → money_source → budget.
# 8-turn limit instead of 20.
# ─────────────────────────────────────────────────────────────────────────────

async def node_fast_qualify(state: EscalationState) -> EscalationState:
    lpmama            = state.get("lpmama", {})
    missing           = [k for k, v in lpmama.items() if v is None]
    messages          = state.get("messages", [])
    lead_last_message = state.get("lead_last_message", "")
    escalation_type   = state.get("escalation_type", "high_motivation")
    market            = state.get("market", "Dallas-Fort Worth")

    if lead_confirmed_appointment(lead_last_message):
        logger.info("ESCALATION | APPOINTMENT CONFIRMED BY LEAD")
        return {
            **state,
            "appointment_set": True,
            "current_node":    "fast_qualify",
            "turn_count":      state.get("turn_count", 0) + 1,
        }

    lead_said = f'They just said: "{lead_last_message}"' if lead_last_message else ""

    # Priority gate order for escalation — urgency is the master key
    priority_order = ["urgency", "location", "money_source", "budget", "motivation"]
    next_gate = next(
        (gate for gate in priority_order if lpmama.get(gate) is None),
        "APPOINTMENT CLOSE",
    )

    type_angle = {
        "distressed_seller": (
            "Reference the urgency clock — the sooner you understand their position, "
            "the faster you can build their path forward. Every turn matters."
        ),
        "hot_investor": (
            "Numbers only. No lifestyle language. Direct qualification. "
            "Respect their time — they have capital, they need deal access."
        ),
        "high_motivation": (
            "Connect each question to the outcome they are reaching for. "
            "Their momentum is real — meet it with equal directness."
        ),
    }

    instruction = (
        f"{lead_said} "
        f"FAST QUALIFICATION — ESCALATION PROTOCOL: "
        f"(1) ACKNOWLEDGE — one sentence, mirror their last 2-3 words or label the emotion underneath. Zero filler. "
        f"(2) ADVANCE — one precise How/What question targeting: {next_gate.upper()}. "
        f"{type_angle.get(escalation_type, type_angle['high_motivation'])} "
        f"STOP. 2 sentences max. Market: {market}."
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
        logger.error("ESCALATION | FAST QUALIFY FAILED | error=%s", str(e))
        response = "What is the actual timeline you are working with on this?"

    messages = messages + [{"role": "assistant", "content": response}]

    turn_count     = state.get("turn_count", 0)
    updated_lpmama = await _extract_escalation_lpmama(
        conversation=messages,
        current_lpmama=lpmama,
        turn_count=turn_count,
    )

    logger.info(
        "ESCALATION | QUALIFY | turn=%s | collected=%s | missing=%s",
        turn_count,
        [k for k, v in updated_lpmama.items() if v],
        [k for k, v in updated_lpmama.items() if not v],
    )

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "lpmama":        updated_lpmama,
        "current_node":  "fast_qualify",
        "turn_count":    turn_count + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — URGENCY CLOSE
# Hard close. Leverage their specific urgency driver. Create real time pressure.
# ─────────────────────────────────────────────────────────────────────────────

async def node_urgency_close(state: EscalationState) -> EscalationState:
    name            = state.get("lead_name", "")
    market          = state.get("market", "Dallas-Fort Worth")
    lpmama          = state.get("lpmama", {})
    escalation_type = state.get("escalation_type", "high_motivation")
    messages        = state.get("messages", [])

    urgency    = lpmama.get("urgency", "")
    motivation = lpmama.get("motivation", "")

    type_framing = {
        "distressed_seller": (
            f"Urgency driver: {urgency or 'time-sensitive situation'}. "
            f"Frame the appointment as the fastest path to a solution — "
            f"property walkthrough + net proceeds math + realistic timeline in one session. "
            f"This lead needs certainty. Give it to them in the close."
        ),
        "hot_investor": (
            f"This investor is ready to move. Frame as a deal pipeline review — "
            f"you will bring specific {market} inventory at their price point, "
            f"run the numbers live, and map the acquisition path. "
            f"Investors respect efficiency — make the meeting feel like it pays for itself."
        ),
        "high_motivation": (
            f"Motivation: {motivation or 'strong personal driver'}. "
            f"Frame the appointment as getting their plan in place before the market moves on them. "
            f"Reference: 'The window you are looking at right now is the same one the last three "
            f"people I worked with wish they had taken.'"
        ),
    }

    instruction = (
        f"Qualification complete for {name or 'this lead'} in {market}. "
        f"Data collected: {json.dumps(lpmama)}. "
        f"{type_framing.get(escalation_type, type_framing['high_motivation'])} "
        f"URGENCY CLOSE: "
        f"Two specific time options — aim for one tomorrow or day after, one later this week. "
        f"Frame as time-sensitive because it is. Direct. Confident. "
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
        logger.error("ESCALATION | URGENCY CLOSE FAILED | error=%s", str(e))
        response = (
            f"Here is what needs to happen {name} — "
            f"I want to sit down with you this week, run the full picture, "
            f"and build a real plan. I have tomorrow at 10am or Thursday at 2pm. "
            f"Which one works?"
        )

    messages = messages + [{"role": "assistant", "content": response}]
    logger.info(
        "ESCALATION | CLOSE OFFERED | lead=%s | type=%s",
        state.get("lead_name"),
        escalation_type,
    )

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "current_node":  "urgency_close",
        "turn_count":    state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — CONFIRM APPOINTMENT
# ─────────────────────────────────────────────────────────────────────────────

async def node_confirm_appointment(state: EscalationState) -> EscalationState:
    name            = state.get("lead_name", "")
    escalation_type = state.get("escalation_type", "high_motivation")
    messages        = state.get("messages", [])

    type_confirm = {
        "distressed_seller": (
            "Confirm and specifically state what you will bring — "
            "net proceeds analysis, realistic timeline, clear next steps. "
            "They need certainty above all. Under 2 sentences."
        ),
        "hot_investor": (
            "Confirm tight. State you will bring deal pipeline data and market numbers ready to run. "
            "Under 2 sentences. Direct. Professional."
        ),
        "high_motivation": (
            "Warm confirmation. Reference their specific goal. "
            "Under 2 sentences. Human and direct."
        ),
    }

    instruction = (
        f"{name or 'The lead'} just confirmed. "
        f"{type_confirm.get(escalation_type, type_confirm['high_motivation'])}"
    )
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.5,
            max_tokens=100,
        )
    except Exception as e:
        logger.error("ESCALATION | CONFIRM FAILED | error=%s", str(e))
        response = f"Locked in {name}. I will have everything ready. See you then."

    messages = messages + [{"role": "assistant", "content": response}]
    logger.info(
        "ESCALATION | CONFIRMED | lead=%s | type=%s",
        state.get("lead_name"),
        escalation_type,
    )

    return {
        **state,
        "messages":        messages,
        "last_response":   response,
        "appointment_set": True,
        "current_node":    "confirm_appointment",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — GRACEFUL EXIT
# When urgency leads still resist after 2 objections. Preserve relationship.
# Zero desperation. Leave a specific forward commitment.
# ─────────────────────────────────────────────────────────────────────────────

async def node_graceful_exit(state: EscalationState) -> EscalationState:
    name            = state.get("lead_name", "")
    market          = state.get("market", "Dallas-Fort Worth")
    escalation_type = state.get("escalation_type", "high_motivation")
    messages        = state.get("messages", [])

    type_exit = {
        "distressed_seller": (
            f"Exit with a specific forward commitment — you will reach out the moment "
            f"something relevant to their situation moves in {market}. "
            f"Make clear the clock is real — but zero pressure. "
            f"Under 2 sentences."
        ),
        "hot_investor": (
            f"Exit efficiently. Tell them you will have first look at new {market} deals "
            f"and reach out directly when the right opportunity hits their criteria. "
            f"Under 2 sentences. Direct."
        ),
        "high_motivation": (
            f"Exit warm. Reference their specific motivation — they will remember you connected to it. "
            f"Leave the door wide open. Under 2 sentences."
        ),
    }

    instruction = (
        f"Exit gracefully for {name or 'this lead'}. "
        f"{type_exit.get(escalation_type, type_exit['high_motivation'])}"
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
        logger.error("ESCALATION | EXIT FAILED | error=%s", str(e))
        response = (
            f"No pressure at all {name}. "
            f"I will be watching {market} closely and reach out the moment "
            f"something moves that matters for your situation."
        )

    messages = messages + [{"role": "assistant", "content": response}]
    logger.info(
        "ESCALATION | EXIT | lead=%s | type=%s | turns=%s",
        state.get("lead_name"),
        escalation_type,
        state.get("turn_count"),
    )

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "needs_nurture": True,
        "current_node":  "graceful_exit",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────

def route_after_fast_qualify(state: EscalationState) -> str:
    if state.get("appointment_set"):
        return "confirm_appointment"

    lpmama   = state.get("lpmama", {})
    # Escalation close trigger: urgency + location + money_source filled, or all collected
    critical = ["urgency", "location", "money_source"]
    filled   = sum(1 for k in critical if lpmama.get(k))

    if filled >= 2 or all(v is not None for v in lpmama.values()):
        return "urgency_close"
    if state.get("turn_count", 0) >= ESCALATION_TURN_LIMIT:
        return "urgency_close"  # force close attempt at limit, don't give up
    return "fast_qualify"


def route_after_urgency_close(state: EscalationState) -> str:
    if state.get("appointment_set"):
        return "confirm_appointment"
    if state.get("objection_count", 0) >= 2:
        return "graceful_exit"
    return "fast_qualify"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_escalation_graph() -> StateGraph:
    graph = StateGraph(EscalationState)

    graph.add_node("initialize",          node_initialize)
    graph.add_node("urgency_open",        node_urgency_open)
    graph.add_node("fast_qualify",        node_fast_qualify)
    graph.add_node("urgency_close",       node_urgency_close)
    graph.add_node("confirm_appointment", node_confirm_appointment)
    graph.add_node("graceful_exit",       node_graceful_exit)

    graph.set_entry_point("initialize")

    graph.add_edge("initialize",   "urgency_open")
    graph.add_edge("urgency_open", "fast_qualify")

    graph.add_conditional_edges(
        "fast_qualify",
        route_after_fast_qualify,
        {
            "fast_qualify":        "fast_qualify",
            "urgency_close":       "urgency_close",
            "confirm_appointment": "confirm_appointment",
        },
    )
    graph.add_conditional_edges(
        "urgency_close",
        route_after_urgency_close,
        {
            "confirm_appointment": "confirm_appointment",
            "fast_qualify":        "fast_qualify",
            "graceful_exit":       "graceful_exit",
        },
    )

    graph.add_edge("confirm_appointment", END)
    graph.add_edge("graceful_exit",       END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_escalation_graph = None


def get_escalation_graph():
    global _escalation_graph
    if _escalation_graph is None:
        _escalation_graph = build_escalation_graph()
    return _escalation_graph


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT — signature locked, called by lead_type_router
# ─────────────────────────────────────────────────────────────────────────────

async def run_escalation_pipeline(context: dict) -> dict:
    payload   = context.get("payload", {})
    full_name = payload.get("full_name", "")
    lead_name = full_name.split()[0] if full_name.strip() else ""

    flags           = context.get("flags", [])
    escalation_type = _detect_escalation_type(flags)

    initial_state: EscalationState = {
        "lead_id":           payload.get("lead_id", ""),
        "lead_name":         lead_name,
        "market":            context.get("market", "dallas_fort_worth"),
        "source":            context.get("source", "website"),
        "lead_type":         context.get("lead_type", "buyer"),
        "timeline":          context.get("timeline", "unknown"),
        "confidence":        context.get("confidence", 0),
        "flags":             flags,
        "escalation_type":   escalation_type,
        "messages":          [],
        "turn_count":        0,
        "lpmama":            DEFAULT_ESCALATION_LPMAMA.copy(),
        "objection_count":   0,
        "appointment_set":   False,
        "needs_nurture":     False,
        "current_node":      "",
        "system_prompt":     "",
        "last_response":     "",
        "lead_last_message": payload.get("message", "") or "",
    }

    graph  = get_escalation_graph()
    result = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": 30},
    )

    logger.info(
        "ESCALATION COMPLETE | type=%s | appointment=%s | turns=%s",
        result.get("escalation_type"),
        result.get("appointment_set"),
        result.get("turn_count"),
    )

    return {
        "status":          "completed",
        "pipeline":        "escalation",
        "escalation_type": result.get("escalation_type", ""),
        "appointment_set": result.get("appointment_set", False),
        "needs_nurture":   result.get("needs_nurture", False),
        "turn_count":      result.get("turn_count", 0),
        "lpmama":          result.get("lpmama", {}),
        "last_response":   result.get("last_response", ""),
        "market":          result.get("market", ""),
        "flags":           result.get("flags", []),
        "context":         context,
    }
