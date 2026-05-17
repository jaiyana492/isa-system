"""
agents/investor_graph.py
CorePilora AI — Investor Closing Pipeline

Full working LangGraph implementation.
Groq LLM powered via shared services.groq_client.
Investor psychology: numbers first, emotion second.
Appointment detection fixed — reads lead input, not Jaiyana output.
LPMAMA extraction rate fixed — every 3 turns, not every turn.
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

DEFAULT_INVESTOR_LPMAMA = {
    "location":     None,
    "price_point":  None,
    "motivation":   None,
    "agent":        None,
    "money_source": None,
    "appointment":  None,
}


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class InvestorState(TypedDict):
    lead_id:              str
    lead_name:            str
    market:               str
    source:               str
    timeline:             str
    confidence:           int
    flags:                list[str]
    messages:             list[dict]
    turn_count:           int
    lpmama:               dict
    investment_strategy:  str
    portfolio_size:       str
    finance_type:         str
    objection_count:      int
    appointment_set:      bool
    needs_nurture:        bool
    current_node:         str
    system_prompt:        str
    last_response:        str
    lead_last_message:    str


# ─────────────────────────────────────────────────────────────────────────────
# LPMAMA EXTRACTOR — INVESTOR
# Only runs every 3 turns to avoid token waste.
# ─────────────────────────────────────────────────────────────────────────────

async def _extract_investor_lpmama(
    conversation: list[dict],
    current_lpmama: dict,
    turn_count: int,
) -> dict:
    if turn_count % 3 != 0:
        return current_lpmama

    extraction_prompt = """
You are a data extraction engine for a real estate ISA system.
Read the conversation and extract confirmed investor qualification data.

Return ONLY valid JSON with these exact keys:
{
  "location": "confirmed target market or submarket or null",
  "price_point": "confirmed budget per deal or null",
  "motivation": "confirmed investment strategy or goal or null",
  "agent": "confirmed yes/no working with agent or null",
  "money_source": "confirmed cash/DSCR/hard money/LOC or null",
  "appointment": "confirmed appointment time or null"
}

Only include values the investor explicitly stated.
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
        logger.warning("INVESTOR LPMAMA EXTRACTION FAILED | error=%s", str(e))
        return current_lpmama


# ─────────────────────────────────────────────────────────────────────────────
# NODE — INITIALIZE
# ─────────────────────────────────────────────────────────────────────────────

async def node_initialize(state: InvestorState) -> InvestorState:
    system_prompt = build_system_prompt(
        lead_type="investor",
        market=state.get("market", "Dallas-Fort Worth"),
        lead_source=state.get("source", "website"),
        lead_name=state.get("lead_name", ""),
        lpmama_state=state.get("lpmama", DEFAULT_INVESTOR_LPMAMA.copy()),
    )

    logger.info(
        "INVESTOR | INIT | lead=%s | market=%s | finance=%s",
        state.get("lead_name"),
        state.get("market"),
        state.get("finance_type"),
    )

    return {
        **state,
        "system_prompt":       system_prompt,
        "current_node":        "initialize",
        "turn_count":          0,
        "objection_count":     0,
        "appointment_set":     False,
        "needs_nurture":       False,
        "last_response":       "",
        "lead_last_message":   "",
        "messages":            state.get("messages", []),
        "lpmama":              state.get("lpmama", DEFAULT_INVESTOR_LPMAMA.copy()),
        "investment_strategy": state.get("investment_strategy", "unknown"),
        "portfolio_size":      state.get("portfolio_size", "unknown"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — OPENING
# ─────────────────────────────────────────────────────────────────────────────

async def node_opening(state: InvestorState) -> InvestorState:
    source            = state.get("source", "website")
    name              = state.get("lead_name", "")
    market            = state.get("market", "Dallas-Fort Worth")
    messages          = state.get("messages", [])
    lead_last_message = state.get("lead_last_message", "")

    lead_context = (
        f'Their inquiry: "{lead_last_message}". Speak to their specific angle.'
        if lead_last_message else ""
    )
    instruction = f"""
Make first contact with {name or 'this investor'}.
Source: {source}. Market: {market}. {lead_context}
This is an investor. Skip lifestyle language completely.
Lead with a specific market intelligence data point or direct strategy question.
Under 2 sentences. Direct. Respect their time. End with one question.
Make them feel you speak their language — numbers, returns, strategy.
"""
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.8,
            max_tokens=80,
        )
    except Exception as e:
        logger.error("INVESTOR | OPENING FAILED | error=%s", str(e))
        response = (
            f"Hey {name} — Jaiyana here. "
            f"I work specifically with investors in {market}. "
            f"What strategy are you running right now?"
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
# NODE — STRATEGY DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

async def node_strategy_discovery(state: InvestorState) -> InvestorState:
    market   = state.get("market", "Dallas-Fort Worth")
    messages = state.get("messages", [])

    lead_last_message = state.get("lead_last_message", "")
    lead_said = f'They just said: "{lead_last_message}"' if lead_last_message else ""
    instruction = f"""
{lead_said}
TURN ARCHITECTURE:
(1) ACKNOWLEDGE — mirror their last 2-3 words as a question, or briefly affirm the substance of what they said. No "great answer."
(2) ADVANCE — one focused question to uncover investment strategy in {market}: buy-and-hold / flip / STR / multifamily, existing portfolio or first deal, target return type.
STOP after the question. Under 2 sentences. Speak like a fellow investor. Numbers and strategy only.
"""
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )
    except Exception as e:
        logger.error("INVESTOR | STRATEGY FAILED | error=%s", str(e))
        response = "Are you running buy-and-hold or more of a flip operation — and is this adding to an existing portfolio?"

    messages = messages + [{"role": "assistant", "content": response}]

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "current_node":  "strategy_discovery",
        "turn_count":    state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — MARKET INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

async def node_market_intelligence(state: InvestorState) -> InvestorState:
    market   = state.get("market", "Dallas-Fort Worth")
    strategy = state.get("investment_strategy", "buy_hold")
    messages = state.get("messages", [])

    lead_last_message = state.get("lead_last_message", "")
    lead_said = f'They revealed: "{lead_last_message}".' if lead_last_message else ""
    instruction = f"""
{lead_said}
Investor strategy: {strategy}. Market: {market}.
Acknowledge what they told you about their strategy briefly. Then deliver market intelligence specific to that strategy in {market}.
Use exact numbers — cap rates, yields, appreciation percentages, days on market. Reference:
- DFW: corporate relocation, 5.2-7.1% cap rates outer suburbs, 100k+ annual relocations
- Houston: Pearland/Sugar Land/Katy cash-flow corridors still positive, energy sector anchor
- Orlando: STR gross yields 8-14% Kissimmee corridor, 75M tourists/year demand floor
- Tampa: 71% appreciation over 5 years, migration pipeline still active
- Miami: $1B+ international inflows, 22% non-US buyer, durable appreciation
Build authority. Position as their intelligence edge.
Under 4 sentences. End with one qualifying question targeting: money source or price point per door.
"""
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.7,
            max_tokens=200,
        )
    except Exception as e:
        logger.error("INVESTOR | MARKET INTEL FAILED | error=%s", str(e))
        response = (
            f"In {market} the submarkets with strongest cash-on-cash returns "
            f"are moving fast right now. What price point per door are you targeting?"
        )

    messages = messages + [{"role": "assistant", "content": response}]

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "current_node":  "market_intelligence",
        "turn_count":    state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — QUALIFY
# Checks lead's incoming message for appointment confirmation.
# ─────────────────────────────────────────────────────────────────────────────

async def node_qualify(state: InvestorState) -> InvestorState:
    lpmama            = state.get("lpmama", {})
    missing           = [k for k, v in lpmama.items() if v is None]
    messages          = state.get("messages", [])
    lead_last_message = state.get("lead_last_message", "")

    if lead_confirmed_appointment(lead_last_message):
        logger.info("INVESTOR | APPOINTMENT CONFIRMED BY LEAD")
        return {
            **state,
            "appointment_set": True,
            "current_node":    "qualify",
            "turn_count":      state.get("turn_count", 0) + 1,
        }

    lead_said = f'The investor just said: "{lead_last_message}"' if lead_last_message else ""
    next_gate = missing[0].upper() if missing else "APPOINTMENT CLOSE"
    instruction = f"""
{lead_said}
TURN ARCHITECTURE — EXECUTE THIS EXACTLY:
(1) ACKNOWLEDGE — mirror their last 2-3 words as a question OR briefly affirm the substance. No filler like "great."
(2) DEEPEN if they revealed a deal thesis, portfolio context, or specific constraint — one follow-up before advancing.
(3) ONE direct question advancing toward: {next_gate}. Money source is critical — no funding, no deal.
STOP. Under 2 sentences. Numbers. Direct. Fellow investor language only.
"""
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )
    except Exception as e:
        logger.error("INVESTOR | QUALIFY FAILED | error=%s", str(e))
        response = "Are you working with cash, DSCR financing, or hard money on this?"

    messages = messages + [{"role": "assistant", "content": response}]

    updated_lpmama = await _extract_investor_lpmama(
        conversation=messages,
        current_lpmama=lpmama,
        turn_count=state.get("turn_count", 0),
    )

    logger.info(
        "INVESTOR | QUALIFY | turn=%s | collected=%s | missing=%s",
        state.get("turn_count", 0),
        [k for k, v in updated_lpmama.items() if v],
        [k for k, v in updated_lpmama.items() if not v],
    )

    return {
        **state,
        "messages":      messages,
        "last_response": response,
        "lpmama":        updated_lpmama,
        "current_node":  "qualify",
        "turn_count":    state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE — HANDLE OBJECTION
# ─────────────────────────────────────────────────────────────────────────────

async def node_handle_objection(state: InvestorState) -> InvestorState:
    count             = state.get("objection_count", 0)
    market            = state.get("market", "Dallas-Fort Worth")
    messages          = state.get("messages", [])
    lead_last_message = state.get("lead_last_message", "")

    lead_said = f'Their exact pushback: "{lead_last_message}"' if lead_last_message else "Investor pushed back."
    instruction = f"""
{lead_said}
Objection #{count + 1}. Market: {market}.
TURN ARCHITECTURE:
(1) BRIEFLY ACKNOWLEDGE the specific concern — one short phrase. No emotional language. "Understood." or mirror their words.
(2) COUNTER with math — completely different angle: wrong numbers = better submarket; market too hot = specific corridor data; need to think = opportunity cost calculation; have agent = deal access difference.
(3) End with one direct question.
Under 3 sentences. Pure logic. No filler.
"""
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.7,
            max_tokens=180,
        )
    except Exception as e:
        logger.error("INVESTOR | OBJECTION FAILED | error=%s", str(e))
        response = "Let me show you a different submarket — same budget, stronger returns. Cash flow or appreciation — which matters more to you?"

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
# Offers options only. Lead must confirm. Detected in node_qualify.
# ─────────────────────────────────────────────────────────────────────────────

async def node_close_appointment(state: InvestorState) -> InvestorState:
    name     = state.get("lead_name", "")
    market   = state.get("market", "Dallas-Fort Worth")
    lpmama   = state.get("lpmama", {})
    strategy = state.get("investment_strategy", "unknown")
    messages = state.get("messages", [])

    instruction = f"""
Qualification complete for {name or 'this investor'}.
Market: {market}. Strategy: {strategy}.
Data: {json.dumps(lpmama)}.

Offer the investment strategy session now.
Frame as: deal pipeline review + market analysis + numbers walkthrough.
Not a generic meeting. A working session with real data.
Two specific time options this week.
Direct. No fluff.
Do NOT confirm the appointment yourself. Wait for their response.
"""
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.6,
            max_tokens=180,
        )
    except Exception as e:
        logger.error("INVESTOR | CLOSE FAILED | error=%s", str(e))
        response = (
            f"Here is what I want to do {name} — "
            f"pull the current deal pipeline for {market}, "
            f"run the numbers live, map out your acquisition plan. "
            f"I have Tuesday at 10am or Thursday at 2pm. Which works?"
        )

    messages = messages + [{"role": "assistant", "content": response}]

    logger.info("INVESTOR | CLOSE OFFERED | lead=%s", state.get("lead_name"))

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

async def node_confirm_appointment(state: InvestorState) -> InvestorState:
    name     = state.get("lead_name", "")
    messages = state.get("messages", [])

    instruction = f"""
{name or 'Investor'} just confirmed the strategy session.
Brief confirmation. State what you will bring — deal analysis and market data.
Under 2 sentences. Professional. Direct.
"""
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.6,
            max_tokens=100,
        )
    except Exception as e:
        logger.error("INVESTOR | CONFIRM FAILED | error=%s", str(e))
        response = f"Locked in {name}. I will have the deal analysis and market data ready. Talk then."

    messages = messages + [{"role": "assistant", "content": response}]

    logger.info("INVESTOR | CONFIRMED | lead=%s", state.get("lead_name"))

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

async def node_route_nurture(state: InvestorState) -> InvestorState:
    name     = state.get("lead_name", "")
    market   = state.get("market", "Dallas-Fort Worth")
    messages = state.get("messages", [])

    instruction = f"""
{name or 'This investor'} is not ready right now in {market}.
Exit professionally. Reference deal flow follow-up.
Under 2 sentences. Direct. Zero desperation.
Investors respect efficiency — keep it tight.
"""
    messages = messages + [{"role": "user", "content": instruction}]

    try:
        response = await call_groq(
            system_prompt=state["system_prompt"],
            messages=messages,
            temperature=0.7,
            max_tokens=80,
        )
    except Exception as e:
        logger.error("INVESTOR | NURTURE EXIT FAILED | error=%s", str(e))
        response = (
            f"Understood {name}. "
            f"When the right deal hits in {market} I will send it your way directly."
        )

    messages = messages + [{"role": "assistant", "content": response}]

    logger.info(
        "INVESTOR | NURTURE | lead=%s | objections=%s",
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
# All use .get() with defaults — no KeyError on missing state keys.
# ─────────────────────────────────────────────────────────────────────────────

def route_after_opening(state: InvestorState) -> str:
    if "HOT_INVESTOR_CLOSE" in state.get("flags", []):
        return "qualify"
    return "strategy_discovery"


def route_after_strategy(state: InvestorState) -> str:
    strategy = state.get("investment_strategy", "unknown")
    if strategy in ["buy_hold", "str", "multifamily"]:
        return "market_intelligence"
    return "qualify"


def route_after_qualify(state: InvestorState) -> str:
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


def route_after_objection(state: InvestorState) -> str:
    if state.get("objection_count", 0) >= 5:
        return "route_nurture"
    return "qualify"


def route_after_close(state: InvestorState) -> str:
    if state.get("appointment_set"):
        return "confirm_appointment"
    if state.get("objection_count", 0) >= 3:
        return "route_nurture"
    return "handle_objection"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_investor_graph() -> StateGraph:
    graph = StateGraph(InvestorState)

    graph.add_node("initialize",          node_initialize)
    graph.add_node("opening",             node_opening)
    graph.add_node("strategy_discovery",  node_strategy_discovery)
    graph.add_node("market_intelligence", node_market_intelligence)
    graph.add_node("qualify",             node_qualify)
    graph.add_node("handle_objection",    node_handle_objection)
    graph.add_node("close_appointment",   node_close_appointment)
    graph.add_node("confirm_appointment", node_confirm_appointment)
    graph.add_node("route_nurture",       node_route_nurture)

    graph.set_entry_point("initialize")

    graph.add_edge("initialize",          "opening")
    graph.add_edge("market_intelligence", "qualify")

    graph.add_conditional_edges(
        "opening",
        route_after_opening,
        {
            "strategy_discovery": "strategy_discovery",
            "qualify":            "qualify",
        },
    )
    graph.add_conditional_edges(
        "strategy_discovery",
        route_after_strategy,
        {
            "market_intelligence": "market_intelligence",
            "qualify":             "qualify",
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

_investor_graph = None


def get_investor_graph():
    global _investor_graph
    if _investor_graph is None:
        _investor_graph = build_investor_graph()
    return _investor_graph


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def run_investor_pipeline(context: dict) -> dict:
    full_name = context.get("payload", {}).get("full_name", "")
    lead_name = full_name.split()[0] if full_name else ""

    initial_state: InvestorState = {
        "lead_id":             context.get("payload", {}).get("lead_id", ""),
        "lead_name":           lead_name,
        "market":              context.get("market", "dallas_fort_worth"),
        "source":              context.get("source", "website"),
        "timeline":            context.get("timeline", "unknown"),
        "confidence":          context.get("confidence", 0),
        "flags":               context.get("flags", []),
        "messages":            [],
        "turn_count":          0,
        "lpmama":              DEFAULT_INVESTOR_LPMAMA.copy(),
        "investment_strategy": "unknown",
        "portfolio_size":      "unknown",
        "finance_type":        context.get("finance_type", "unknown"),
        "objection_count":     0,
        "appointment_set":     False,
        "needs_nurture":       False,
        "current_node":        "",
        "system_prompt":       "",
        "last_response":       "",
        "lead_last_message":   context.get("payload", {}).get("message", "") or "",
    }

    graph  = get_investor_graph()
    result = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": 30},
    )

    logger.info(
        "INVESTOR COMPLETE | appointment=%s | nurture=%s | turns=%s",
        result.get("appointment_set"),
        result.get("needs_nurture"),
        result.get("turn_count"),
    )

    return {
        "status":          "completed",
        "pipeline":        "investor",
        "appointment_set": result.get("appointment_set", False),
        "needs_nurture":   result.get("needs_nurture", False),
        "turn_count":      result.get("turn_count", 0),
        "lpmama":          result.get("lpmama", {}),
        "last_response":   result.get("last_response", ""),
        "market":          result.get("market", ""),
        "strategy":        result.get("investment_strategy", ""),
        "context":         context,
    }
