"""
services/scorer.py
CorePilora AI — Lead Scoring Engine

Scores every lead 1-100 based on:
- Motivation strength
- Timeline urgency
- Financial readiness
- Engagement level
- Market activity

Score drives: call priority, nurture urgency, Jaiyana's closing intensity.
Hot lead = immediate aggressive pursuit.
Cold lead = nurture sequence.
"""

from __future__ import annotations

import logging
from typing import Optional

from services.classifier import (
    ClassificationResult,
    Timeline,
    FinanceType,
    LeadType,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SCORING WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

class ScoreWeights:
    """Scoring component weights. Total must equal 100."""
    TIMELINE     = 30    # How soon are they moving
    FINANCE      = 25    # Can they actually close
    CONFIDENCE   = 20    # Classifier confidence in lead type
    MOTIVATION   = 15    # Signal strength from classifier
    ENGAGEMENT   = 10    # Source quality and initial engagement


# ─────────────────────────────────────────────────────────────────────────────
# TIMELINE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def _score_timeline(timeline: str) -> int:
    """
    HOT (0-30 days)    = 30 points
    WARM (31-90 days)  = 18 points
    NURTURE (90+ days) = 6 points
    UNKNOWN            = 10 points
    """
    scores = {
        Timeline.HOT.value:     30,
        Timeline.WARM.value:    18,
        Timeline.NURTURE.value: 6,
        Timeline.UNKNOWN.value: 10,
    }
    return scores.get(timeline, 10)


# ─────────────────────────────────────────────────────────────────────────────
# FINANCE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def _score_finance(finance_type: str) -> int:
    """
    CASH            = 25 points — ready to close now
    PRE_APPROVED    = 22 points — financing in place
    INVESTOR_LOC    = 18 points — experienced, has capital access
    NEEDS_LENDER    = 8 points  — not ready, needs work
    UNKNOWN         = 10 points
    """
    scores = {
        FinanceType.CASH.value:         25,
        FinanceType.PRE_APPROVED.value: 22,
        FinanceType.INVESTOR_LOC.value: 18,
        FinanceType.NEEDS_LENDER.value: 8,
        FinanceType.UNKNOWN.value:      10,
    }
    return scores.get(finance_type, 10)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def _score_confidence(confidence: int) -> int:
    """
    Map classifier confidence (0-100) to scoring component (0-20).
    High confidence = clear intent = higher score.
    """
    if confidence >= 80:
        return 20
    if confidence >= 60:
        return 15
    if confidence >= 40:
        return 10
    if confidence >= 20:
        return 5
    return 2


# ─────────────────────────────────────────────────────────────────────────────
# MOTIVATION SCORE
# Based on raw signal strength from classifier.
# ─────────────────────────────────────────────────────────────────────────────

def _score_motivation(score_breakdown: dict, lead_type: str) -> int:
    """
    Use winner score from classifier as motivation proxy.
    Higher raw score = stronger expressed motivation.
    """
    winner_score = score_breakdown.get(lead_type, 0)

    if winner_score >= 20:
        return 15
    if winner_score >= 14:
        return 12
    if winner_score >= 7:
        return 8
    if winner_score >= 3:
        return 4
    return 2


# ─────────────────────────────────────────────────────────────────────────────
# ENGAGEMENT SCORE
# Based on lead source quality.
# ─────────────────────────────────────────────────────────────────────────────

def _score_engagement(source: str) -> int:
    """
    WEBSITE  = 10 — highest intent, they sought you out
    ZILLOW   = 8  — active property search
    FACEBOOK = 5  — ad response, lower intent
    INSTAGRAM = 4 — visual engagement, lowest intent
    UNKNOWN  = 3
    """
    scores = {
        "website":   10,
        "zillow":    8,
        "facebook":  5,
        "instagram": 4,
        "unknown":   3,
    }
    return scores.get(source, 3)


# ─────────────────────────────────────────────────────────────────────────────
# FLAG BONUSES
# Escalation flags add bonus points.
# ─────────────────────────────────────────────────────────────────────────────

def _score_flags(flags: list) -> int:
    """
    DISTRESSED_SELLER   = +10  — urgent situation, high motivation
    HIGH_MOTIVATION     = +8   — life event driving move
    HOT_INVESTOR_CLOSE  = +10  — deadline driven investor
    """
    bonus = 0
    flag_bonuses = {
        "DISTRESSED_SELLER":  10,
        "HIGH_MOTIVATION":    8,
        "HOT_INVESTOR_CLOSE": 10,
    }
    for flag in flags:
        bonus += flag_bonuses.get(flag, 0)
    return bonus


# ─────────────────────────────────────────────────────────────────────────────
# LEAD TEMPERATURE
# ─────────────────────────────────────────────────────────────────────────────

def get_temperature(score: int) -> str:
    """
    Convert numeric score to temperature label.
    Used for display, logging, and priority routing.
    """
    if score >= 75:
        return "HOT"
    if score >= 50:
        return "WARM"
    if score >= 30:
        return "COOL"
    return "COLD"


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — SCORE LEAD
# ─────────────────────────────────────────────────────────────────────────────

def score_lead(result: ClassificationResult) -> dict:
    """
    Score a classified lead on a 1-100 scale.

    Args:
        result: Full ClassificationResult from classify_lead().

    Returns:
        dict with score, temperature, and component breakdown.
    """
    timeline_score    = _score_timeline(result.timeline.value)
    finance_score     = _score_finance(result.finance_type.value)
    confidence_score  = _score_confidence(result.confidence)
    motivation_score  = _score_motivation(
        result.score_breakdown,
        result.lead_type.value,
    )
    engagement_score  = _score_engagement(result.source.value)
    flag_bonus        = _score_flags(result.flags)

    raw_score = (
        timeline_score
        + finance_score
        + confidence_score
        + motivation_score
        + engagement_score
        + flag_bonus
    )

    # Cap at 100
    final_score = min(raw_score, 100)
    temperature = get_temperature(final_score)

    logger.info(
        "SCORER | lead_type=%s | score=%s | temp=%s | "
        "timeline=%s finance=%s confidence=%s motivation=%s engagement=%s flags=%s",
        result.lead_type.value,
        final_score,
        temperature,
        timeline_score,
        finance_score,
        confidence_score,
        motivation_score,
        engagement_score,
        flag_bonus,
    )

    return {
        "score":       final_score,
        "temperature": temperature,
        "breakdown": {
            "timeline":    timeline_score,
            "finance":     finance_score,
            "confidence":  confidence_score,
            "motivation":  motivation_score,
            "engagement":  engagement_score,
            "flag_bonus":  flag_bonus,
        },
    }