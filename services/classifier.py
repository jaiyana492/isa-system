"""
services/classifier.py
CorePilora AI — Real Estate Lead Intelligent System (ISA)

Multi-layer lead classification engine.
Markets: Dallas-Fort Worth | Houston | Orlando | Tampa Bay | Miami
Sources: Zillow | Facebook Ads | Instagram Ads | Website Forms
Lead Types: BUYER | SELLER | INVESTOR
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class LeadType(str, Enum):
    BUYER    = "buyer"
    SELLER   = "seller"
    INVESTOR = "investor"
    UNKNOWN  = "unknown"


class LeadSource(str, Enum):
    ZILLOW    = "zillow"
    FACEBOOK  = "facebook"
    INSTAGRAM = "instagram"
    WEBSITE   = "website"
    UNKNOWN   = "unknown"


class Market(str, Enum):
    DFW     = "dallas_fort_worth"
    HOUSTON = "houston"
    ORLANDO = "orlando"
    TAMPA   = "tampa"
    MIAMI   = "miami"
    DUBAI   = "dubai"        # future expansion
    UNKNOWN = "unknown"


class Timeline(str, Enum):
    HOT     = "hot"       # 0–30 days
    WARM    = "warm"      # 31–90 days
    NURTURE = "nurture"   # 90+ days / vague
    UNKNOWN = "unknown"


class FinanceType(str, Enum):
    CASH         = "cash"
    PRE_APPROVED = "pre_approved"
    NEEDS_LENDER = "needs_lender"
    INVESTOR_LOC = "investor_loc"   # Line of credit / hard money / DSCR
    UNKNOWN      = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT TIERS
# Every signal is not equal. High-intent phrases crush vague language.
# ─────────────────────────────────────────────────────────────────────────────

class W:
    """Signal weight constants. Adjust only with data to back it up."""
    EXPLICIT    = 10   # "I want to buy a home right now"
    STRONG      = 7    # "pre-approved", "listing appointment"
    MODERATE    = 4    # submarket mention, property type
    SOFT        = 2    # lifestyle signal, vague curiosity
    SUBMARKET   = 3    # market geography signal
    SOURCE_BIAS = 3    # source-based correction weight


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL LIBRARY — BUYER
# ─────────────────────────────────────────────────────────────────────────────

BUYER_SIGNALS: list[tuple[int, list[str]]] = [

    # ── EXPLICIT (10) ──────────────────────────────────────────────────────
    (W.EXPLICIT, [
        "want to buy", "looking to buy", "ready to buy", "need to buy",
        "buying a home", "buying a house", "purchase a home",
        "purchase a house", "find me a home", "help me buy",
        "first home", "first house", "forever home", "dream home",
        "starter home", "primary residence", "owner occupied",
        "want to move in", "ready to move", "place to live",
        "need a place", "looking for a place to live",
    ]),

    # ── STRONG (7) ────────────────────────────────────────────────────────
    (W.STRONG, [
        "pre-approved", "pre approved", "got pre-approved", "already approved",
        "approval letter", "pre-qual", "pre qualification",
        "working with a lender", "lender approved", "mortgage approved",
        "fha loan", "va loan", "conventional loan", "usda loan",
        "jumbo loan", "down payment ready", "earnest money",
        "cash buyer", "paying cash for a home",
        "making an offer", "want to make an offer", "submit an offer",
        "under contract on a home", "closing soon", "closing in",
        "moving in 30", "moving in 60", "moving in 90",
        "need to be in by", "school starts", "lease ends",
        "lease is up", "lease ending",
    ]),

    # ── MODERATE (4) ──────────────────────────────────────────────────────
    (W.MODERATE, [
        "3 bedroom", "4 bedroom", "5 bedroom", "3 bed", "4 bed",
        "2 bath", "3 bath", "2 car garage", "single family",
        "single-family", "townhouse", "townhome", "patio home",
        "new construction home", "move-in ready", "move in ready",
        "good school district", "top schools", "rated schools",
        "gated community", "master on main", "master down",
        "open floor plan", "backyard", "pool home", "hoa community",
        "waterfront home", "lake view", "ocean view",
        "golf community", "quiet street", "family friendly",
        "price range", "budget is", "in the", "up to",
        "under 200k", "under 300k", "under 400k", "under 500k",
        "under 600k", "300k", "350k", "400k", "450k",
        "500k", "550k", "600k", "650k", "700k", "800k",
        "zillow listing", "saw on zillow", "favorite on zillow",
        "redfin", "trulia", "realtor.com", "homes.com",
    ]),

    # ── SOFT (2) ──────────────────────────────────────────────────────────
    (W.SOFT, [
        "just looking", "browsing homes", "exploring options",
        "curious about homes", "thinking about buying",
        "maybe next year", "in a few months",
        "what can i afford", "how does buying work",
        "is now a good time to buy", "mortgage rates",
        "interest rates", "what's available",
    ]),
]


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL LIBRARY — SELLER
# ─────────────────────────────────────────────────────────────────────────────

SELLER_SIGNALS: list[tuple[int, list[str]]] = [

    # ── EXPLICIT (10) ──────────────────────────────────────────────────────
    (W.EXPLICIT, [
        "want to sell", "need to sell", "selling my home",
        "sell my house", "sell my property", "list my home",
        "list my house", "put my house on market", "ready to sell",
        "selling my condo", "want to list", "need to list",
        "get rid of my house", "offload my property",
        "move out and sell",
    ]),

    # ── STRONG (7) ────────────────────────────────────────────────────────
    (W.STRONG, [
        "what is my home worth", "home value", "what can i sell for",
        "how much is my house worth", "what will i net",
        "how much can i get", "property value", "market analysis",
        "comparative market analysis", "cma", "home appraisal",
        "need to sell fast", "sell fast", "quick sale",
        "need cash from my home", "equity", "cash out",
        "foreclosure", "pre-foreclosure", "behind on mortgage",
        "behind on payments", "short sale", "underwater",
        "divorce settlement", "estate sale", "probate sale",
        "inherited property", "inherited house", "need to sell inherited",
        "1031 exchange sell side", "relocating and selling",
    ]),

    # ── MODERATE (4) ──────────────────────────────────────────────────────
    (W.MODERATE, [
        "downsizing", "downsize", "empty nester",
        "kids moved out", "retirement home", "retiring and selling",
        "upsizing", "need more space", "outgrown the house",
        "job transfer", "moving out of state", "relocation package",
        "insurance too high", "property taxes too high",
        "hoa too expensive", "maintenance too much",
        "zestimate", "comps in my area", "what are homes selling for",
        "for sale by owner", "fsbo", "thinking of selling",
        "seller's market", "good time to sell",
        "hurricane damage", "flood zone issues",
        "snowbird selling", "second home selling",
        "vacation home selling", "seasonal home",
    ]),

    # ── SOFT (2) ──────────────────────────────────────────────────────────
    (W.SOFT, [
        "considering selling", "maybe sell", "not sure if i should sell",
        "just curious what my home is worth",
        "is it a good time to sell", "thinking about it",
        "weighing options", "exploring selling",
    ]),
]


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL LIBRARY — INVESTOR
# ─────────────────────────────────────────────────────────────────────────────

INVESTOR_SIGNALS: list[tuple[int, list[str]]] = [

    # ── EXPLICIT (10) ──────────────────────────────────────────────────────
    (W.EXPLICIT, [
        "investment property", "rental property", "income property",
        "looking to invest", "want to invest in real estate",
        "building a portfolio", "expanding portfolio", "add to my portfolio",
        "passive income property", "buy and hold", "cash flowing property",
        "want cash flow", "want a rental", "landlord",
        "buy a rental", "acquire rental",
    ]),

    # ── STRONG (7) ────────────────────────────────────────────────────────
    (W.STRONG, [
        "cap rate", "capitalization rate", "noi", "net operating income",
        "cash on cash", "cash-on-cash", "coc return",
        "gross rent multiplier", "grm", "irr", "arv",
        "after repair value", "rehab budget", "rehab cost",
        "fix and flip", "flipping houses", "flip a house",
        "wholesale deal", "off market deal", "pocket listing",
        "brrrr", "brrr", "buy rehab rent refinance repeat",
        "dscr loan", "dscr", "hard money", "private money lender",
        "1031 exchange", "tax deferred exchange",
        "multi family", "multifamily", "duplex", "triplex",
        "fourplex", "4plex", "apartment building", "small apartment",
        "short term rental", "airbnb", "vrbo", "str",
        "vacation rental income", "furnished rental",
        "corporate housing", "medium term rental",
        "nnn", "triple net", "commercial lease",
    ]),

    # ── MODERATE (4) ──────────────────────────────────────────────────────
    (W.MODERATE, [
        "cash flow", "monthly cash flow", "positive cash flow",
        "appreciation play", "value add", "value-add property",
        "distressed property", "distressed asset",
        "equity play", "leverage", "debt service",
        "already own rentals", "i own properties",
        "currently have rentals", "my portfolio",
        "generational wealth", "wealth building",
        "real estate portfolio", "rei", "real estate investing",
        "passive income", "financial freedom",
        "10 units", "20 units", "50 units", "100 doors",
        "mixed use", "commercial property", "retail space",
        "warehouse", "industrial", "flex space",
        "subject to", "seller finance", "creative finance",
        "pre construction", "pre-construction", "off plan",
        "new development investment",
    ]),

    # ── SOFT (2) ──────────────────────────────────────────────────────────
    (W.SOFT, [
        "thinking about investing", "curious about rentals",
        "is real estate a good investment",
        "thinking about a rental property",
        "should i buy a rental", "how does investing work",
        "roi", "return on investment", "tax benefits",
        "write off", "depreciation", "equity growth",
    ]),
]


# ─────────────────────────────────────────────────────────────────────────────
# MARKET DETECTION SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

MARKET_SIGNALS: dict[Market, list[str]] = {

    Market.DFW: [
        # Core cities
        "dallas", "fort worth", "dfw", "north texas",
        # Buyer/Seller hotspots
        "frisco", "plano", "mckinney", "allen", "prosper", "celina",
        "fairview", "murphy", "wylie", "rockwall", "rowlett",
        "richardson", "garland", "grand prairie", "arlington",
        "mansfield", "midlothian", "desoto", "cedar hill",
        "highland park", "university park", "uptown dallas",
        "lake highlands", "southlake", "westlake", "colleyville",
        "grapevine", "flower mound", "lewisville", "denton",
        "little elm", "the colony", "coppell", "irving",
        "carrollton", "addison", "farmers branch", "bedford",
        "euless", "hurst", "keller", "roanoke",
        # Investor submarkets
        "oak cliff", "east dallas", "south dallas", "west dallas",
        "deep ellum", "bishop arts", "lower greenville",
    ],

    Market.HOUSTON: [
        "houston", "htx", "h-town", "harris county",
        "katy", "sugar land", "the woodlands", "woodlands",
        "pearland", "cypress", "spring", "humble",
        "league city", "friendswood", "missouri city",
        "stafford", "richmond", "conroe", "tomball", "magnolia",
        "heights", "montrose", "river oaks", "memorial",
        "bellaire", "west university", "tanglewood",
        "energy corridor", "midtown houston", "downtown houston",
        "third ward", "fifth ward", "east end houston",
        "second ward", "inner loop",
        "rosenberg", "fulshear", "richmond tx",
        "kingwood", "atascocita", "new caney",
    ],

    Market.ORLANDO: [
        "orlando", "central florida", "orange county fl",
        "winter park", "lake nona", "dr phillips", "dr. phillips",
        "windermere", "ocoee", "apopka", "altamonte springs",
        "longwood", "sanford", "kissimmee", "celebration",
        "hunters creek", "metrowest", "baldwin park",
        "oviedo", "winter springs", "casselberry",
        "champions gate", "davenport", "haines city",
        "clermont", "minneola",
    ],

    Market.TAMPA: [
        "tampa", "tampa bay", "hillsborough county",
        "st pete", "saint pete", "saint petersburg", "st. pete",
        "clearwater", "brandon", "riverview", "wesley chapel",
        "land o lakes", "lutz", "new tampa", "south tampa",
        "davis islands", "hyde park", "seminole heights",
        "palm harbor", "dunedin", "safety harbor", "oldsmar",
        "pinellas county", "pasco county", "sarasota",
        "bradenton", "lakeland", "polk county",
    ],

    Market.MIAMI: [
        "miami", "south florida", "miami-dade", "miami dade",
        "miami beach", "coral gables", "coconut grove",
        "brickell", "edgewater", "wynwood", "little havana",
        "doral", "kendall", "homestead", "hialeah",
        "aventura", "sunny isles", "bal harbour", "surfside",
        "pinecrest", "south miami", "north miami",
        "fort lauderdale", "broward county",
        "hollywood fl", "pembroke pines", "miramar",
        "weston", "plantation fl", "davie fl",
        "boca raton", "delray beach", "palm beach",
    ],

    Market.DUBAI: [
        "dubai", "uae", "abu dhabi", "emirates",
        "jvc", "jumeirah village circle", "dubai marina",
        "downtown dubai", "business bay", "palm jumeirah",
        "dubai hills", "difc", "jlt", "jumeirah lake towers",
        "creek harbour", "emaar", "damac", "nakheel", "meraas",
        "golden visa", "off plan", "handover", "freehold dubai",
        "leasehold dubai", "service charge dubai",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# TIMELINE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

TIMELINE_SIGNALS: dict[Timeline, list[str]] = {

    Timeline.HOT: [
        "asap", "as soon as possible", "right now", "immediately",
        "this month", "within 30 days", "30 days", "next 30",
        "end of month", "need to close", "closing soon",
        "already under contract", "already found a home",
        "lease ends next month", "lease up soon",
        "job starts", "transfer date", "moving date set",
        "kids start school", "school starts soon",
        "approved already", "offer submitted",
    ],

    Timeline.WARM: [
        "next 60 days", "next 90 days", "60 days", "90 days",
        "couple months", "few months", "spring", "summer",
        "before summer", "before fall", "this quarter",
        "Q1", "Q2", "Q3", "Q4",
        "end of year", "by december", "by january",
        "a few months", "in about 2 months", "in about 3 months",
        "mid year", "second half of the year",
    ],

    Timeline.NURTURE: [
        "next year", "maybe next year", "sometime next year",
        "down the road", "eventually", "not sure when",
        "just looking", "just browsing", "no rush",
        "when the time is right", "keeping my options open",
        "still thinking", "not ready yet", "exploring",
        "6 months", "6 to 12 months", "a year from now",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# FINANCE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

FINANCE_SIGNALS: dict[FinanceType, list[str]] = {

    FinanceType.CASH: [
        "cash buyer", "paying cash", "all cash", "cash offer",
        "no mortgage", "no financing", "cash purchase",
        "liquid funds", "wire transfer", "funds available",
    ],

    FinanceType.PRE_APPROVED: [
        "pre-approved", "pre approved", "got approved",
        "approval letter", "lender approved", "financing in place",
        "already have financing", "mortgage ready",
        "loan approved", "underwritten",
    ],

    FinanceType.NEEDS_LENDER: [
        "not pre-approved", "need a lender", "looking for a lender",
        "haven't applied", "haven't been approved",
        "not sure about financing", "need to get pre-approved",
        "first step is financing", "where do i start",
    ],

    FinanceType.INVESTOR_LOC: [
        "hard money", "private money", "dscr", "dscr loan",
        "bridge loan", "line of credit", "heloc", "loc",
        "business credit", "llc financing", "entity purchase",
        "portfolio loan", "blanket mortgage",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_SIGNALS: dict[LeadSource, list[str]] = {
    LeadSource.ZILLOW: [
        "zillow", "zestimate", "zillow listing", "saved on zillow",
        "favorite on zillow", "saw on zillow", "zillow estimate",
    ],
    LeadSource.FACEBOOK: [
        "facebook", "fb ad", "facebook ad", "saw your facebook",
        "your facebook post", "facebook reel", "fb post",
    ],
    LeadSource.INSTAGRAM: [
        "instagram", "ig ad", "ig reel", "your reel",
        "instagram ad", "saw on instagram", "your story",
        "instagram post", "ig post",
    ],
    LeadSource.WEBSITE: [
        "your website", "your site", "contact form",
        "filled out your form", "found you online",
        "reached out from your site", "website form",
        "online form", "corepilora.com",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    lead_type:       LeadType
    confidence:      int                        # 0–100
    market:          Market      = Market.UNKNOWN
    timeline:        Timeline    = Timeline.UNKNOWN
    finance_type:    FinanceType = FinanceType.UNKNOWN
    source:          LeadSource  = LeadSource.UNKNOWN
    score_breakdown: dict        = field(default_factory=dict)
    flags:           list[str]   = field(default_factory=list)  # escalation flags

    def is_hot(self) -> bool:
        return self.timeline == Timeline.HOT

    def is_qualified(self) -> bool:
        return (
            self.lead_type != LeadType.UNKNOWN
            and self.confidence >= 40
        )

    def needs_lender(self) -> bool:
        return self.finance_type == FinanceType.NEEDS_LENDER

    def to_dict(self) -> dict:
        return {
            "lead_type":    self.lead_type.value,
            "confidence":   self.confidence,
            "market":       self.market.value,
            "timeline":     self.timeline.value,
            "finance_type": self.finance_type.value,
            "source":       self.source.value,
            "scores":       self.score_breakdown,
            "flags":        self.flags,
            "is_hot":       self.is_hot(),
            "is_qualified": self.is_qualified(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _score_signals(text: str, signal_groups: list[tuple[int, list[str]]]) -> int:
    """
    Score text against weighted signal groups.
    Each matched phrase contributes its tier weight.
    Multiple matches in same tier stack — longer phrases score once.
    """
    total = 0
    for weight, phrases in signal_groups:
        for phrase in phrases:
            if phrase in text:
                total += weight
    return total


def _detect_enum(
    text: str,
    signal_map: dict,
    default,
):
    """Generic detector — returns first enum key whose signals fire."""
    best_key  = default
    best_count = 0
    for key, phrases in signal_map.items():
        count = sum(1 for p in phrases if p in text)
        if count > best_count:
            best_count = count
            best_key   = key
    return best_key


def _detect_flags(text: str) -> list[str]:
    """
    Flag high-urgency or special-routing signals for the ISA.
    These bypass normal flow and get immediate escalation notes.
    """
    flags = []
    distress_phrases = [
        "foreclosure", "pre-foreclosure", "behind on payments",
        "can't afford", "losing my home", "eviction", "short sale",
        "underwater", "owe more than", "bankruptcy",
    ]
    motivation_phrases = [
        "divorce", "separation", "death", "probate",
        "estate sale", "inherited", "job loss", "laid off",
    ]
    hot_investor = [
        "close fast", "close in 7", "close in 10", "all cash offer",
        "1031 deadline", "closing deadline",
    ]

    for p in distress_phrases:
        if p in text:
            flags.append("DISTRESSED_SELLER")
            break
    for p in motivation_phrases:
        if p in text:
            flags.append("HIGH_MOTIVATION")
            break
    for p in hot_investor:
        if p in text:
            flags.append("HOT_INVESTOR_CLOSE")
            break
    return flags


def _source_bias(
    source: LeadSource,
    buyer_score: int,
    seller_score: int,
    investor_score: int,
) -> tuple[int, int, int]:
    """
    Apply source-based score correction.

    Zillow:    Buyer-heavy. Slight investor boost.
    Facebook:  Broad. Slight buyer boost.
    Instagram: Buyer + investor visual-driven.
    Website:   Highest intent — minimal correction needed.
    """
    b, s, i = buyer_score, seller_score, investor_score

    if source == LeadSource.ZILLOW:
        b += W.SOURCE_BIAS
        i += 1

    elif source == LeadSource.FACEBOOK:
        b += W.SOURCE_BIAS - 1
        s += 1

    elif source == LeadSource.INSTAGRAM:
        b += W.SOURCE_BIAS - 1
        i += W.SOURCE_BIAS - 1

    elif source == LeadSource.WEBSITE:
        # Highest intent — trust the raw signals, no correction needed
        pass

    return b, s, i


def _compute_confidence(
    winner_score: int,
    total_score: int,
    second_score: int,
) -> int:
    """
    Confidence = share of winner score, penalized by how close second place is.
    High separation → high confidence. Near-tie → lower confidence.
    """
    if total_score == 0:
        return 0

    base = int((winner_score / total_score) * 100)

    # Separation penalty
    separation = winner_score - second_score
    if separation <= 2:
        base = max(base - 20, 30)
    elif separation <= 5:
        base = max(base - 10, 40)

    return min(base, 98)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def classify_lead(
    raw_message:  str | None = None,
    lead_source:  str | None = None,
    market_hint:  str | None = None,
) -> ClassificationResult:
    """
    Classify an incoming lead into BUYER | SELLER | INVESTOR.

    Args:
        raw_message:  Raw text from the lead (form fill, chat, SMS, webhook body).
        lead_source:  Source string: "zillow" | "facebook" | "instagram" | "website"
        market_hint:  Optional market override (e.g. passed from webhook metadata).

    Returns:
        ClassificationResult with full intelligence breakdown.
    """

    # ── Guard ──────────────────────────────────────────────────────────────
    if not raw_message or not raw_message.strip():
        return ClassificationResult(
            lead_type=LeadType.BUYER,
            confidence=25,
            flags=["EMPTY_MESSAGE"],
        )

    text = _normalize(raw_message)

    # ── Source detection ───────────────────────────────────────────────────
    detected_source = _detect_enum(
        text, SOURCE_SIGNALS, LeadSource.UNKNOWN
    )
    if lead_source:
        manual = lead_source.strip().lower()
        source_map = {
            "zillow":    LeadSource.ZILLOW,
            "facebook":  LeadSource.FACEBOOK,
            "instagram": LeadSource.INSTAGRAM,
            "website":   LeadSource.WEBSITE,
        }
        detected_source = source_map.get(manual, detected_source)

    # ── Raw scoring ────────────────────────────────────────────────────────
    buyer_score    = _score_signals(text, BUYER_SIGNALS)
    seller_score   = _score_signals(text, SELLER_SIGNALS)
    investor_score = _score_signals(text, INVESTOR_SIGNALS)

    # ── Source bias correction ─────────────────────────────────────────────
    buyer_score, seller_score, investor_score = _source_bias(
        detected_source, buyer_score, seller_score, investor_score
    )

    total_score = buyer_score + seller_score + investor_score
    score_breakdown = {
        "buyer":    buyer_score,
        "seller":   seller_score,
        "investor": investor_score,
        "total":    total_score,
    }

    # ── Winner selection ───────────────────────────────────────────────────
    scores = {
        LeadType.BUYER:    buyer_score,
        LeadType.SELLER:   seller_score,
        LeadType.INVESTOR: investor_score,
    }
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner_type, winner_score = sorted_scores[0]
    _, second_score           = sorted_scores[1]

    if total_score == 0:
        return ClassificationResult(
            lead_type=LeadType.BUYER,
            confidence=25,
            source=detected_source,
            score_breakdown=score_breakdown,
            flags=["NO_SIGNAL_MATCH"],
        )

    # ── Confidence ─────────────────────────────────────────────────────────
    confidence = _compute_confidence(winner_score, total_score, second_score)

    # ── Secondary detections ───────────────────────────────────────────────
    detected_market = _detect_enum(text, MARKET_SIGNALS, Market.UNKNOWN)
    if market_hint:
        hint_map = {
            "dallas":           Market.DFW,
            "dallas_fort_worth": Market.DFW,
            "dfw":              Market.DFW,
            "north texas":      Market.DFW,
            "houston":          Market.HOUSTON,
            "orlando":          Market.ORLANDO,
            "central florida":  Market.ORLANDO,
            "tampa":            Market.TAMPA,
            "tampa bay":        Market.TAMPA,
            "miami":            Market.MIAMI,
            "south florida":    Market.MIAMI,
            "dubai":            Market.DUBAI,
        }
        detected_market = hint_map.get(market_hint.strip().lower(), detected_market)

    detected_timeline = _detect_enum(text, TIMELINE_SIGNALS, Timeline.UNKNOWN)
    detected_finance  = _detect_enum(text, FINANCE_SIGNALS,  FinanceType.UNKNOWN)
    flags             = _detect_flags(text)

    return ClassificationResult(
        lead_type=winner_type,
        confidence=confidence,
        market=detected_market,
        timeline=detected_timeline,
        finance_type=detected_finance,
        source=detected_source,
        score_breakdown=score_breakdown,
        flags=flags,
    )