"""
test_api.py
CorePilora AI ISA — API Connection & Integration Test

Usage:
  python test_api.py                        # tests http://localhost:8000
  python test_api.py http://your-domain.com  # tests remote host

Tests (in order):
  1.  GET  /                              Root alive
  2.  GET  /api/v1/health/               Basic health
  3.  GET  /api/v1/health/deep           Deep health (DB + Redis + Groq)
  4.  POST /api/v1/webhook/lead?source=website  Buyer lead
  5.  POST /api/v1/webhook/lead?source=website  Seller lead
  6.  POST /api/v1/webhook/lead?source=website  Investor lead
  7.  POST /api/v1/webhook/lead?source=zillow   Zillow normalizer
  8.  POST /api/v1/webhook/lead           No signature → 401
  9.  POST /api/v1/webhook/lead           Empty body   → 400
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL       = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

if not WEBHOOK_SECRET:
    print("ERROR: WEBHOOK_SECRET not found in .env — cannot compute signatures")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# ANSI COLORS
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _ok(msg: str) -> str:
    return f"{GREEN}[PASS]{RESET}  {msg}"

def _fail(msg: str) -> str:
    return f"{RED}[FAIL]{RESET}  {msg}"

def _warn(msg: str) -> str:
    return f"{YELLOW}[WARN]{RESET}  {msg}"

def _head(msg: str) -> str:
    return f"\n{BOLD}{CYAN}{msg}{RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# SIGNATURE HELPER
# Must match core/security.py verify_webhook_signature exactly.
# ─────────────────────────────────────────────────────────────────────────────

def _sign(body: bytes) -> str:
    """Compute HMAC-SHA256 of body using WEBHOOK_SECRET. Returns 'sha256={hex}'."""
    mac = hmac.new(
        key        = WEBHOOK_SECRET.encode("utf-8"),
        msg        = body,
        digestmod  = hashlib.sha256,
    )
    return f"sha256={mac.hexdigest()}"


def _json_body(payload: dict) -> bytes:
    """Serialize payload to UTF-8 JSON bytes — same representation sent over wire."""
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

results: list[tuple[str, bool]] = []


def _run(
    name:            str,
    method:          str,
    path:            str,
    expected_status: int,
    body:            Optional[dict]  = None,
    headers:         Optional[dict]  = None,
    signed:          bool            = True,
    show_body:       bool            = True,
) -> Optional[dict]:
    """
    Execute one HTTP test.
    Returns parsed JSON response or None on failure.
    """
    url   = BASE_URL + path
    hdrs  = {"Content-Type": "application/json", **(headers or {})}
    raw   = _json_body(body) if body else b""

    if signed and raw:
        hdrs["X-Webhook-Signature"] = _sign(raw)

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=30.0) as client:
            if method == "GET":
                resp = client.get(url, headers=hdrs)
            else:
                resp = client.post(url, content=raw, headers=hdrs)
    except httpx.ConnectError:
        print(_fail(f"{name} — Cannot connect to {BASE_URL}"))
        print(f"         Server not running? Start with: uvicorn main:app --reload")
        results.append((name, False))
        return None
    except Exception as e:
        print(_fail(f"{name} — {e}"))
        results.append((name, False))
        return None

    ms      = round((time.perf_counter() - t0) * 1000)
    passed  = resp.status_code == expected_status

    status_label = f"HTTP {resp.status_code}"
    timing_label = f"{ms}ms"

    if passed:
        print(_ok(f"{name:<45} {status_label}  {timing_label}"))
    else:
        print(_fail(f"{name:<45} got {resp.status_code}, expected {expected_status}  {timing_label}"))

    results.append((name, passed))

    try:
        parsed = resp.json()
    except Exception:
        parsed = {"raw": resp.text[:200]}

    if show_body and parsed:
        compact = json.dumps(parsed, indent=2)
        # Indent and truncate long responses
        lines = compact.split("\n")
        preview = lines[:20]
        if len(lines) > 20:
            preview.append(f"  ... ({len(lines) - 20} more lines)")
        for line in preview:
            print(f"         {line}")

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# TEST PAYLOADS
# ─────────────────────────────────────────────────────────────────────────────

BUYER_PAYLOAD = {
    "full_name": "Marcus Thompson",
    "phone":     "+12145550101",
    "email":     "marcus.thompson@gmail.com",
    "source":    "website",
    "market":    "dallas_fort_worth",
    "message":   (
        "Hi, I'm looking to buy a home in Frisco or Allen. "
        "Budget around $550k. Pre-approved already. "
        "Looking to move in 60 days."
    ),
}

SELLER_PAYLOAD = {
    "full_name": "Sandra Reyes",
    "phone":     "+17135550202",
    "email":     "sreyes@yahoo.com",
    "source":    "website",
    "market":    "houston",
    "message":   (
        "I need to sell my house in Katy. "
        "It's a 4 bed 3 bath, around 2800 sqft. "
        "We're relocating for work and need to close within 90 days."
    ),
}

INVESTOR_PAYLOAD = {
    "full_name": "David Chen",
    "phone":     "+14075550303",
    "email":     "david.chen@investments.com",
    "source":    "website",
    "market":    "orlando",
    "message":   (
        "Looking for multifamily or short-term rental investment properties. "
        "Cash buyer, $1.2M budget. "
        "Need 8% cap rate minimum. Portfolio of 12 units already."
    ),
}

ZILLOW_PAYLOAD = {
    "name":    "Jennifer Walsh",
    "phone":   "+13055550404",
    "email":   "jwalsh@gmail.com",
    "market":  "miami",
    "message": "Interested in the 3BR condo listing in Brickell. Can we schedule a showing?",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TEST SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  CorePilora AI ISA — API Test Suite{RESET}")
    print(f"  Target: {CYAN}{BASE_URL}{RESET}")
    print(f"{'=' * 60}{RESET}")

    # ── Infrastructure ────────────────────────────────────────────────────
    print(_head("[ 1 ] INFRASTRUCTURE"))

    _run(
        name            = "Root alive",
        method          = "GET",
        path            = "/",
        expected_status = 200,
        show_body       = True,
    )

    _run(
        name            = "Basic health",
        method          = "GET",
        path            = "/api/v1/health/",
        expected_status = 200,
        show_body       = True,
    )

    deep = _run(
        name            = "Deep health (DB + Redis + Groq)",
        method          = "GET",
        path            = "/api/v1/health/deep",
        expected_status = 200,
        show_body       = True,
    )

    if deep:
        comps = deep.get("components", {})
        for svc, status in comps.items():
            if status != "healthy":
                print(_warn(f"  {svc} is {status}"))

    # ── Webhook — happy path ──────────────────────────────────────────────
    print(_head("[ 2 ] WEBHOOK — LEAD INTAKE"))

    _run(
        name            = "Buyer lead (DFW, pre-approved, 60d)",
        method          = "POST",
        path            = "/api/v1/webhook/lead?source=website",
        expected_status = 200,
        body            = BUYER_PAYLOAD,
        signed          = True,
        show_body       = True,
    )

    time.sleep(0.5)  # avoid rate-limit on same secret between tests

    _run(
        name            = "Seller lead (Houston, 90d close)",
        method          = "POST",
        path            = "/api/v1/webhook/lead?source=website",
        expected_status = 200,
        body            = SELLER_PAYLOAD,
        signed          = True,
        show_body       = True,
    )

    time.sleep(0.5)

    _run(
        name            = "Investor lead (Orlando, cash, 8% cap)",
        method          = "POST",
        path            = "/api/v1/webhook/lead?source=website",
        expected_status = 200,
        body            = INVESTOR_PAYLOAD,
        signed          = True,
        show_body       = True,
    )

    time.sleep(0.5)

    _run(
        name            = "Zillow normalizer (Miami buyer)",
        method          = "POST",
        path            = "/api/v1/webhook/lead?source=zillow",
        expected_status = 200,
        body            = ZILLOW_PAYLOAD,
        signed          = True,
        show_body       = True,
    )

    # ── Webhook — rejection cases ─────────────────────────────────────────
    print(_head("[ 3 ] WEBHOOK — REJECTION CASES (expected failures)"))

    _run(
        name            = "No signature → 401",
        method          = "POST",
        path            = "/api/v1/webhook/lead?source=website",
        expected_status = 401,
        body            = BUYER_PAYLOAD,
        signed          = False,
        show_body       = False,
    )

    _run(
        name            = "Wrong signature → 401",
        method          = "POST",
        path            = "/api/v1/webhook/lead?source=website",
        expected_status = 401,
        body            = BUYER_PAYLOAD,
        headers         = {"X-Webhook-Signature": "sha256=deadbeef"},
        signed          = False,
        show_body       = False,
    )

    _run(
        name            = "Missing phone → 400 or 422",
        method          = "POST",
        path            = "/api/v1/webhook/lead?source=website",
        expected_status = 400,
        body            = {"full_name": "No Phone", "email": "x@x.com"},
        signed          = True,
        show_body       = False,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    failed = total - passed

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"{GREEN}{BOLD}  ALL {total} TESTS PASSED{RESET}")
    else:
        print(f"{BOLD}  {passed}/{total} passed  |  {RED}{failed} failed{RESET}")
        print(f"\n  Failed tests:")
        for name, ok in results:
            if not ok:
                print(f"    {RED}[X]{RESET} {name}")
    print(f"{'=' * 60}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
