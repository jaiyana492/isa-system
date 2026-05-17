"""
test_live_call.py
CorePilora AI ISA — Real Call Test

Submits a lead with YOUR phone number.
The system classifies it, saves it, then triggers
a real Twilio outbound call to your phone.
Jaiyana will speak via ElevenLabs TTS.

Usage:
  python test_live_call.py +1XXXXXXXXXX
  python test_live_call.py +1XXXXXXXXXX "I want to buy a home in Frisco, budget 500k"

Requirements before running:
  1. Server running:  uvicorn main:app --reload
  2. ngrok running:   ngrok http --domain=dipped-reawake-capably.ngrok-free.dev 8000
  3. Twilio console:  phone number voice URL set to
                      https://dipped-reawake-capably.ngrok-free.dev/api/v1/voice/incoming
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL       = "http://localhost:8000"
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

if not WEBHOOK_SECRET:
    print(f"{RED}ERROR: WEBHOOK_SECRET not in .env{RESET}")
    sys.exit(1)

if len(sys.argv) < 2:
    print(f"{RED}Usage: python test_live_call.py +1XXXXXXXXXX [message]{RESET}")
    sys.exit(1)

YOUR_PHONE = sys.argv[1].strip()
MESSAGE    = sys.argv[2] if len(sys.argv) > 2 else (
    "Hi, I'm looking to buy a home in Frisco or Allen. "
    "My budget is around 550k. I'm pre-approved and want to move within 60 days."
)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNATURE
# ─────────────────────────────────────────────────────────────────────────────

def _sign(body: bytes) -> str:
    mac = hmac.new(
        key       = WEBHOOK_SECRET.encode("utf-8"),
        msg       = body,
        digestmod = hashlib.sha256,
    )
    return f"sha256={mac.hexdigest()}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  CorePilora AI ISA — Live Call Test{RESET}")
    print(f"  Phone : {CYAN}{YOUR_PHONE}{RESET}")
    print(f"  Server: {CYAN}{BASE_URL}{RESET}")
    print(f"{'=' * 60}{RESET}\n")

    # ── Step 1: confirm server is alive ──────────────────────────────────
    print(f"{BOLD}[1] Checking server...{RESET}")
    try:
        r = httpx.get(f"{BASE_URL}/", timeout=5.0)
        data = r.json()
        print(f"    {GREEN}OK{RESET} — {data.get('isa_name')} {data.get('version')} {data.get('status')}")
    except Exception as e:
        print(f"    {RED}FAIL — server not responding: {e}{RESET}")
        print(f"    Start with: .\\venv\\Scripts\\uvicorn.exe main:app --reload")
        sys.exit(1)

    # ── Step 2: deep health ───────────────────────────────────────────────
    print(f"\n{BOLD}[2] Checking all services...{RESET}")
    try:
        r = httpx.get(f"{BASE_URL}/api/v1/health/deep", timeout=15.0)
        components = r.json().get("components", {})
        all_ok = True
        for svc, status in components.items():
            icon = GREEN + "OK  " + RESET if status == "healthy" else RED + "FAIL" + RESET
            print(f"    {icon} {svc}: {status}")
            if status != "healthy":
                all_ok = False
        if not all_ok:
            print(f"\n    {YELLOW}Warning: some services degraded — call may still work{RESET}")
    except Exception as e:
        print(f"    {YELLOW}Health check error: {e}{RESET}")

    # ── Step 3: submit lead with real phone ───────────────────────────────
    print(f"\n{BOLD}[3] Submitting lead with your number...{RESET}")

    payload = {
        "full_name": "Live Test",
        "phone":     YOUR_PHONE,
        "email":     "livetest@corepilora.ai",
        "source":    "website",
        "market":    "dallas_fort_worth",
        "message":   MESSAGE,
    }

    body    = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sig     = _sign(body)
    headers = {
        "Content-Type":       "application/json",
        "X-Webhook-Signature": sig,
    }

    t0 = time.perf_counter()
    try:
        r = httpx.post(
            f"{BASE_URL}/api/v1/webhook/lead?source=website",
            content = body,
            headers = headers,
            timeout = 60.0,
        )
        ms   = round((time.perf_counter() - t0) * 1000)
        data = r.json()

        if r.status_code == 200:
            print(f"    {GREEN}PASS{RESET} — HTTP 200 in {ms}ms")
            print(f"    Lead ID  : {CYAN}{data.get('lead_id')}{RESET}")
            print(f"    Lead type: {data.get('classification', {}).get('lead_type') if data.get('classification') else 'N/A'}")
            print(f"    Pipeline : {data.get('pipeline_result', {}).get('pipeline') if data.get('pipeline_result') else 'N/A'}")
            print(f"    Status   : {data.get('status')}")
        else:
            print(f"    {RED}FAIL{RESET} — HTTP {r.status_code} in {ms}ms")
            print(f"    Response : {json.dumps(data, indent=2)[:400]}")
            sys.exit(1)

    except Exception as e:
        print(f"    {RED}FAIL — {e}{RESET}")
        sys.exit(1)

    # ── Step 4: inform ────────────────────────────────────────────────────
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{GREEN}{BOLD}  Lead submitted. Twilio is calling {YOUR_PHONE} now.{RESET}")
    print(f"  Pick up — Jaiyana will speak within 3-5 seconds.")
    print(f"\n  If no call arrives within 30 seconds:")
    print(f"  {YELLOW}  1. Check ngrok is running{RESET}")
    print(f"  {YELLOW}  2. Check Twilio console — phone number voice URL must be:{RESET}")
    print(f"  {CYAN}     https://dipped-reawake-capably.ngrok-free.dev/api/v1/voice/incoming{RESET}")
    print(f"  {YELLOW}  3. Check server logs: tail -f server_err.log{RESET}")
    print(f"{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
