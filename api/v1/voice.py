"""
api/v1/voice.py
CorePilora AI — Real-Time Voice Pipeline

Twilio Media Streams WebSocket → Deepgram STT → Groq LLM → ElevenLabs TTS → Twilio

Endpoints:
  POST /voice/incoming  — Twilio inbound call webhook → TwiML with WebSocket URL
  WS   /voice/stream    — Full-duplex voice bridge
  POST /voice/status    — Twilio call status callback → Redis cleanup
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Optional

import httpx
import websockets
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from config.redis import get_redis
from config.settings import settings
from services.cache import get_lead_cache
from services.conversation import create_session, complete_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_DG_WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?encoding=mulaw&sample_rate=8000&model=nova-2-phonecall"
    "&endpointing=300&utterance_end_ms=1000&interim_results=false"
)

_ULAW_BYTES_PER_SEC = 8000  # 8kHz × 1 byte/sample
_TTL_CALL_META      = 3600  # 1 hour — covers full call duration


# ─────────────────────────────────────────────────────────────────────────────
# REDIS KEY BUILDERS — voice-specific
# ─────────────────────────────────────────────────────────────────────────────

def _key_call_phone(call_sid: str) -> str:
    return f"call:{call_sid}:phone"


def _key_phone_index(phone: str) -> str:
    return f"phone:index:{phone}"


# ─────────────────────────────────────────────────────────────────────────────
# LEAD CONTEXT LOADER
# Zero DB reads during live call — all from Redis.
# ─────────────────────────────────────────────────────────────────────────────

async def _get_lead_context(phone: str) -> dict[str, Any]:
    """
    Look up lead by phone number via phone:index:{phone} Redis key.
    Returns full lead context if found, minimal fallback dict if unknown caller.
    """
    try:
        redis   = await get_redis()
        lead_id = await redis.get(_key_phone_index(phone))

        if not lead_id:
            logger.info("VOICE | Unknown caller | phone=%s", phone)
            return {"phone": phone, "full_name": "there", "lead_type": "buyer"}

        lead_data = await get_lead_cache(lead_id)
        if not lead_data:
            return {
                "phone":     phone,
                "full_name": "there",
                "lead_type": "buyer",
                "lead_id":   lead_id,
            }

        logger.info(
            "VOICE | Lead loaded | id=%s | name=%s | type=%s",
            lead_id, lead_data.get("full_name"), lead_data.get("lead_type"),
        )
        return lead_data

    except Exception as e:
        logger.error("VOICE | _get_lead_context FAILED | phone=%s | error=%s", phone, str(e))
        return {"phone": phone, "full_name": "there", "lead_type": "buyer"}


# ─────────────────────────────────────────────────────────────────────────────
# DEEPGRAM LISTENER — runs as a concurrent asyncio.Task
# Puts final transcripts into queue. Filters output while Jaiyana is speaking.
# ─────────────────────────────────────────────────────────────────────────────

async def _deepgram_listener(
    dg_ws,
    q:           asyncio.Queue,
    is_speaking: list[bool],
) -> None:
    """
    Read Deepgram transcript events and push final transcripts to queue.
    is_speaking[0] == True means Jaiyana's audio is playing — skip transcript
    to prevent Deepgram from picking up Jaiyana's own voice through the line.
    """
    try:
        async for raw in dg_ws:
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            if data.get("type") != "Results":
                continue
            if not data.get("is_final", False):
                continue

            alts = data.get("channel", {}).get("alternatives", [])
            if not alts:
                continue

            text = alts[0].get("transcript", "").strip()
            if not text:
                continue

            if is_speaking[0]:
                logger.debug("VOICE | Transcript filtered (speaking) | text=%.40s", text)
                continue

            logger.info("VOICE | Transcript | text=%s", text)
            await q.put(text)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("VOICE | _deepgram_listener error | %s", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# TTS — ELEVENLABS
# ulaw_8000 output is Twilio-native mulaw. Zero conversion overhead.
# ─────────────────────────────────────────────────────────────────────────────

async def _synthesize(text: str) -> Optional[bytes]:
    """
    TTS with cache-first strategy:
      0. Redis audio cache  — instant, zero API cost
      1. ElevenLabs (primary) — eleven_turbo_v2_5, ulaw_8000
      2. Deepgram Aura (fallback) — aura-asteria-en, mulaw 8000Hz
    Returns raw mulaw bytes for direct Telnyx delivery. None if all fail.
    """
    # ── Cache check first ─────────────────────────────────────────────────
    try:
        from services.audio_cache import get_audio_for_response, store_audio
        cached = await get_audio_for_response(text)
        if cached:
            logger.info("VOICE | TTS cache HIT | chars=%s", len(text))
            return cached
    except Exception:
        pass

    # ── Primary: ElevenLabs ───────────────────────────────────────────────
    el_url = (
        f"https://api.elevenlabs.io/v1/text-to-speech"
        f"/{settings.ELEVENLABS_VOICE_ID}"
        f"/stream?output_format=ulaw_8000&optimize_streaming_latency=3"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                el_url,
                headers={
                    "xi-api-key":   settings.ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text":           text,
                    "model_id":       "eleven_turbo_v2_5",
                    "voice_settings": {
                        "stability":         0.4,
                        "similarity_boost":  0.85,
                        "style":             0.3,
                        "use_speaker_boost": True,
                    },
                },
            )
            r.raise_for_status()
            logger.info("VOICE | TTS via ElevenLabs | chars=%s", len(text))
            audio = r.content
            try:
                from services.audio_cache import store_audio
                await store_audio(text, audio)
            except Exception:
                pass
            return audio
    except Exception as e:
        logger.warning("VOICE | ElevenLabs TTS failed — trying Deepgram | error=%s", str(e))

    # ── Fallback: Deepgram Aura TTS ───────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                "https://api.deepgram.com/v1/speak"
                "?model=aura-asteria-en&encoding=mulaw&sample_rate=8000&container=none",
                headers={
                    "Authorization": f"Token {settings.DEEPGRAM_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={"text": text},
            )
            r.raise_for_status()
            logger.info("VOICE | TTS via Deepgram fallback | chars=%s", len(text))
            return r.content
    except Exception as e:
        logger.error("VOICE | Both TTS providers failed | text=%.40s | error=%s", text, str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO DELIVERY TO TWILIO
# ─────────────────────────────────────────────────────────────────────────────

async def _send_audio(ws: WebSocket, stream_sid: str, audio: bytes) -> bool:
    """Send audio to Telnyx. Returns True on success, False on failure."""
    try:
        await ws.send_text(json.dumps({
            "event":     "media",
            "streamSid": stream_sid,
            "media":     {"payload": base64.b64encode(audio).decode()},
        }))
        logger.info("VOICE | Audio sent | sid=%s | bytes=%s", stream_sid, len(audio))
        return True
    except Exception as e:
        logger.error("VOICE | _send_audio FAILED | sid=%s | error=%s", stream_sid, str(e))
        return False


async def _clear_audio(ws: WebSocket, stream_sid: str) -> None:
    try:
        await ws.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# JAIYANA TURN — single Groq call, voice-optimized
# ─────────────────────────────────────────────────────────────────────────────

async def _jaiyana_turn(
    messages:  list[dict],
    context:   dict,
    lead_type: str,
) -> str:
    """
    Generate Jaiyana's next voice response via Groq.
    max_tokens=120 keeps responses short enough for phone speech.
    """
    from agents.persona import build_system_prompt
    from services.groq_client import call_groq

    system_prompt = build_system_prompt(
        lead_type  = lead_type,
        market     = context.get("market", "Dallas-Fort Worth"),
        lead_name  = context.get("full_name", "there"),
        lead_source = context.get("source", "inbound_call"),
    )

    voice_directive = (
        "\n\n## VOICE CALL MANDATE — OVERRIDES EVERYTHING\n"
        "You are on a LIVE PHONE CALL right now. Non-negotiable rules:\n"
        "1. Two sentences maximum. Hard stop after the second.\n"
        "2. Zero bullet points. Zero numbered lists. Zero asterisks.\n"
        "3. End with exactly one question. Never end with a statement.\n"
        "4. Speak naturally — as you would say it out loud, not as you would write it.\n"
    )

    try:
        return await call_groq(
            system_prompt = system_prompt + voice_directive,
            messages      = messages,
            temperature   = 0.7,
            max_tokens    = 120,
        )
    except Exception as e:
        logger.error("VOICE | _jaiyana_turn FAILED | error=%s", str(e))
        return "Give me just one second — are you still there?"


# ─────────────────────────────────────────────────────────────────────────────
# OPENING LINE — context-aware first words
# ─────────────────────────────────────────────────────────────────────────────

def _build_opening(lead_context: dict) -> str:
    """Build Jaiyana's first words tailored to lead type and name."""
    raw_name   = (lead_context.get("full_name") or "").strip()
    first_name = raw_name.split()[0] if raw_name and raw_name.lower() != "there" else ""
    lead_type  = lead_context.get("lead_type", "buyer")

    address = f" {first_name}" if first_name else ""

    if lead_type == "seller":
        topic = "your home and what you're looking to do next"
    elif lead_type == "investor":
        topic = "the investment opportunities you were looking into"
    else:
        topic = "the properties you were checking out in the area"

    return (
        f"Hey{address}, this is Jaiyana with CorePilora — "
        f"I'm following up about {topic}. "
        f"What's driving your interest in making a move right now?"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE CONVERSATION TURN — handles one transcript end-to-end
# Recursive for barge-in (max depth 10 to prevent stack overflow).
# ─────────────────────────────────────────────────────────────────────────────

async def _handle_transcript(
    transcript:  str,
    ws:          WebSocket,
    stream_sid:  str,
    messages:    list[dict],
    lead_context: dict,
    lead_type:   str,
    is_speaking: list[bool],
    q:           asyncio.Queue,
    _depth:      int = 0,
) -> None:
    """
    Process one lead transcript:
      1. Append to message history
      2. Call Groq for Jaiyana's response
      3. Synthesize via ElevenLabs
      4. Send audio to Twilio
      5. Wait for audio to finish OR lead to interrupt (barge-in)
      6. If interrupted: clear audio, recurse with interrupt transcript
    """
    if _depth > 10:
        return

    logger.info("VOICE TURN | depth=%s | lead_said=%s", _depth, transcript)
    messages.append({"role": "user", "content": transcript})

    response_text = await _jaiyana_turn(messages, lead_context, lead_type)
    messages.append({"role": "assistant", "content": response_text})
    logger.info("VOICE TURN | jaiyana=%s", response_text[:80])

    audio = await _synthesize(response_text)
    if not audio:
        logger.warning("VOICE TURN | TTS returned no audio | depth=%s", _depth)
        return

    is_speaking[0] = True
    await _send_audio(ws, stream_sid or "default", audio)

    audio_duration = len(audio) / _ULAW_BYTES_PER_SEC + 0.5
    try:
        interrupt = await asyncio.wait_for(q.get(), timeout=audio_duration)
        # Lead spoke before audio finished — barge-in
        is_speaking[0] = False
        await _clear_audio(ws, stream_sid)
        await _handle_transcript(
            interrupt, ws, stream_sid, messages,
            lead_context, lead_type, is_speaking, q,
            _depth=_depth + 1,
        )
    except asyncio.TimeoutError:
        # Audio finished normally
        is_speaking[0] = False


# ─────────────────────────────────────────────────────────────────────────────
# SESSION PERSISTENCE — fire-and-forget after call ends
# ─────────────────────────────────────────────────────────────────────────────

async def _persist_voice_session(
    lead_id:  Optional[str],
    messages: list[dict],
    call_sid: str,
) -> None:
    if not lead_id:
        return
    try:
        sid = await create_session(
            lead_id     = lead_id,
            channel     = "voice",
            pipeline    = "voice_inbound",
            market      = "",
            lead_source = "inbound_call",
            lead_type   = "voice",
        )
        if sid:
            await complete_session(
                session_id      = sid,
                outcome         = "voice_completed",
                messages        = messages,
                appointment_set = False,
                objection_count = 0,
                twilio_call_sid = call_sid,
            )
            logger.info(
                "VOICE | Session persisted | lead=%s | call=%s | turns=%s",
                lead_id, call_sid,
                len([m for m in messages if m["role"] == "user"]),
            )
    except Exception as e:
        logger.error(
            "VOICE | _persist_voice_session FAILED | lead=%s | error=%s",
            lead_id, str(e),
        )


# ─────────────────────────────────────────────────────────────────────────────
# TWIML BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _twiml_connect_stream(call_sid: str, lead_name: str = "") -> str:
    ws_url = f"wss://{settings.APP_DOMAIN}/api/v1/voice/stream?call_sid={call_sid}"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{ws_url}" />'
        "</Connect>"
        "</Response>"
    )


def _twiml_reject() -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response><Reject /></Response>'


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT — POST /voice/incoming
# ─────────────────────────────────────────────────────────────────────────────

def _validate_telnyx_signature(request: Request) -> bool:
    # Always allow — signature validation can be enforced later via TELNYX_PUBLIC_KEY
    return True


@router.post("/incoming")
async def voice_incoming(request: Request) -> Response:
    """
    Telnyx TeXML inbound call webhook.
    Logs all form fields for diagnosis.
    Returns TeXML instructing Telnyx to open a Media Stream WebSocket.
    """
    try:
        form      = await request.form()
        form_data = dict(form)

        # Log every field Telnyx sends — critical for diagnosing field-name issues
        logger.info("VOICE INCOMING | raw_form=%s", json.dumps(form_data))

        # Telnyx TeXML sends CallSid (same as Twilio); fall back to their own IDs
        call_sid = (
            str(form_data.get("CallSid") or
                form_data.get("callSid") or
                form_data.get("call_sid") or
                form_data.get("telnyx_call_control_id") or
                "")
        )
        from_number = str(form_data.get("From") or form_data.get("from") or "")

        if not _validate_telnyx_signature(request):
            logger.warning("VOICE INCOMING | Signature validation failed — rejecting")
            return Response(_twiml_reject(), media_type="application/xml", status_code=403)

        logger.info("VOICE INCOMING | sid=%s | from=%s", call_sid, from_number)

        # Do NOT reject on missing call_sid — just continue with empty sid so the
        # stream still connects; we log it above so we can see what Telnyx sent.
        if not call_sid:
            logger.warning("VOICE INCOMING | CallSid missing in form — proceeding without sid")

        lead_name = ""
        if from_number and call_sid:
            try:
                redis = await get_redis()
                await redis.setex(_key_call_phone(call_sid), _TTL_CALL_META, from_number)
                lead_id = await redis.get(_key_phone_index(from_number))
                if lead_id:
                    from services.cache import get_lead_cache
                    ctx = await get_lead_cache(lead_id)
                    lead_name = (ctx or {}).get("full_name", "")
            except Exception as _redis_err:
                logger.warning("VOICE INCOMING | Redis unavailable — continuing | error=%s", str(_redis_err))

        texml = _twiml_connect_stream(call_sid, lead_name)
        logger.info("VOICE INCOMING | texml_response=%s", texml)
        return Response(texml, media_type="application/xml")

    except Exception as e:
        logger.error("VOICE INCOMING ERROR | error=%s", str(e))
        return Response(_twiml_reject(), media_type="application/xml")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT — WebSocket /voice/stream
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/stream")
async def voice_stream(ws: WebSocket, call_sid: str = "") -> None:
    """
    Full-duplex voice bridge.

    Concurrency model:
      _twilio_loop()         — tight receive loop: reads Twilio WS, forwards audio to Deepgram
      _deepgram_listener()   — asyncio.Task: reads Deepgram, pushes transcripts to queue
      _conversation_loop()   — asyncio.Task: reads queue, calls Groq, sends TTS to Twilio

    _twilio_loop awaited directly. _conversation_loop and _deepgram_listener
    run as Tasks and are cancelled when _twilio_loop finishes.
    """
    await ws.accept()

    # Shared mutable state (list containers allow mutation across nested scopes)
    stream_sid:  list[str]  = [""]
    is_speaking: list[bool] = [False]
    messages:    list[dict] = []
    q:           asyncio.Queue = asyncio.Queue()
    stream_ready               = asyncio.Event()
    lead_context: dict         = {}
    dg_ws                      = None
    dg_task                    = None
    conv_task                  = None

    try:
        # ── Resolve caller → lead context ─────────────────────────────────
        phone = ""
        if call_sid:
            try:
                redis = await get_redis()
                phone = (await redis.get(_key_call_phone(call_sid))) or ""
            except Exception:
                pass

        lead_context = (
            await _get_lead_context(phone) if phone
            else {"full_name": "there", "lead_type": "buyer"}
        )
        lead_id   = lead_context.get("lead_id")
        lead_type = lead_context.get("lead_type", "buyer")

        logger.info(
            "VOICE STREAM | sid=%s | phone=%s | lead=%s | type=%s",
            call_sid, phone, lead_id, lead_type,
        )

        # ── Connect Deepgram STT ──────────────────────────────────────────
        try:
            dg_ws = await websockets.connect(
                _DG_WS_URL,
                additional_headers={"Authorization": f"Token {settings.DEEPGRAM_API_KEY}"},
                ping_interval=20,
                ping_timeout=10,
            )
            dg_task = asyncio.create_task(
                _deepgram_listener(dg_ws, q, is_speaking)
            )
            logger.info("VOICE | Deepgram STT connected | sid=%s", call_sid)
        except Exception as e:
            logger.error("VOICE | Deepgram STT connect FAILED | sid=%s | error=%s", call_sid, str(e))
            dg_ws  = None
            dg_task = None

        # ── Twilio receive loop — runs in the main coroutine ──────────────
        async def _twilio_loop() -> None:
            async for raw in ws.iter_text():
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                event = msg.get("event")

                if event == "connected":
                    # Telnyx sends "connected" before "start" — just log it
                    logger.info("VOICE | Stream connected event | raw=%s", json.dumps(msg)[:300])

                elif event == "start":
                    start_data = msg.get("start", {})
                    # Telnyx TeXML streams: try every known SID field name
                    stream_sid[0] = (
                        start_data.get("streamSid") or
                        start_data.get("stream_id") or
                        start_data.get("streamId") or
                        start_data.get("callSid") or
                        msg.get("streamSid") or
                        msg.get("stream_id") or
                        msg.get("callSid") or
                        "default"
                    )
                    logger.info(
                        "VOICE | Stream started | sid=%s | raw=%s",
                        stream_sid[0], json.dumps(msg)[:500],
                    )
                    stream_ready.set()

                elif event == "media":
                    chunk = msg.get("media", {}).get("payload", "")
                    if chunk and dg_ws:
                        try:
                            await dg_ws.send(base64.b64decode(chunk))
                        except Exception:
                            pass

                elif event == "stop":
                    logger.info("VOICE | Stream stopped | sid=%s", stream_sid[0])
                    break

        # ── Conversation loop — concurrent with _twilio_loop ──────────────
        async def _conversation_loop() -> None:
            await stream_ready.wait()
            # Give Telnyx 300ms to fully initialize the stream before sending audio
            await asyncio.sleep(0.3)
            sid = stream_sid[0]
            logger.info("VOICE | Stream ready — sending opening | sid=%s", sid)

            opening = _build_opening(lead_context)
            messages.append({"role": "assistant", "content": opening})

            # Synthesize and deliver opening greeting via ElevenLabs/Deepgram
            try:
                logger.info("VOICE | Synthesizing opening | text=%.60s", opening)
                audio = await _synthesize(opening)
                if audio:
                    sid = stream_sid[0]
                    logger.info("VOICE | Delivering opening audio | sid=%s | bytes=%s", sid, len(audio))
                    is_speaking[0] = True
                    sent = await _send_audio(ws, sid, audio)
                    if not sent:
                        logger.error("VOICE | Opening audio delivery FAILED")
                    audio_duration = len(audio) / _ULAW_BYTES_PER_SEC + 0.5
                    await asyncio.sleep(audio_duration)
                    is_speaking[0] = False
                    logger.info("VOICE | Opening delivered | chars=%s | duration=%.1fs", len(opening), audio_duration)
                else:
                    logger.error("VOICE | Opening TTS returned no audio — caller will hear silence")
            except Exception as e:
                logger.error("VOICE | Opening synthesis failed | %s", str(e))
                is_speaking[0] = False

            while True:
                try:
                    transcript = await asyncio.wait_for(q.get(), timeout=90.0)
                except asyncio.TimeoutError:
                    logger.info("VOICE | Idle 90s — ending conversation | sid=%s", sid)
                    break
                except asyncio.CancelledError:
                    break

                await _handle_transcript(
                    transcript, ws, sid, messages,
                    lead_context, lead_type, is_speaking, q,
                )

        # ── Run both loops concurrently ────────────────────────────────────
        conv_task = asyncio.create_task(_conversation_loop())

        try:
            await _twilio_loop()
        finally:
            if conv_task and not conv_task.done():
                conv_task.cancel()
            try:
                await conv_task
            except (asyncio.CancelledError, Exception):
                pass

    except WebSocketDisconnect:
        logger.info("VOICE | WebSocket disconnected | sid=%s", call_sid)
    except Exception as e:
        logger.error("VOICE STREAM ERROR | sid=%s | error=%s", call_sid, str(e))
    finally:
        # Cancel Deepgram listener
        if dg_task and not dg_task.done():
            dg_task.cancel()
        try:
            if dg_task:
                await dg_task
        except (asyncio.CancelledError, Exception):
            pass

        # Close Deepgram WebSocket
        if dg_ws:
            try:
                await dg_ws.close()
            except Exception:
                pass

        # Persist voice session (fire-and-forget — never blocks WebSocket close)
        if messages:
            asyncio.create_task(
                _persist_voice_session(
                    lead_id  = lead_context.get("lead_id"),
                    messages = messages,
                    call_sid = call_sid,
                )
            )

        logger.info(
            "VOICE STREAM CLOSED | sid=%s | turns=%s",
            call_sid,
            len([m for m in messages if m["role"] == "user"]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT — GET /voice/test  (diagnostic only — remove before go-live)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/test")
async def voice_test() -> dict:
    """Returns the TeXML that would be sent for a test call — use to verify endpoint is up."""
    texml = _twiml_connect_stream("TEST_SID")
    return {
        "status": "ok",
        "webhook_url": f"https://{settings.APP_DOMAIN}/api/v1/voice/incoming",
        "stream_url":  f"wss://{settings.APP_DOMAIN}/api/v1/voice/stream",
        "texml":       texml,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT — POST /voice/status
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/status")
async def voice_status(request: Request) -> Response:
    """
    Twilio call status callback (completed, no-answer, busy, failed).
    Cleans up Redis call metadata.
    """
    try:
        form        = await request.form()
        call_sid    = str(form.get("CallSid",     ""))
        call_status = str(form.get("CallStatus",  ""))
        duration    = str(form.get("CallDuration", "0"))

        logger.info(
            "VOICE STATUS | sid=%s | status=%s | duration=%ss",
            call_sid, call_status, duration,
        )

        if call_sid:
            try:
                redis = await get_redis()
                await redis.delete(_key_call_phone(call_sid))
            except Exception:
                pass

    except Exception as e:
        logger.error("VOICE STATUS ERROR | error=%s", str(e))

    return Response(content="", status_code=204)
