"""
services/groq_client.py
CorePilora AI — Shared Groq LLM Client

Single source of truth for all Groq API calls.
All three agent graphs (buyer, seller, investor) import from here.
Model is driven by settings.GROQ_MODEL — change once, applies everywhere.

Context budget for llama3-70b-8192 (8192 token limit):
  System prompt  ≈ 4500 tokens
  max_tokens     ≈  300 tokens
  Available msgs ≈ 3400 tokens  →  cap at last 8 messages (~250 tokens each)
"""

from __future__ import annotations

import logging

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
MAX_CONTEXT_MSGS   = 8   # last 4 full turns — stays within 8192 token budget


def _trim_messages(messages: list[dict]) -> list[dict]:
    """Keep only the last MAX_CONTEXT_MSGS messages to prevent context overflow."""
    return messages[-MAX_CONTEXT_MSGS:] if len(messages) > MAX_CONTEXT_MSGS else messages


async def call_groq(
    system_prompt: str,
    messages:      list[dict],
    temperature:   float = 0.7,
    max_tokens:    int   = 300,
) -> str:
    """
    Send a chat completion request to Groq.

    Args:
        system_prompt: Jaiyana's full persona + context prompt.
        messages:      Conversation history [{role, content}].
        temperature:   LLM temperature (0.1 = extraction, 0.8 = creative opening).
        max_tokens:    Max tokens in response.

    Returns:
        LLM response string. Raises httpx.HTTPStatusError on API failure.
    """
    payload = {
        "model": settings.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            *_trim_messages(messages),
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json=payload,
        )
        response.raise_for_status()

    data    = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    logger.debug("GROQ | tokens=%s", data.get("usage", {}).get("total_tokens"))
    return content
