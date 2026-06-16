"""
llm_client.py
---------------
Reliable OpenRouter client via httpx (fixes IncompleteRead from urllib).
Supports streaming and non-streaming chat completions.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator, Iterator

import httpx

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)


def _headers() -> dict[str, str]:
    return {
        "content-type": "application/json",
        "authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}",
        "http-referer": "https://memorae.local",
        "x-title": "Memorae Memory Agent",
    }


def _model() -> str:
    return os.environ.get("MEMORAE_LLM_MODEL", "moonshotai/kimi-k2-thinking")


def chat_complete(
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> str:
    body = {
        "model": _model(),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        for attempt in range(3):
            try:
                resp = client.post(CHAT_URL, headers=_headers(), json=body)
                resp.raise_for_status()
                data = resp.json()
                msg = (data.get("choices") or [{}])[0].get("message", {})
                return (msg.get("content") or "").strip()
            except (httpx.ReadError, httpx.RemoteProtocolError):
                if attempt == 2:
                    raise
    return ""


def chat_stream(
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> Iterator[str]:
    body = {
        "model": _model(),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "messages": messages,
    }
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        with client.stream("POST", CHAT_URL, headers=_headers(), json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                text = delta.get("content") or ""
                if text:
                    yield text


async def achat_stream(
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> AsyncIterator[str]:
    body = {
        "model": _model(),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "messages": messages,
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        async with client.stream("POST", CHAT_URL, headers=_headers(), json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                text = delta.get("content") or ""
                if text:
                    yield text
