"""
agent.py
--------
LangChain agentic memory loop: ChatOpenRouter + event search tools + streaming.
"""

from __future__ import annotations

import json
import os
import sys
from typing import AsyncIterator, Iterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk

from .engine import Engine
from .event_tools import build_event_tools
from .prompts import build_system_prompt


def _log(msg: str) -> None:
    print(f"[memorae:agent] {msg}", file=sys.stderr, flush=True)


def _model():
    from langchain_openrouter import ChatOpenRouter

    model_name = os.environ.get("MEMORAE_LLM_MODEL", "moonshotai/kimi-k2-thinking")
    kwargs: dict = {
        "model": model_name,
        "temperature": 0.35,
        "max_tokens": 4096,
        "max_retries": 2,
        "app_title": "Memorae Memory Agent",
        "app_url": "https://memorae.local",
    }
    if "thinking" in model_name.lower() or "kimi" in model_name.lower():
        kwargs["reasoning"] = {"effort": "medium"}
    return ChatOpenRouter(**kwargs)


class MemoryAgent:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._tools = build_event_tools(engine)
        self._reasoning_buf = ""
        now = engine.now.isoformat().replace("+00:00", "Z")
        system = build_system_prompt(
            owner=engine.store.owner,
            now=now,
            event_count=len(engine.store._visible),
            sources=engine.store.get_sources(),
        )
        self._agent = create_agent(
            model=_model(),
            tools=self._tools,
            system_prompt=system,
        )
        _log(f"ready | owner={engine.store.owner} | events={len(engine.store._visible)} | rag={engine.rag_index is not None}")

    def _extract_text(self, chunk) -> str:
        if chunk is None:
            return ""
        if isinstance(chunk, AIMessageChunk):
            content = chunk.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                return "".join(parts)
        if hasattr(chunk, "content"):
            c = chunk.content
            return c if isinstance(c, str) else ""
        return str(chunk) if chunk else ""

    def _extract_reasoning(self, chunk) -> str:
        if not isinstance(chunk, AIMessageChunk):
            return ""
        blocks = getattr(chunk, "content_blocks", None) or []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                t = block.get("reasoning", "") or block.get("text", "")
                if t:
                    return t
        extra = getattr(chunk, "additional_kwargs", {}) or {}
        rc = extra.get("reasoning_content") or extra.get("reasoning")
        return rc if isinstance(rc, str) else ""

    def _merge_reasoning(self, chunk: str) -> tuple[str, str]:
        """Merge streaming reasoning; return (full_text, delta_only)."""
        if not chunk:
            return self._reasoning_buf, ""
        buf = self._reasoning_buf
        if chunk == buf:
            return buf, ""
        if chunk.startswith(buf):
            delta = chunk[len(buf):]
            self._reasoning_buf = chunk
            return self._reasoning_buf, delta
        if buf.startswith(chunk):
            return buf, ""
        overlap = min(len(buf), len(chunk), 300)
        for i in range(overlap, 0, -1):
            if buf[-i:] == chunk[:i]:
                delta = chunk[i:]
                self._reasoning_buf = buf + delta
                return self._reasoning_buf, delta
        if buf.endswith(chunk) or chunk in buf:
            return buf, ""
        self._reasoning_buf = buf + chunk
        return self._reasoning_buf, chunk

    def _map_stream_event(self, event: dict) -> list[dict]:
        kind = event.get("event", "")
        data = event.get("data", {}) or {}
        name = event.get("name", "")
        out: list[dict] = []

        if kind == "on_chat_model_start":
            _log("model thinking…")
            out.append({"type": "phase", "phase": "thinking", "label": "Thinking…"})

        if kind == "on_chat_model_stream":
            chunk = data.get("chunk")
            reasoning = self._extract_reasoning(chunk)
            text = self._extract_text(chunk)
            if reasoning:
                full, delta = self._merge_reasoning(reasoning)
                if delta:
                    out.append({"type": "reasoning", "text": delta, "full": full})
            if text:
                out.append({"type": "phase", "phase": "writing", "label": "Writing answer…"})
                out.append({"type": "token", "text": text})

        if kind == "on_tool_start":
            inp = data.get("input", {})
            inp_dict = inp if isinstance(inp, dict) else {"raw": str(inp)}
            reason = inp_dict.get("reason", "")
            tool_name = name.split(":")[-1] if ":" in name else name
            _log(f"tool start: {tool_name} | reason: {reason}")
            out.append({"type": "phase", "phase": "tool", "label": f"Searching · {tool_name}"})
            out.append({
                "type": "tool_start",
                "name": tool_name,
                "reason": reason,
                "input": {k: v for k, v in inp_dict.items() if k != "reason"},
            })

        if kind == "on_tool_end":
            output = data.get("output")
            summary = ""
            if output is not None:
                raw = output if isinstance(output, str) else str(output)
                try:
                    parsed = json.loads(raw)
                    summary = (
                        f"matched={parsed.get('total_matched', '?')}, "
                        f"returned={parsed.get('returned', '?')}, "
                        f"hidden={parsed.get('hidden_due_to_limit', 0)}"
                    )
                except (json.JSONDecodeError, TypeError):
                    summary = raw[:300]
            tool_name = name.split(":")[-1] if ":" in name else name
            _log(f"tool end: {tool_name} | {summary}")
            out.append({"type": "tool_end", "name": tool_name, "summary": summary})

        return out

    async def astream_events(self, query: str) -> AsyncIterator[dict]:
        if os.environ.get("MEMORAE_LLM") != "1" or not os.environ.get("OPENROUTER_API_KEY"):
            yield {"type": "error", "text": "LLM not configured. Set MEMORAE_LLM=1 and OPENROUTER_API_KEY."}
            yield {"type": "done"}
            return

        _log(f"query: {query[:120]}{'…' if len(query) > 120 else ''}")
        self._reasoning_buf = ""

        yield {
            "type": "meta",
            "owner": self.engine.store.owner,
            "now": self.engine.now.isoformat().replace("+00:00", "Z"),
            "events": len(self.engine.store._visible),
            "sources": self.engine.store.get_sources(),
            "rag_enabled": self.engine.rag_index is not None,
        }
        yield {"type": "phase", "phase": "started", "label": "Connecting to memory…"}

        try:
            async for event in self._agent.astream_events(
                {"messages": [{"role": "user", "content": query}]},
                version="v2",
            ):
                for mapped in self._map_stream_event(event):
                    yield mapped
        except Exception as e:
            _log(f"error: {e}")
            yield {"type": "error", "text": str(e)}

        _log("done")
        yield {"type": "done"}

    async def achat_stream(self, query: str, budget: int = 2500) -> AsyncIterator[dict]:
        async for ev in self.astream_events(query):
            yield ev

    def chat_stream(self, query: str) -> Iterator[dict]:
        import asyncio
        async def _run():
            async for ev in self.astream_events(query):
                yield ev
        loop = asyncio.new_event_loop()
        try:
            gen = _run()
            while True:
                try:
                    yield loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    def chat(self, query: str, budget: int = 2500) -> dict:
        answer = ""
        tool_calls: list[dict] = []
        for ev in self.chat_stream(query):
            if ev.get("type") == "token":
                answer += ev.get("text", "")
            elif ev.get("type") in ("tool_start", "tool_end"):
                tool_calls.append(ev)
            elif ev.get("type") == "error":
                answer = ev.get("text", answer)
        return {
            "query": query,
            "answer": answer,
            "intent": "agentic",
            "tool_calls": tool_calls,
            "debug": {
                "now": self.engine.now.isoformat().replace("+00:00", "Z"),
                "owner": self.engine.store.owner,
                "mode": "langchain_agent",
            },
        }
