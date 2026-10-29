"""Normalize raw Claude Code CLI stream-json into frontend events.

Raw event types observed from `claude -p --output-format stream-json
--verbose --include-partial-messages` (v2.1.201):
  system(init) | rate_limit_event | stream_event | assistant | user | result

We map these onto a small, stable set of frontend events and sanitize any
real filesystem paths / secrets before they leave the server.
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterable

from .config import get_settings

# --- path / secret sanitization -----------------------------------------
# Rewrites are applied in order. Extend as we learn the real cluster paths.
_SANITIZE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(re.escape(get_settings().data_root.rstrip("/")) + r"[^\s\"']*"), "<dataset>"),
    (re.compile(r"/home/[^\s\"']*/\.claude[^\s\"']*"), "<config>"),
    (re.compile(r"/home/[^\s\"']*/\.ssh[^\s\"']*"), "<config>"),
    (re.compile(r"(?i)sk-[a-z0-9-]{16,}"), "<redacted-key>"),
    (re.compile(re.escape(os.path.expanduser("~")) + r"[^\s\"']*"), "<home>"),
]


def sanitize(text: str | None) -> str | None:
    if not text:
        return text
    out = text
    for pat, repl in _SANITIZE_PATTERNS:
        out = pat.sub(repl, out)
    return out


# The agent renders previews to disk and points at them with plain markdown
# (`![preview](/tmp/pax_preview.png)`). That path means nothing to a browser, so
# we note every image it references and let /api/image serve exactly those.
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)")


def image_refs(text: str | None) -> list[str]:
    """Local image paths the agent linked to in its own markdown."""
    out: list[str] = []
    for m in _MD_IMAGE.finditer(text or ""):
        p = m.group(1)
        # Only paths on this box; http(s):// and data: URIs need no help.
        if p.startswith("/") and not p.startswith("//"):
            out.append(p)
    return out


def _content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def normalize(raw: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield zero or more normalized frontend events for one raw event.

    Frontend event shape: {"kind": <str>, ...payload}. The websocket layer
    adds session_id / seq / ts.
    """
    rtype = raw.get("type")

    if rtype == "system" and raw.get("subtype") == "init":
        yield {
            "kind": "system_status",
            "status": "session_ready",
            "model": raw.get("model"),
            "permission_mode": raw.get("permissionMode"),
            "cli_version": raw.get("claude_code_version"),
            # cwd deliberately sanitized; never expose the real path.
            "cwd": sanitize(raw.get("cwd")),
        }
        return

    if rtype == "rate_limit_event":
        info = raw.get("rate_limit_info", {})
        yield {
            "kind": "rate_limit",
            "status": info.get("status"),
            "rate_limit_type": info.get("rateLimitType"),
            "resets_at": info.get("resetsAt"),
        }
        return

    if rtype == "stream_event":
        event = raw.get("event", {})
        etype = event.get("type")
        if etype == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                yield {"kind": "assistant_text_delta", "text": sanitize(delta.get("text", ""))}
        elif etype == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                yield {"kind": "tool_use_start", "tool": block.get("name"), "tool_use_id": block.get("id")}
        elif etype == "message_stop":
            yield {"kind": "assistant_turn_end"}
        return

    if rtype == "assistant":
        # We always run with --include-partial-messages, so the text (via
        # text_delta) and tool_use (via content_block_start) already arrived
        # through stream_event. Re-emitting the full assistant message here
        # would DUPLICATE every paragraph and every tool call. So drop it.
        #
        # It is still worth reading: a text_delta can split `![x](/p.png)` down
        # the middle, so this whole message is the only place an image path is
        # guaranteed to be intact. Register them (session.emit keeps these out
        # of the transcript) so /api/image knows what it is allowed to serve.
        for block in _content_blocks(raw.get("message", {})):
            if block.get("type") != "text":
                continue
            for path in image_refs(block.get("text")):
                yield {"kind": "agent_image", "path": path, "src": sanitize(path)}
        return

    if rtype == "user":
        for block in _content_blocks(raw.get("message", {})):
            if block.get("type") == "tool_result":
                yield {
                    "kind": "tool_result",
                    "tool_use_id": block.get("tool_use_id"),
                    "content": sanitize(_short_text(block.get("content"))),
                }
        return

    if rtype == "result":
        yield {
            "kind": "final_result",
            "subtype": raw.get("subtype"),
            "is_error": raw.get("is_error", False),
            "text": sanitize(raw.get("result")),
            "duration_ms": raw.get("duration_ms"),
        }
        return

    # Unknown top-level type -> not user-facing; drop it (raw stream is
    # separately logged to claude_events.jsonl in the real design, §11).
    return


def _short_json(obj: Any, limit: int = 2000) -> str:
    import json

    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "…"


def _short_text(content: Any, limit: int = 4000) -> str:
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                # A Read of a PNG comes back as an image block with no "text",
                # which would otherwise render as an empty result box.
                parts.append("[image]" if c.get("type") == "image" else c.get("text", ""))
            else:
                parts.append(str(c))
        s = "\n".join(parts)
    else:
        s = str(content)
    return s if len(s) <= limit else s[:limit] + "…"
