"""LLM-authored *additions* to the completion email — not a rewrite.

The email keeps its fixed template (opening line, intro, grace, session id). The
one thing the template can't fill well is describing each delivered file, so a
short-lived, read-only `claude -p` opens the project folder (projects/YYYYMMDD_HH/),
reads the context (DONE.json, USAGE_MANUAL.txt, logs), and returns a friendly
label + one-line description per file — so an attachment reads
"test_predictions.csv — per-sample predictions and Dice" and a link reads
"Teaching figures (frontal + lateral), 6 MB — Download".

This is the Claude Code CLI (same subscription auth as the main run), NOT the
Messages API: the point is to hand the model a *directory* and let its built-in
Read/Glob/Grep tools pull the context itself.

Best-effort: any failure returns None and the email renders from the template
alone (plain filenames).
"""
from __future__ import annotations

import asyncio
import json
import logging

from . import datasets
from .claude_runner import _agent_env, _confidentiality_prompt
from .config import Settings
from .events import sanitize

log = logging.getLogger("cawebui.refine")

_LANG_NAME = {"en": "English", "de": "German", "ja": "Japanese", "zh": "Chinese (Simplified)"}

_MAX_DESC = 400   # one line per file (German compounds run long); never cut mid-word


def _clip(t: str, maxlen: int) -> str:
    t = t.strip()
    if len(t) <= maxlen:
        return t
    cut = t.rfind(" ", 0, maxlen)   # back off to a word boundary, never mid-word
    return (t[:cut] if cut > 0 else t[:maxlen]).rstrip()


def _prompt(lang: str, filenames: list[str]) -> str:
    language = _LANG_NAME.get(lang, "English")
    files = "\n".join(f"  - {n}" for n in filenames) or "  (none)"
    return (
        "A clinician's autonomous model-building session has just finished. Your "
        "current working directory is that session's output folder. Read what you "
        "need to understand each deliverable — DONE.json, USAGE_MANUAL.txt, and the "
        "logs.\n\n"
        "You are filling in one small part of a completion email whose wording is "
        "otherwise fixed: a short label and a one-sentence description for each "
        "delivered file. Return ONLY a JSON object (no prose, no code fences, no "
        "explanation) with exactly this shape:\n\n"
        "{\n"
        '  "<exact filename>": {"label": "<short human-friendly name>", '
        '"description": "<exactly one sentence: what this file is / what it is for>"}\n'
        "}\n\n"
        "Describe exactly these files, using each filename verbatim as the key:\n"
        f"{files}\n\n"
        f"Write the label and description text in {language}. Each description is "
        "exactly one sentence. Describe what each file contains and what it is for; "
        "do NOT state or guess which language a document is written in. This is a "
        "research artifact, not a clinical decision-making device. Output the JSON only."
    )


def _extract_json(raw: str) -> dict | None:
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    t = t.strip()
    try:
        obj = json.loads(t)
    except ValueError:
        # tolerate leading/trailing prose around the object
        a, b = t.find("{"), t.rfind("}")
        if a < 0 or b <= a:
            return None
        try:
            obj = json.loads(t[a : b + 1])
        except ValueError:
            return None
    return obj if isinstance(obj, dict) else None


def _norm(data: dict, filenames: list[str]) -> dict:
    """Sanitize + clamp the model's JSON into {filename: {label?, description?}}.
    Tolerates a {"files": {...}} wrapper as well as a flat object."""
    raw_files = data.get("files") if isinstance(data.get("files"), dict) else data
    allowed = set(filenames)
    files: dict[str, dict] = {}
    for name, info in raw_files.items():
        if name not in allowed or not isinstance(info, dict):
            continue
        label = info.get("label")
        desc = info.get("description")
        entry = {}
        if isinstance(label, str) and label.strip():
            entry["label"] = _clip(sanitize(label.strip()), _MAX_DESC)
        if isinstance(desc, str) and desc.strip():
            entry["description"] = _clip(sanitize(desc.strip()), _MAX_DESC)
        if entry:
            files[name] = entry
    return files


async def describe_deliverables(
    settings: Settings, project_dir: str, lang: str, filenames: list[str]
) -> dict | None:
    """Run a read-only `claude -p` in `project_dir` and return
    {filename: {"label", "description"}}, or None if disabled / times out /
    produces nothing parseable."""
    if not settings.email_llm_refine or not project_dir:
        return None
    cmd = [
        settings.claude_bin, "-p", _prompt(lang, filenames),
        "--output-format", "text",
        "--permission-mode", "bypassPermissions",  # avoid a headless permission-prompt hang
    ]
    withheld = datasets.withheld_keys()
    if withheld:
        cmd += ["--append-system-prompt", _confidentiality_prompt(withheld)]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=project_dir, env=_agent_env(),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        log.warning("[refine] could not spawn claude: %s", e)
        return None
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=settings.email_refine_timeout)
    except asyncio.TimeoutError:
        log.warning("[refine] timed out after %ss; falling back to template", settings.email_refine_timeout)
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        return None
    if proc.returncode != 0:
        log.warning("[refine] claude exited %s: %s", proc.returncode, (err or b"").decode("utf-8", "replace")[:300])
        return None
    data = _extract_json((out or b"").decode("utf-8", "replace"))
    if data is None:
        log.warning("[refine] could not parse JSON from claude output")
        return None
    files = _norm(data, filenames)
    return files or None
