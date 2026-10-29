"""Real-training progress: tail projects/**/run*/run.log.

program.md tells the agent to redirect training output to run.log and NOT tee
it, so during a multi-minute training run the Claude transcript goes silent.
We tail those log files independently and stream them to the browser so the
clinician has live progress to watch.

Emits:
  {"kind": "run_log", "run": "<sanitized run dir>", "lines": [...]}
  {"kind": "run_status", "phase": "metric", "metric": <float>, "raw": "<line>"}
      (best-effort: the metric name/format is agent-generated, so parsing is
       heuristic and never authoritative.)
  {"kind": "usage_manual", "text": "...", "path": "..."}  (for on-page display)
  {"kind": "done", "path": "<DONE.json>", "done": {...}}  (completion signal)

Completion is keyed on DONE.json, the harness's atomic end-of-run marker
(FINALIZE.md) — not on USAGE_MANUAL, which the agent may write before the run is
actually finished.
"""
from __future__ import annotations

import asyncio
import glob
import os
import re
from pathlib import Path
from typing import Awaitable, Callable

from . import deliverables
from .config import Settings
from .events import sanitize

EmitFn = Callable[[dict], Awaitable[None]]

# Heuristic: pull a plausible validation metric off a log line, e.g.
# "val_auc: 0.8342", "AUC = 0.71", "mAP@50 0.514".
_METRIC_RE = re.compile(
    r"(?i)\b(val[_ ]?\w*|auc|map@?\d*|f1|sensitivity|specificity|acc\w*|dice|iou)\b"
    r"[^0-9\-]{0,6}(-?\d+\.\d+)"
)


class RunLogTailer:
    def __init__(self, settings: Settings, emit: EmitFn) -> None:
        self.settings = settings
        self.emit = emit
        self._offsets: dict[str, int] = {}
        self._done_seen: set[str] = set()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        pattern = str(self.settings.projects_dir / "**" / "run*" / "run.log")
        done_pattern = str(self.settings.projects_dir / "**" / "DONE.json")
        # Prime: skip everything already on disk from PREVIOUS sessions. Only
        # content written after we start (i.e. THIS run's logs / THIS run's
        # DONE.json) counts. Without this, a stale run.log or a leftover
        # DONE.json from a prior session would replay / falsely complete.
        for path in glob.glob(pattern, recursive=True):
            try:
                self._offsets[path] = os.path.getsize(path)
            except OSError:
                pass
        self._done_seen = set(glob.glob(done_pattern, recursive=True))
        try:
            while True:
                for path in sorted(glob.glob(pattern, recursive=True)):
                    await self._read_new(path)
                # A NEW DONE.json means the harness finished the whole run
                # (FINALIZE.md) -> completion signal.
                for dpath in glob.glob(done_pattern, recursive=True):
                    if dpath in self._done_seen:
                        continue
                    data = deliverables.load_done(dpath)
                    if data is None:
                        continue  # not fully written / unreadable yet; retry next tick
                    self._done_seen.add(dpath)
                    await self._emit_done(dpath, data)
                await asyncio.sleep(self.settings.run_log_poll_interval)
        except asyncio.CancelledError:
            return

    async def _emit_done(self, path: str, data: dict) -> None:
        # Surface the usage manual for on-page display first (best-effort), so a
        # user still on the page sees it; the email delivers it as a file too.
        arts = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
        man = arts.get("usage_manual")
        if isinstance(man, str) and man:
            mp = Path(man)
            if not mp.is_absolute():
                mp = Path(self.settings.resolved_claude_cwd()) / man
            try:
                text = mp.read_text(encoding="utf-8", errors="replace")
                await self.emit({"kind": "usage_manual", "text": sanitize(text), "path": sanitize(str(mp))})
            except OSError:
                pass
        # The completion event carries internal file paths, so session.emit()
        # captures it and does NOT broadcast it to the browser.
        await self.emit({"kind": "done", "path": path, "done": data})

    async def _read_new(self, path: str) -> None:
        try:
            size = os.path.getsize(path)
        except OSError:
            return
        off = self._offsets.get(path, 0)
        if size <= off:
            self._offsets[path] = min(off, size)  # handle truncation/rotation
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(off)
                chunk = f.read()
                self._offsets[path] = f.tell()
        except OSError:
            return

        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            return
        run_id = sanitize(os.path.relpath(os.path.dirname(path), str(self.settings.projects_dir)))
        # cap how much log we forward per tick to avoid flooding the UI
        tail = lines[-40:]
        await self.emit({"kind": "run_log", "run": run_id, "lines": [sanitize(x) for x in tail]})

        for ln in tail:
            m = _METRIC_RE.search(ln)
            if m:
                try:
                    await self.emit({"kind": "run_status", "phase": "metric",
                                     "metric_name": m.group(1), "metric": float(m.group(2)),
                                     "raw": sanitize(ln)})
                except ValueError:
                    pass
