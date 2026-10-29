"""Turn a harness DONE.json into completion-email deliverables.

The harness writes `projects/YYYYMMDD_HH/DONE.json` at the very end of a run
(FINALIZE.md). It lists, by path, the artifacts to hand back to the clinician.
This module resolves those paths, decides what can ride as an email attachment
vs. what must be a signed download link, zips product folders, and returns the
keyword arguments for `emailer.send_success`.

Delivery policy (per the demo's decisions) — the delivered set is a FIXED
whitelist of DONE.json artifact keys, never "everything in artifacts":
  * DELIVERED: the usage manual, the test-predictions CSV, and the doctor-format
    outputs. Nothing else is added on our own initiative.
  * NEVER delivered: the checkpoint (large, not clinician-facing) and results.tsv
    (the internal experiment/selection log — provenance, not a deliverable).
  * a directory artifact is zipped first.
  * a file (or a zip) up to `attach_max_bytes` rides as an attachment; anything
    larger — or once the running total passes `total_attach_budget` — becomes a
    signed 7-day download link (downloads.make_url).
  * links require `public_url`; without it an oversized file is skipped with a
    log line rather than breaking the email.

Never raises: a missing or unreadable artifact is skipped, not fatal — a
completion email must go out even if some deliverable could not be prepared.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path

from . import downloads
from .config import Settings

log = logging.getLogger("cawebui.deliver")

# FIXED whitelist of single-file artifact keys we hand back, in display order.
# `checkpoint` (large) and `results_tsv` (internal experiment/selection log) are
# deliberately NOT delivered — do not add keys here without a clinical reason.
_SINGLE_KEYS = ("usage_manual", "test_predictions_csv")


def load_done(path: str | Path) -> dict | None:
    """Parse a DONE.json, or None if absent / not yet fully written / invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def find_done(settings: Settings, newer_than: float = 0.0) -> tuple[str, dict] | None:
    """Locate a DONE.json under projects_dir written after `newer_than`
    (defends against a leftover marker from a previous session). Newest wins.
    Returns (path, parsed) or None."""
    pattern = str(settings.projects_dir / "**" / "DONE.json")
    paths = sorted(glob.glob(pattern, recursive=True), key=_safe_mtime, reverse=True)
    for p in paths:
        if _safe_mtime(p) < newer_than:
            continue
        data = load_done(p)
        if data is not None:
            return p, data
    return None


def build(settings: Settings, done_path: str, done: dict) -> dict:
    """Produce kwargs for `emailer.send_success` from a parsed DONE.json:
    {attachments, links, primary_metric, status}."""
    status = str(done.get("status") or "success")
    pm = done.get("primary_metric")
    primary_metric = pm if isinstance(pm, dict) else None

    # DONE.json records artifact paths relative to the harness repo root
    # (resolved_claude_cwd), e.g. "projects/YYYYMMDD_HH/USAGE_MANUAL.txt".
    base = Path(settings.resolved_claude_cwd())
    delivery_dir = (
        Path(done_path).resolve().parent / "_delivery" if done_path else base / "_delivery"
    )
    artifacts = done.get("artifacts") if isinstance(done.get("artifacts"), dict) else {}

    # Candidate source paths, in delivery order.
    candidates: list[Path] = []
    for key in _SINGLE_KEYS:
        candidates += _as_paths(artifacts.get(key), base)
    candidates += _as_paths(artifacts.get("doctor_format_outputs"), base)

    attachments: list[str] = []
    links: list[tuple[str, str, int]] = []   # (filename, url, size_bytes)
    spent = 0
    for src in _dedup(candidates):
        prepared = _prepare(src, delivery_dir)   # file as-is, or a zip of a dir
        if prepared is None:
            continue
        size = _safe_size(prepared)
        if size <= settings.attach_max_bytes and spent + size <= settings.total_attach_budget:
            attachments.append(str(prepared))
            spent += size
        else:
            url = _link_for(prepared, delivery_dir, settings)
            if url:
                links.append((prepared.name, url, size))
            else:
                log.warning("[deliver] %s too large to attach and no public_url; skipped", prepared.name)

    return {"attachments": attachments, "links": links, "primary_metric": primary_metric, "status": status}


# --- helpers -------------------------------------------------------------
def _as_paths(value, base: Path) -> list[Path]:
    """Normalize an artifact value (str or list of str) to existing Paths,
    resolving relative paths against the harness root."""
    if not value:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    out: list[Path] = []
    for v in items:
        if not isinstance(v, str) or not v:
            continue
        p = Path(v)
        if not p.is_absolute():
            p = base / v
        out.append(p)
    return out


def _dedup(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def _prepare(src: Path, delivery_dir: Path) -> Path | None:
    """A file is returned as-is; a directory is zipped into `delivery_dir`."""
    try:
        if src.is_dir():
            return _zip_dir(src, delivery_dir)
        if src.is_file():
            return src
    except OSError:
        pass
    return None


def _zip_dir(src: Path, delivery_dir: Path) -> Path | None:
    try:
        delivery_dir.mkdir(parents=True, exist_ok=True)
        zpath = delivery_dir / (src.name + ".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(src):
                for fn in files:
                    fp = Path(root) / fn
                    zf.write(fp, fp.relative_to(src.parent))
        return zpath
    except OSError as e:
        log.warning("[deliver] zip of %s failed: %s", src, e)
        return None


def _link_for(path: Path, delivery_dir: Path, settings: Settings) -> str | None:
    """A signed download URL for a file. Download tokens are only honoured for
    paths under projects_dir, so a file outside it is first staged into the
    (linkable) `_delivery` dir. Requires public_url."""
    if not settings.public_url:
        return None
    projects = settings.projects_dir.resolve()
    p = path.resolve()
    if projects not in p.parents and p != projects:
        try:
            delivery_dir.mkdir(parents=True, exist_ok=True)
            dst = (delivery_dir / p.name).resolve()
            shutil.copy2(p, dst)
            p = dst
        except OSError as e:
            log.warning("[deliver] could not stage %s for download: %s", path, e)
            return None
    return downloads.make_url(p, settings.public_url)


def _safe_size(p: str | Path) -> int:
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def _safe_mtime(p: str | Path) -> float:
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0
