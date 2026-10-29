"""Dataset showcase manifest loader.

Loads ../datasets.yaml (display metadata, sanitized) and exposes cards to the
frontend. Real filesystem paths live ONLY in prepare.py on the backend repo
and are never sent to the browser. This manifest is the whitelist: the doctor
can only pick from these, and (later) the agent is only pointed at these.

Localization: clinicians recognize disease and organ names in their working
language, so the *clinical* fields carry per-language variants under `i18n`.
The dataset's own name ("SIIM-ACR Pneumothorax") and its license are proper
nouns / legal identifiers and stay verbatim in every language.
"""
from __future__ import annotations

import functools

import yaml

from .config import get_settings

# Fields that are safe to expose to the browser.
# NOTE: no predefined "task_types" — that would prime the clinician (see UI
# feedback). We describe what annotations exist and which label categories they
# cover, and let the clinician decide the task themselves.
_PUBLIC_FIELDS = {
    "key", "display_name", "dataset", "modality", "annotations",
    "label_categories", "num_images", "license", "source_url", "thumbnail",
    "description",
}

# Only these are translated. `dataset`, `license` and `source_url` are not.
_LOCALIZABLE = {"display_name", "modality", "annotations", "label_categories", "description"}


@functools.lru_cache
def _load_raw() -> list[dict]:
    path = get_settings().datasets_manifest
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError("datasets.yaml must be a list")
    return data


def _visible(d: dict) -> bool:
    """An `internal: true` dataset (in-house patient data) is only shown when
    the deployment is explicitly started with CAWEBUI_visibility=internal."""
    return not d.get("internal", False) or get_settings().show_internal


def cards(lang: str = "en") -> list[dict]:
    """Sanitized dataset cards for the frontend (no real paths), localized.

    Missing translations fall back to the English top-level value, so a partly
    translated manifest degrades field-by-field instead of breaking the card.
    Internal datasets are omitted entirely unless visibility=internal.
    """
    out = []
    for d in _load_raw():
        if not _visible(d):
            continue
        card = {k: v for k, v in d.items() if k in _PUBLIC_FIELDS}
        overlay = (d.get("i18n") or {}).get(lang) or {}
        for k, v in overlay.items():
            if k in _LOCALIZABLE:
                card[k] = v
        out.append(card)
    return out


def keys() -> list[str]:
    """Every key in the manifest, internal ones included — this is the list that
    must stay in sync with prepare.py, regardless of what we show the user."""
    return [d["key"] for d in _load_raw()]


def withheld_keys() -> list[str]:
    """Datasets registered in the harness but withheld from THIS deployment."""
    return [d["key"] for d in _load_raw() if not _visible(d)]
