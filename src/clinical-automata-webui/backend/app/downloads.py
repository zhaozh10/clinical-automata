"""Signed, time-limited download links for large deliverables.

When a deliverable (e.g. a zipped image folder) is too big to attach to the
completion email, we instead serve it over the existing Cloudflare tunnel via a
token-signed link:  https://<demo-host>/api/download/<token>

The token is an itsdangerous-signed blob carrying the file's absolute path and
is only accepted for paths under `projects_dir` — so a valid token cannot be
crafted (no secret) and a leaked one cannot traverse to arbitrary files.
"""
from __future__ import annotations

from pathlib import Path

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .config import get_settings

# Link lifetime: a deliverable should stay fetchable for a whole allocation week.
DOWNLOAD_MAX_AGE = 7 * 24 * 3600  # 7 days, in seconds


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_settings().secret_key, salt="cawebui-download")


def make_token(path: str | Path) -> str:
    """Sign an absolute file path into a download token."""
    return _serializer().dumps({"path": str(Path(path).resolve())})


def make_url(path: str | Path, base_url: str) -> str:
    """Full download URL for a file, e.g. base_url='https://demo.example.com'."""
    return f"{base_url.rstrip('/')}/api/download/{make_token(path)}"


def resolve_token(token: str) -> Path:
    """Validate a token and return the file Path, or raise ValueError with a
    reason ('expired' | 'invalid' | 'forbidden' | 'missing')."""
    try:
        data = _serializer().loads(token, max_age=DOWNLOAD_MAX_AGE)
    except SignatureExpired:
        raise ValueError("expired")
    except BadSignature:
        raise ValueError("invalid")
    p = Path(data.get("path", "")).resolve()
    allowed = get_settings().projects_dir.resolve()
    # constrain to the projects tree (defence in depth against path traversal)
    if allowed not in p.parents and p != allowed:
        raise ValueError("forbidden")
    if not p.is_file():
        raise ValueError("missing")
    return p
