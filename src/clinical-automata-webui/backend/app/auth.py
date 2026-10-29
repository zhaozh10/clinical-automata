"""Site access + identity: a per-user code (from access_codes.txt) is entered
on every visit and carried, signed, in the cookie. The code IS the identity —
the same code rejoins the same session across browsers/devices."""
from __future__ import annotations

from fastapi import HTTPException, Request, WebSocket, status
from itsdangerous import BadSignature, URLSafeTimedSerializer

from .config import get_settings


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_settings().secret_key, salt="cawebui-auth")


def issue_token(code: str, email: str = "", lang: str = "en") -> str:
    return _serializer().dumps({"code": code, "email": email, "lang": lang})


def token_identity(token: str | None) -> dict | None:
    """Return the {code, email, lang} identity in a valid token, else None."""
    if not token:
        return None
    try:
        data = _serializer().loads(token, max_age=get_settings().cookie_max_age)
        return data if isinstance(data, dict) else None
    except BadSignature:
        return None
    except Exception:
        return None


def token_code(token: str | None) -> str | None:
    d = token_identity(token)
    return d.get("code") if d else None


def check_access_code(code: str) -> bool:
    return code in get_settings().valid_codes()


def request_code(request: Request) -> str | None:
    return token_code(request.cookies.get(get_settings().cookie_name))


def request_email(request: Request) -> str:
    d = token_identity(request.cookies.get(get_settings().cookie_name))
    return d.get("email", "") if d else ""


def request_lang(request: Request) -> str:
    """UI language chosen at login; the user-facing emails follow it."""
    d = token_identity(request.cookies.get(get_settings().cookie_name))
    return (d.get("lang") or "en") if d else "en"


def require_http(request: Request) -> str:
    """FastAPI dependency: returns the caller's identity code, or 401."""
    code = request_code(request)
    if not code:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not authenticated")
    return code


def ws_code(ws: WebSocket) -> str | None:
    token = ws.cookies.get(get_settings().cookie_name) or ws.query_params.get("token")
    return token_code(token)
