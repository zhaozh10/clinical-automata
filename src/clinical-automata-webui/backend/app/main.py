"""FastAPI app: login, datasets, session lifecycle, and the websocket that
carries the bidirectional Claude Code CLI stream."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, datasets, downloads
from .config import get_settings
from .session import State, manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("cawebui")

app = FastAPI(title="Clinical-Automata Web UI (demo)")

# CORS for the Vite dev server (cookies need explicit origins + credentials).
_dev_origins = os.environ.get("CAWEBUI_dev_origins", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _dev_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- models --------------------------------------------------------------
class LoginBody(BaseModel):
    code: str
    email: str = ""
    lang: str = "en"  # UI language; also decides the language of the emails


class UserMessage(BaseModel):
    type: str  # "user_message" | "task_submit"
    text: str


# --- auth ----------------------------------------------------------------
@app.post("/api/login")
def login(body: LoginBody, response: Response):
    code = body.code.strip()
    if not auth.check_access_code(code):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad access code")
    email = body.email.strip()
    # The address is the only way to reach a clinician once they close the tab,
    # so a session must never start without one: an interrupted run would end in
    # silence for the one person who needs to hear about it. The frontend gates
    # on this too, but a stale client or a direct POST would otherwise slip past.
    if "@" not in email:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="a valid email is required"
        )
    s = get_settings()
    response.set_cookie(
        key=s.cookie_name,
        value=auth.issue_token(code, email, body.lang.strip() or "en"),
        max_age=s.cookie_max_age, httponly=True, samesite="lax",
    )
    return {"ok": True}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(get_settings().cookie_name)
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    code = auth.request_code(request)
    return {"authenticated": code is not None, "code": code}


# --- content -------------------------------------------------------------
# (the welcome screen is UI chrome and lives in the frontend i18n dictionary)
@app.get("/api/datasets")
def get_datasets(request: Request, _: str = Depends(auth.require_http)):
    # Clinical fields follow the language chosen at login; the dataset's own
    # name and license stay verbatim.
    return {"datasets": datasets.cards(auth.request_lang(request))}


@app.get("/api/session/status")
def session_status(_: None = Depends(auth.require_http)):
    return {"occupied": manager.occupied}


# --- session lifecycle ---------------------------------------------------
@app.post("/api/session/start")
async def session_start(request: Request, code: str = Depends(auth.require_http)):
    email = auth.request_email(request)
    ok, result = await manager.start_session(code, email, auth.request_lang(request))
    if not ok:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="occupied")
    sess = manager.get(result)
    return {"session_id": result, "state": sess.state.value if sess else "active"}


@app.get("/api/image")
def agent_image(src: str, _: str = Depends(auth.require_http)):
    """Serve an image the agent linked to from its own markdown.

    Only paths the agent actually showed are servable. The agent shares /tmp
    with everyone else on the node, so taking any path here would turn this into
    a read-anything endpoint for the whole box.
    """
    sess = manager.current
    real = sess.agent_images.get(src) if sess else None
    if not real:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown image")
    path = Path(real)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="image no longer on disk")
    return FileResponse(path)


@app.get("/api/session/mine")
def session_mine(code: str = Depends(auth.require_http)):
    """The session owned by this code, if any (identity-based rejoin)."""
    sess = manager.mine(code)
    if not sess:
        return {"active": False}
    return {
        "active": True,
        "session_id": sess.id,
        "state": sess.state.value,
        "task_submitted": sess.task_submitted,
        "task_text": sess.task_text,
    }


@app.get("/api/session/{session_id}/resume")
def session_resume(session_id: str, _: None = Depends(auth.require_http)):
    """Is this (client-remembered) session still the live one? Lets a client
    that closed the page rejoin its own running session instead of being told
    'occupied' by its own stale lock."""
    sess = manager.get(session_id)
    if not sess:
        return {"active": False}
    return {
        "active": True,
        "state": sess.state.value,
        "task_submitted": sess.task_submitted,
        "task_text": sess.task_text,
    }


@app.post("/api/session/{session_id}/complete")
async def session_complete(session_id: str, _: None = Depends(auth.require_http)):
    sess = manager.get(session_id)
    if not sess:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such session")
    await manager.release(session_id, reason="user_complete")
    return {"ok": True}


# --- websocket -----------------------------------------------------------
@app.websocket("/ws/{session_id}")
async def ws_endpoint(ws: WebSocket, session_id: str):
    if not auth.ws_code(ws):
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    sess = manager.get(session_id)
    if not sess:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await ws.accept()
    sess.sockets.add(ws)
    # Replay everything so far so a (re)connecting client is fully in sync
    # (initial turn_state/system events are broadcast before the WS connects).
    await sess.replay_to(ws)
    try:
        while True:
            data = await ws.receive_json()
            try:
                msg = UserMessage(**data)
            except Exception:
                continue
            await sess.handle_user_message(msg.text, is_task_submission=(msg.type == "task_submit"))
    except WebSocketDisconnect:
        sess.sockets.discard(ws)
    except Exception as e:  # pragma: no cover
        log.warning("ws error: %s", e)
        sess.sockets.discard(ws)


# --- health --------------------------------------------------------------
@app.on_event("shutdown")
async def _on_shutdown():
    """SIGTERM (slurm allocation expiry, pkill, redeploy) lands here before the
    process disappears. Without this the agent tree is hard-killed: a train.py
    can be orphaned on the GPU and the clinician never learns their run died.
    """
    await manager.shutdown()


@app.get("/api/health")
def health():
    s = get_settings()
    return {"ok": True, "claude_mode": s.claude_mode, "gpu_mode": s.gpu_mode}


@app.get("/api/download/{token}")
def download(token: str):
    """Serve a deliverable via a signed, path-constrained, time-limited link.
    The token itself is the authorization — no login needed."""
    try:
        path = downloads.resolve_token(token)
    except ValueError as e:
        codes = {"expired": 410, "invalid": 403, "forbidden": 403, "missing": 404}
        raise HTTPException(status_code=codes.get(str(e), 403), detail=str(e))
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


# --- static frontend (built) --------------------------------------------
# Cache policy (prevents the "blank page after redeploy" trap):
#   * /assets/index-<hash>.js|css  -> immutable, cache forever. The hash is in
#     the filename, so a new build has a new URL; the old file is never wanted.
#   * index.html                   -> no-cache, always revalidate. It names the
#     current bundle, so a stale copy would point at a deleted hash (404 -> blank
#     page). Revalidating on every load means a plain refresh always self-heals.
_ASSET_CACHE = "public, max-age=31536000, immutable"
_HTML_CACHE = "no-cache, no-store, must-revalidate"


@app.middleware("http")
async def _static_cache_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/assets/"):
        response.headers["Cache-Control"] = _ASSET_CACHE
    elif not path.startswith("/api/"):
        # the SPA HTML shell (and any root file that falls through to it)
        response.headers["Cache-Control"] = _HTML_CACHE
    return response


_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidate = _dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
