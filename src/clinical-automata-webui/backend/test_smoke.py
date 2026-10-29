"""End-to-end skeleton test (mock claude mode, no API cost).

Run: CAWEBUI_claude_mode=mock python test_smoke.py
Exercises: login/auth, datasets, single-occupancy lock, onboarding ->
ACTIVE, websocket chat, one-shot task submit -> pseudo-GPU run -> completion.
"""
import os
import tempfile

os.environ.setdefault("CAWEBUI_claude_mode", "mock")
# Self-contained two-code file so we can test identity-based rejoin vs occupied.
_codes = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
# Mirrors the real file: a header comment, and codes annotated with who they are
# for. The annotation must NOT become part of the code (it locked out invited
# users once) — codeB exercises that.
_codes.write("# access codes — one per invited user\ncodeA\ncodeB  # japan\n")
_codes.close()
os.environ["CAWEBUI_access_codes_file"] = _codes.name

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def drain_until(ws, kinds, limit=400):
    """Read events until one of `kinds` is seen (or limit). Returns list."""
    seen = []
    for _ in range(limit):
        evt = ws.receive_json()
        seen.append(evt)
        if evt.get("kind") in kinds:
            return seen
    raise AssertionError(f"did not see any of {kinds}; got kinds={[e.get('kind') for e in seen]}")


def main():
    c = TestClient(app)

    # health + unauthenticated guard
    assert c.get("/api/health").json()["claude_mode"] == "mock"
    assert c.get("/api/datasets").status_code == 401, "datasets must require auth"

    # bad + good login (code = identity; email + UI language captured here now)
    assert c.post("/api/login", json={"code": "wrong", "email": "x@y.com"}).status_code == 401
    assert (
        c.post("/api/login", json={"code": "codeA", "email": "doc@example.com", "lang": "ja"}).status_code
        == 200
    )
    assert c.get("/api/me").json()["authenticated"] is True
    assert c.get("/api/me").json()["code"] == "codeA"

    # datasets — default visibility is public, so the in-house one is withheld
    cards = c.get("/api/datasets").json()["datasets"]
    keys = [d["key"] for d in cards]
    # the exact public list is pinned in test_datasets.py; here we only check that
    # nothing withheld reaches an external user
    for withheld in ("pelvic ct (fracture)",):
        assert withheld not in keys, f"withheld dataset '{withheld}' leaked to an external user!"
    assert all("num_images" in d or d["key"].endswith("segmentation)") for d in cards)
    assert all("path" not in d for d in cards), "real paths must not leak to frontend"
    print(f"[ok] datasets (public): {keys}")

    # lock + identity: codeA starts a session -> activates immediately (no onboarding step)
    r = c.post("/api/session/start")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert r.json()["state"] == "active", "start should activate directly now"
    # the language chosen at login reaches the session (it decides the email language)
    from app.main import manager  # noqa: PLC0415

    assert manager.get(sid).onboarding.language == "ja", "UI language not carried into the session"
    # same code re-starting -> REJOINS its own session (same id), not occupied
    r2 = c.post("/api/session/start")
    assert r2.status_code == 200 and r2.json()["session_id"] == sid, "same code should rejoin"
    assert c.get("/api/session/mine").json()["session_id"] == sid
    # a DIFFERENT code -> occupied
    c2 = TestClient(app)
    c2.post("/api/login", json={"code": "codeB", "email": "b@example.com", "lang": "de"})
    assert c2.post("/api/session/start").status_code == 409, "different code must be occupied"
    assert c2.get("/api/session/mine").json()["active"] is False
    print(f"[ok] start -> active directly; same code rejoins, different code -> 409 occupied")

    # websocket: chat, then one-shot task submit
    with c.websocket_connect(f"/ws/{sid}") as ws:
        # on connect, replay must catch the client up incl. the initial
        # turn_state=awaiting_user broadcast before the WS existed (bug fix).
        replay = drain_until(ws, {"turn_state"})
        assert any(e.get("kind") == "turn_state" and e.get("state") == "awaiting_user" for e in replay), \
            "client never received awaiting_user -> chat input would be stuck disabled"
        assert any(e.get("kind") == "session_state" and e.get("state") == "active" for e in replay)
        print("[ok] ws replay delivers active + awaiting_user on connect")

        # plain chat
        ws.send_json({"type": "user_message", "text": "Hi, what can you do?"})
        seen = drain_until(ws, {"final_result"})
        assert any(e.get("kind") == "assistant_text_delta" for e in seen), "no streamed chat reply"
        print("[ok] chat streamed reply")

        # task submit -> autonomous pseudo-GPU run -> completion
        ws.send_json({"type": "task_submit", "text": "Build a pneumothorax classifier robust to chest drains."})
        seen = drain_until(ws, {"session_state"}, limit=800)
        kinds = [e.get("kind") for e in seen]
        assert "task_submitted" in kinds
        assert any(k == "run_status" for k in kinds), "no pseudo-GPU run_status events"
        assert any(e.get("kind") == "usage_manual" for e in seen), "no usage manual"
        completed = [e for e in seen if e.get("kind") == "session_state" and e.get("state") == "completed"]
        assert completed, f"session did not complete; kinds={kinds}"
        print(f"[ok] task submit -> {sum(k=='run_status' for k in kinds)} run_status events -> completed")

    # after success: still occupied (16h grace), until explicit complete
    assert c.get("/api/session/status").json()["occupied"] is True
    assert c.post(f"/api/session/{sid}/complete").status_code == 200
    assert c.get("/api/session/status").json()["occupied"] is False
    print("[ok] complete -> lock released")

    # "End session" also logs out, so the browser falls back to the access-code
    # page and the next visitor is not mistaken for this one.
    assert c.post("/api/logout").status_code == 200
    assert c.get("/api/me").json()["authenticated"] is False
    assert c.get("/api/datasets").status_code == 401, "logout must drop the identity cookie"
    print("[ok] logout -> identity cleared, back to the access-code page")

    # new session can now be acquired (fresh login)
    assert c.post("/api/login", json={"code": "codeA", "email": "doc@example.com", "lang": "en"}).status_code == 200
    assert c.post("/api/session/start").status_code == 200
    print("[ok] lock reusable after release")
    print("\nALL SKELETON CHECKS PASSED")


if __name__ == "__main__":
    main()
