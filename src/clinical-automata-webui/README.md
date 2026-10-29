# Clinical-Automata Web UI (demo)

A browser front-end that lets a clinician experience the `clinical-automata`
Claude Code CLI harness: describe, in plain language, the medical-imaging model
they want, and an autonomous coding agent builds it — no programming required.

This repo contains the **initial skeleton** (framework), with training/GPU stubbed
(pseudo-GPU) and email/questionnaire/persistence deferred.

## Layout

```
backend/            FastAPI app (uv-managed)
  app/
    main.py         routes + websocket + SPA serving
    config.py       env-driven settings (CAWEBUI_*)
    auth.py         shared-passcode login (signed cookie)
    session.py      single-occupancy lock + state machine
    claude_runner.py  bidirectional stream-json runner (real + mock)
    events.py       raw claude JSONL -> frontend events (+ path sanitize)
    gpu.py          pseudo-GPU training simulation (replaceable stub)
    datasets.py     loads ../datasets.yaml (sanitized cards)
    emailer.py      email stub (HTTPS API later)
  test_smoke.py     full state-machine test (mock mode, no API cost)
  test_real_claude.py  real bidirectional integration test (spends a little $)
frontend/           Vite + React + TypeScript three-pane UI
datasets.yaml       dataset showcase manifest (keys match backend prepare.py)
```

The harness itself is the **sibling** repo `../clinical-automata-backend`
(configured via `CAWEBUI_backend_dir`); it is *not* touched by the skeleton.

## Configuration (env vars, prefix `CAWEBUI_`)

Site-specific values (secret key, access code, email, domain, data root) are set in
`../config.env`; see `../config.env.example` and the top-level README. Any setting
below can also be given as an env var or in `backend/.env`.

| var | default | meaning |
|-----|---------|---------|
| `claude_mode` | `mock` | `mock` (canned, no cost) or `real` (spawn `claude`) |
| `gpu_mode` | `pseudo` | `pseudo` (simulated training) — real GPU not wired yet |
| `permission_mode` | `default` | use `bypassPermissions` for the real harness |
| `access_code` | `CHANGE_ME_ACCESS_CODE` | fallback site passcode (per-user codes go in `access_codes.txt`) |
| `secret_key` | `CHANGE_ME_SECRET_KEY` | cookie/download-link signing secret — **must be replaced** |
| `backend_dir` | `../clinical-automata-backend` | harness location |

## Run (dev)

Backend:
```bash
cd backend
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install fastapi "uvicorn[standard]" pydantic pydantic-settings pyyaml itsdangerous python-multipart websockets
CAWEBUI_claude_mode=mock uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend (dev server, proxies /api and /ws to :8000):
```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

Or build the frontend once and let the backend serve it at `/`:
```bash
cd frontend && npm run build     # -> frontend/dist
# then just run uvicorn; it serves the SPA
```

## Tests

```bash
cd backend && source .venv/bin/activate
CAWEBUI_claude_mode=mock python test_smoke.py        # full flow, no API cost
python test_real_claude.py                           # real claude (small cost)
```

## Not yet wired

- Real GPU training (currently pseudo-GPU simulation).
- Email delivery (HTTPS email API) — currently a logging stub.
- Persistence to `runs/` + SQLite; post-experience questionnaire.
- cloudflared ingress (deploy step).
- Auth for the **real** demo should use an Anthropic **API key** (billing),
  not a personal subscription — the smoke test showed the CLI runs under a
  5-hour rate limit with overage disabled.
