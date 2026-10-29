"""Session lifecycle + single-occupancy lock + state machine.

States (one active session at a time):
    PREVIEW (no session held)
    ONBOARDING  -> collecting age/experience/language/email, claude not spawned
    ACTIVE      -> claude spawned; chat + (once) task submission -> autonomous run
    COMPLETED   -> success + email + 16h grace countdown
    INTERRUPTED -> abnormal exit + email + lock released immediately

The lock is a single global slot. `start_session` fails with "occupied" if a
session is already held.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from . import deliverables, email_refine, emailer
from .claude_runner import BaseRunner, make_runner
from .config import Settings, get_settings
from .runlog import RunLogTailer

log = logging.getLogger("cawebui.session")


class State(str, Enum):
    ONBOARDING = "onboarding"
    ACTIVE = "active"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


@dataclass
class Onboarding:
    age: str = ""
    experience: str = ""
    language: str = ""
    email: str = ""


@dataclass
class Session:
    id: str
    settings: Settings
    owner_code: str = ""  # the access code that owns this session (identity)
    state: State = State.ONBOARDING
    onboarding: Onboarding = field(default_factory=Onboarding)
    created_at: float = field(default_factory=time.time)
    autonomous: bool = False
    task_submitted: bool = False
    # The task, kept verbatim so a client that reloads, rejoins from another
    # device still sees what it asked for.
    task_text: str = ""
    # Image paths the agent linked to in its own markdown -> where they live.
    # /api/image will serve these and nothing else.
    agent_images: dict[str, str] = field(default_factory=dict)
    saw_usage_manual: bool = False
    usage_manual_text: str = ""  # last usage manual text, for on-page display
    # Completion is keyed on DONE.json (the harness's end-of-run marker), not on
    # the usage manual (which the agent may write before the run has finished).
    saw_done: bool = False
    done: dict | None = None      # parsed DONE.json (artifacts, metric, status)
    done_path: str = ""           # its path on disk (for resolving artifacts)
    seq: int = 0
    turn: str = "agent_busy"             # last broadcast turn state (until runner ready)
    pending_tools: int = 0               # tool_use started but not yet resulted
    last_activity: float = field(default_factory=time.time)
    runner: BaseRunner | None = None
    sockets: set[WebSocket] = field(default_factory=set)
    history: list[dict] = field(default_factory=list)
    tailer: RunLogTailer | None = None
    _grace_task: asyncio.Task | None = None
    _watchdog_task: asyncio.Task | None = None
    _manager: "SessionManager | None" = None

    HISTORY_CAP = 5000

    # -- websocket fan-out ------------------------------------------------
    async def _broadcast(self, event: dict[str, Any]) -> None:
        self.seq += 1
        payload = {"session_id": self.id, "seq": self.seq, "ts": time.time(), **event}
        self.history.append(payload)
        if len(self.history) > self.HISTORY_CAP:
            del self.history[: len(self.history) - self.HISTORY_CAP]
        dead = []
        for ws in self.sockets:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.sockets.discard(ws)

    async def replay_to(self, ws: WebSocket) -> None:
        """Catch a (re)connecting client up on everything so far
        (the user may leave and come back via the completion email)."""
        for payload in list(self.history):
            await ws.send_json(payload)

    async def _set_turn(self, state: str) -> None:
        if self.turn == state:
            return
        self.turn = state
        await self._broadcast({"kind": "turn_state", "state": state})

    async def emit(self, event: dict[str, Any]) -> None:
        """Runner -> here. Fan out, then apply terminal-state transitions."""
        kind = event.get("kind")
        if kind == "agent_image":
            # Registry only, never broadcast: the client already has the path
            # from the agent's markdown. This just records that the agent chose
            # to show it, which is what makes it safe to serve.
            self.agent_images[event["src"]] = event["path"]
            return
        if kind == "done":
            # Harness completion marker (DONE.json). It carries internal file
            # paths, so it is captured here and NOT broadcast to the browser
            # (the browser learns of completion via session_state instead).
            self.saw_done = True
            self.done = event.get("done")
            self.done_path = event.get("path", "")
            if self.autonomous and self.state == State.ACTIVE:
                await self._to_completed()
            return
        self.last_activity = time.time()
        if kind == "tool_use_start":
            self.pending_tools += 1  # a tool is in flight -> agent genuinely busy
        elif kind == "tool_result":
            self.pending_tools = max(0, self.pending_tools - 1)
        if kind == "usage_manual":
            self.saw_usage_manual = True
            # Keep the manual itself: the completion email ships it to the user,
            # who may never come back to the page before the grace period ends.
            self.usage_manual_text = event.get("text", "") or ""
        await self._broadcast(event)

        if kind == "final_result":
            # The whole agent turn finished -> user's turn. (Primary signal.)
            await self._set_turn("awaiting_user")
        elif kind == "process_exit":
            # Process died. Completion is DONE.json, not a clean exit: the
            # harness must never leave the project without one (FINALIZE.md), so
            # rc==0 with no DONE.json is a premature exit, i.e. an interruption.
            if self.autonomous and self.state == State.ACTIVE:
                rc = event.get("returncode")
                if rc == 0 and not self.saw_done:
                    # The tailer polls on an interval and may not have seen the
                    # marker yet; check disk directly before giving up on it.
                    found = await asyncio.to_thread(
                        deliverables.find_done, self.settings, self.created_at
                    )
                    if found:
                        self.done_path, self.done = found
                        self.saw_done = True
                if rc == 0 and self.saw_done:
                    await self._to_completed()
                else:
                    await self._to_interrupted(f"agent process exited (rc={rc})")

    async def _idle_watchdog(self) -> None:
        """Safety net for the `result` signal. If the agent has gone silent
        after speaking — no tool in flight, no events for idle_unlock_seconds —
        the turn is effectively over (e.g. `result` stalled on a rate limit),
        so re-enable input. Normal mid-turn gaps (message -> next tool) are only
        1-2s, so this never fires mid-turn; a real tool run keeps pending_tools>0.
        """
        try:
            while True:
                await asyncio.sleep(3)
                if self.state not in (State.ACTIVE, State.COMPLETED):
                    continue
                if self.turn == "agent_busy" and self.pending_tools <= 0:
                    idle = time.time() - self.last_activity
                    if idle > self.settings.idle_unlock_seconds:
                        log.info("watchdog: agent idle %.0fs after speaking; re-enabling input (%s)", idle, self.id)
                        await self._set_turn("awaiting_user")
        except asyncio.CancelledError:
            return

    # -- state transitions ------------------------------------------------
    async def _to_completed(self) -> None:
        self.state = State.COMPLETED
        await self._broadcast({"kind": "session_state", "state": self.state.value})
        if self.onboarding.email:
            # Resolve the deliverables named in DONE.json into attachments +
            # download links (zips product folders; skips the checkpoint). Off
            # the event loop: it touches disk and may zip large folders.
            kwargs = {"attachments": [], "links": [], "primary_metric": None, "status": "success"}
            if self.done is not None:
                try:
                    kwargs = await asyncio.to_thread(
                        deliverables.build, self.settings, self.done_path, self.done
                    )
                except Exception as e:  # a bad artifact must not sink the email
                    log.warning("deliverable build failed for %s: %s", self.id, e)
            # Let a short-lived, read-only claude open the project folder and write
            # a one-line description per delivered file. Best-effort: None -> the
            # template renders with plain filenames.
            descriptions = None
            if self.done_path:
                project_dir = str(Path(self.done_path).resolve().parent)
                names = [Path(a).name for a in kwargs.get("attachments", [])]
                names += [fn for fn, _url, _sz in kwargs.get("links", [])]
                try:
                    descriptions = await email_refine.describe_deliverables(
                        self.settings, project_dir, self.onboarding.language, names
                    )
                except Exception as e:  # refining must never block the email
                    log.warning("deliverable describe failed for %s: %s", self.id, e)
            await emailer.send_success(
                self.onboarding.email,
                self.id,
                self.onboarding.language,
                grace_hours=round(self.settings.grace_seconds_after_success / 3600),
                descriptions=descriptions,
                **kwargs,
            )
        # 16h grace countdown, then auto-release.
        self._grace_task = asyncio.create_task(self._grace_then_release())

    async def _grace_then_release(self) -> None:
        try:
            await asyncio.sleep(self.settings.grace_seconds_after_success)
        except asyncio.CancelledError:
            return
        log.info("grace expired for session %s; releasing lock", self.id)
        if self._manager:
            await self._manager.release(self.id, reason="grace_timeout")

    async def _to_interrupted(self, reason: str = "unknown") -> None:
        if self.state == State.INTERRUPTED:
            return  # idempotent: shutdown() and the runner's process_exit can race
        self.state = State.INTERRUPTED
        log.info("session %s interrupted (%s)", self.id, reason)
        await self._broadcast({"kind": "session_state", "state": self.state.value})
        if self.onboarding.email:
            await emailer.send_interruption(self.onboarding.email, self.id, self.onboarding.language)
        # The owner always hears about an interruption, and why — otherwise a
        # slurm expiry silently eats a clinician's run with nobody the wiser.
        await emailer.send_owner_interruption(
            self.onboarding.email, self.owner_code, self.id, reason, self.task_submitted
        )
        if self._manager:
            await self._manager.release(self.id, reason="interrupted")

    async def shutdown(self) -> None:
        """The server itself is going down — slurm allocation expiry, a pkill,
        or a redeploy. Uvicorn gives us this one window before the process ends:
        tear the agent tree down cleanly (claude -> bash -> uv -> train.py) so no
        training is left burning GPU, and tell the clinician + the owner rather
        than letting the run vanish in silence.

        Only runs on SIGTERM. A SIGKILL (kill -9) bypasses this entirely — slurm
        does send SIGTERM first, with a grace period, so we normally get here.
        """
        if self.state == State.ACTIVE:
            await self._to_interrupted("server_shutdown")
        if self.runner:
            await self.runner.close()  # os.killpg on the whole process group

    # -- user actions -----------------------------------------------------
    async def _reset_backend_to_main(self) -> None:
        """Switch the harness working tree back to a clean `main` before each
        session, so every agent branches from the pristine template — not from
        the previous user's automata/<tag> branch. Only touches tracked files
        (git checkout -f); gitignored outputs like projects/ are NOT deleted.
        """
        if self.settings.gpu_mode != "real" or not self.settings.reset_to_main:
            return
        cwd = self.settings.resolved_claude_cwd()
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", cwd, "checkout", "-f", "main",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, err = await proc.communicate()
            if proc.returncode != 0:
                log.warning("reset-to-main failed for %s: %s", self.id, err.decode("utf-8", "replace")[:200])
            else:
                log.info("backend reset to clean main for session %s", self.id)
        except Exception as e:  # never block session start on this
            log.warning("reset-to-main error for %s: %s", self.id, e)

    async def start_runner(self) -> None:
        self.state = State.ACTIVE
        # Notify the project owner that a session started (which code + email).
        await emailer.send_owner_new_session(
            self.onboarding.email, self.owner_code, self.id, self.onboarding.language
        )
        # Start every agent from a clean main (isolation between users).
        await self._reset_backend_to_main()
        self.runner = make_runner(self.settings, self.emit, self.onboarding.language)
        await self.runner.start()
        self._watchdog_task = asyncio.create_task(self._idle_watchdog())
        await self._broadcast({"kind": "session_state", "state": self.state.value})
        await self._set_turn("awaiting_user")

    def _start_tailer(self) -> None:
        """Begin streaming run.log progress to the browser (real GPU only).

        Idempotent: we must not stack a second tailer if anything else calls in.
        """
        if self.settings.gpu_mode != "real" or self.tailer is not None:
            return
        self.tailer = RunLogTailer(self.settings, self.emit)
        self.tailer.start()

    async def handle_user_message(self, text: str, is_task_submission: bool) -> None:
        # Allow follow-up chat after completion too (the agent is still alive
        # and standing by; the user may ask about the results / usage manual).
        if not self.runner or self.state not in (State.ACTIVE, State.COMPLETED):
            return
        if is_task_submission and self.task_submitted:
            return  # one-shot
        # Echo the user's message into the transcript (original text, WITHOUT
        # the kickoff suffix) so it renders as a user bubble and is replayed on
        # reconnect.
        await self._broadcast({"kind": "user_message", "text": text})
        if is_task_submission:
            self.task_submitted = True
            self.autonomous = True
            self.task_text = text  # verbatim, before the kickoff suffix
            text = text + self.settings.task_kickoff_suffix
            await self._broadcast({"kind": "task_submitted", "text": self.task_text})
            # Real harness: the agent runs training itself and redirects it to
            # run.log; tail those files so the clinician sees live progress.
            self._start_tailer()
        self.last_activity = time.time()
        await self._set_turn("agent_busy")
        await self.runner.send_user(text)

    async def teardown(self) -> None:
        if self._grace_task:
            self._grace_task.cancel()
        if self._watchdog_task:
            self._watchdog_task.cancel()
        if self.tailer:
            await self.tailer.stop()
        if self.runner:
            await self.runner.close()


class SessionManager:
    """Global single-occupancy holder."""

    def __init__(self) -> None:
        self.current: Session | None = None
        self._lock = asyncio.Lock()

    @property
    def occupied(self) -> bool:
        return self.current is not None and self.current.state in (
            State.ONBOARDING, State.ACTIVE, State.COMPLETED,
        )

    async def start_session(self, owner_code: str, email: str, lang: str = "en") -> tuple[bool, str]:
        async with self._lock:
            if self.occupied:
                assert self.current is not None
                if self.current.owner_code == owner_code:
                    # Same user rejoining. If they picked a different language on
                    # this login, honour it — otherwise the UI would be in the new
                    # language but the completion/interruption emails would still
                    # go out in the language of the original login.
                    self.current.onboarding.language = lang
                    return True, self.current.id  # rejoin own session
                return False, "occupied"
            settings = get_settings()
            sid = uuid.uuid4().hex
            sess = Session(id=sid, settings=settings, owner_code=owner_code)
            sess.onboarding.email = email
            sess.onboarding.language = lang  # drives the language of the emails
            sess._manager = self
            self.current = sess
            log.info("acquired lock, session=%s owner=%s", sid, owner_code)
        # Runner start (git reset + spawn claude + owner email) happens outside
        # the lock. The session already reads as occupied, so no race.
        await sess.start_runner()
        return True, sid

    async def shutdown(self) -> None:
        """Called from uvicorn's shutdown hook — see Session.shutdown()."""
        sess = self.current
        if not sess:
            return
        log.info("server shutting down; tearing down session %s", sess.id)
        try:
            await sess.shutdown()
        except Exception as e:  # a dying server must not hang on cleanup
            log.warning("shutdown of session %s failed: %s", sess.id, e)

    def mine(self, owner_code: str) -> Session | None:
        """The current session iff it belongs to this code (identity-based
        rejoin, works across browsers/devices)."""
        if self.occupied and self.current and self.current.owner_code == owner_code:
            return self.current
        return None

    def get(self, session_id: str) -> Session | None:
        if self.current and self.current.id == session_id:
            return self.current
        return None

    async def release(self, session_id: str, reason: str) -> None:
        async with self._lock:
            if self.current and self.current.id == session_id:
                log.info("releasing lock, session=%s reason=%s", session_id, reason)
                await self.current.teardown()
                self.current = None


manager = SessionManager()
