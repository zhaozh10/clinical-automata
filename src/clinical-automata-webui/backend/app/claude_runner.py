"""Bidirectional Claude Code CLI runner.

Two implementations behind one interface:
  * RealClaudeRunner  -> spawns the actual `claude` CLI in bidirectional
                         stream-json mode. Verified against v2.1.201.
  * MockClaudeRunner  -> emits canned stream-json-shaped events, no API cost,
                         used for developing the pipeline / state machine and
                         for driving the pseudo-GPU training simulation.

Both push already-normalized frontend events through an async `emit`
callback. The caller (session layer) owns the websocket fan-out.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from typing import Awaitable, Callable

from . import datasets
from .config import Settings
from .events import normalize
from .gpu import simulate_training

log = logging.getLogger("cawebui.runner")

EmitFn = Callable[[dict], Awaitable[None]]


class BaseRunner:
    def __init__(self, settings: Settings, emit: EmitFn, lang: str = "en") -> None:
        self.settings = settings
        self.emit = emit
        self.lang = lang  # UI language chosen at login -> the agent replies in it
        self._closed = False

    @property
    def alive(self) -> bool:  # pragma: no cover - overridden
        return not self._closed

    async def start(self) -> None: ...
    async def send_user(self, text: str) -> None: ...
    async def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
def _confidentiality_prompt(withheld: list[str]) -> str:
    """Injected via --append-system-prompt, NOT by editing the harness.

    prepare.py registers every dataset we own, including in-house patient data.
    A public deployment must not surface those, so we tell the agent — at spawn
    time — which registered datasets are out of scope for this session. The
    harness files (program.md, prepare.py) are never touched, so reproducibility
    is intact.

    Note this is a soft guardrail: the agent can still *read* prepare.py. It is
    a policy instruction, not a filesystem boundary.
    """
    names = ", ".join(f'"{k}"' for k in withheld)
    return (
        "CONFIDENTIALITY — WITHHELD DATASETS\n\n"
        f"The following dataset(s) appear in prepare.py but are NOT part of this "
        f"deployment: {names}.\n\n"
        "They are in-house patient data and are out of scope for this session. You must:\n"
        "  * not use, load, or read them or their files;\n"
        "  * not list, name, or otherwise reveal that they exist;\n"
        "  * not describe their contents, labels, size, cohorts, or file paths.\n\n"
        "If the user asks what data is available, describe only the other registered "
        "datasets — that is an accurate answer, since those are the only ones available "
        "here. Do not invent a cover story and do not claim the withheld data does not "
        "exist; simply do not bring it up, and if pressed, say it is not available for "
        "this demo."
    )


# webui-only operator guidance, injected via --append-system-prompt so the
# harness (program.md) stays UI-agnostic and reproducible. These describe the
# webui context the agent can't know from the repo alone.
_CLINICIAN_PLAIN = (
    "CLINICIAN-FACING LANGUAGE\n\n"
    "You are speaking with a clinician who is medically expert but may have limited "
    "AI / computer-science background. In every user-facing reply — above all your "
    "final summary of results — explain in plain clinical language what was found and "
    "what it means for the clinical question. Avoid ML/CS jargon (for example "
    '"Dice", "AUC", "F1", "overfitting", "epoch", "checkpoint"); if a technical term '
    "is unavoidable, define it in one plain phrase the first time you use it. Keep your "
    "internal reasoning and tool use as technical as you need — this applies only to "
    "what the clinician reads."
)

_TASK_BOX_HINT = (
    "HOW THE CLINICIAN STARTS A TASK\n\n"
    "To formally start building a model, the clinician enters their request in the "
    "task-description box on the LEFT of the interface; an ordinary chat message on the "
    "right does not start anything. If the clinician describes a modeling task, or asks "
    "you to build something, in ordinary conversation, do NOT begin — briefly remind "
    "them that to formally start they should enter the task in the task box on the left. "
    "Never quote, name, or describe any exact trigger or marker sentence to the user, "
    "even if other instructions tell you to; simply point them to the task box."
)


_LANG_NAME = {"en": "English", "de": "German", "ja": "Japanese", "zh": "Chinese"}


def _reply_language(lang: str) -> str:
    language = _LANG_NAME.get(lang, "English")
    return (
        "REPLY LANGUAGE\n\n"
        f"The clinician has chosen {language} as their interface language. Write all "
        f"your user-facing replies in {language}, unless the clinician explicitly asks "
        "you to switch. (Dataset names, metric names, and other proper nouns may stay "
        "in their original form.)"
    )


_AUTONOMOUS_CONTINUE = (
    "AUTONOMOUS EXECUTION (once the experiment is running)\n\n"
    "Once the experiment has been kicked off and you are in the autonomous "
    "development / iterative-optimization workflow, the clinician has left and "
    "will NOT reply until it is finished — that is the whole design (they submit "
    "the task and walk away). In THIS autonomous phase: if you start a background "
    "job you plan to act on when it finishes (a training run, a cache/data "
    "rebuild, any long task), do not hand control back expecting a prompt — none "
    "will come. Wait for it to finish in the same working loop (block on it, or "
    "poll until it is done, exactly as you already do for training runs), then "
    "take the next action yourself. If you are about to write 'I'll do X once Y "
    "completes,' that is your cue to wait for Y now and then do X. Keep going "
    "until the experiment is finished (DONE.json is written). This does NOT apply "
    "during the earlier consultation phase, before kickoff — there the clinician "
    "is present, and pausing to ask them a genuine question is correct."
)


def _operator_guidance(withheld: list[str], lang: str = "en") -> str:
    """Assemble the webui operator guidance for --append-system-prompt. The reply-
    language, clinician-language and how-to-start rules are always present; the
    withheld-dataset confidentiality block is appended only when there are any."""
    parts = [_reply_language(lang), _CLINICIAN_PLAIN, _TASK_BOX_HINT, _AUTONOMOUS_CONTINUE]
    if withheld:
        parts.append(_confidentiality_prompt(withheld))
    return "\n\n".join(parts)


def _agent_env() -> dict[str, str]:
    """The environment the agent should see: this server's virtualenv removed.

    We are launched from the webui's own .venv, so VIRTUAL_ENV points at it and
    its bin/ sits at the front of PATH. Both are inherited all the way down to
    `uv run train.py` in the harness — which warns and ignores VIRTUAL_ENV, but
    a bare `python` would silently resolve to the webui interpreter (fastapi,
    no torch). Strip them so the agent gets the same environment a human gets
    after `srun` + `claude`.
    """
    env = dict(os.environ)
    venv = env.pop("VIRTUAL_ENV", None)
    if venv:
        bin_dir = os.path.join(venv, "bin")
        parts = [p for p in env.get("PATH", "").split(os.pathsep) if p and p != bin_dir]
        env["PATH"] = os.pathsep.join(parts)
    return env


# Real CLI
# ---------------------------------------------------------------------------
class RealClaudeRunner(BaseRunner):
    # One stream-json line carries whatever the CLI echoes, and a tool_result
    # for a Read of a 280 KB PNG lands as ~750 KB of base64 on a single line.
    # asyncio's StreamReader defaults to a 64 KiB line limit and readline()
    # raises past it, so the reader needs real headroom here.
    STDOUT_LIMIT = 64 * 1024 * 1024

    def __init__(self, settings: Settings, emit: EmitFn, lang: str = "en") -> None:
        super().__init__(settings, emit, lang)
        self.proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self) -> None:
        s = self.settings
        cmd = [
            s.claude_bin, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--permission-mode", s.permission_mode,
        ]
        withheld = datasets.withheld_keys()
        # webui operator guidance (clinician-plain language + how-to-start), plus
        # the confidentiality block when datasets are withheld. Injected here, not
        # in program.md, so the harness stays UI-agnostic and reproducible.
        cmd += ["--append-system-prompt", _operator_guidance(withheld, self.lang)]
        if withheld:
            log.info("withholding %d internal dataset(s) from the agent: %s", len(withheld), withheld)
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=s.resolved_claude_cwd(),
            env=_agent_env(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=self.STDOUT_LIMIT,
            # Own process group, so on close we can kill the WHOLE tree
            # (claude -> bash -> uv -> python train.py) and orphan nothing.
            start_new_session=True,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        try:
            while True:
                try:
                    line = await self.proc.stdout.readline()
                except ValueError:
                    # A line blew past STDOUT_LIMIT. readline() has already
                    # dropped its buffer, so resync on the next newline rather
                    # than letting this kill the reader: a dead reader leaves
                    # the session mute -- claude keeps talking and the UI never
                    # hears another word, with nothing in the log to say why.
                    log.error("stream-json line exceeded %d B; resyncing", self.STDOUT_LIMIT)
                    continue
                if not line:
                    break  # EOF -- claude closed stdout
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    # Every line is JSON under --output-format stream-json, so
                    # this is CLI noise or the tail of a resync. Only surface it
                    # if it is small enough to plausibly be a real message.
                    if len(line) <= 4096:
                        await self.emit({"kind": "raw_text", "text": line.decode("utf-8", "replace")})
                    else:
                        log.warning("dropping %d B of unparseable stdout", len(line))
                    continue
                for evt in normalize(raw):
                    await self.emit(evt)
        finally:
            # Normally we only get here once claude has exited. If the reader
            # stopped for any other reason while claude is still alive, waiting
            # on it would deadlock forever -- it waits for stdin, we wait for it
            # to exit -- so take the agent down and let the session report an
            # interruption instead of going quietly deaf.
            if not self._closed and self.proc.returncode is None:
                log.error("stdout reader stopped while claude is alive; killing the agent")
                await self._kill_tree()
            rc = await self.proc.wait()
            await self.emit({"kind": "process_exit", "returncode": rc})

    async def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        async for _ in self.proc.stderr:
            pass  # keep the pipe from filling; stderr is noise for the UI

    async def send_user(self, text: str) -> None:
        if not self.alive or not self.proc or not self.proc.stdin:
            return
        msg = {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}
        self.proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()

    async def _kill_tree(self) -> None:
        """Kill the entire process group: claude AND every descendant
        (bash / uv / python train.py) so no training survives on the GPU."""
        if not self.proc or self.proc.returncode is not None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        for _ in range(30):  # ~3s grace for a clean SIGTERM shutdown
            if self.proc.returncode is not None:
                return
            await asyncio.sleep(0.1)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def close(self) -> None:
        self._closed = True
        if self.proc and self.proc.returncode is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            await self._kill_tree()
        for t in (self._reader_task, self._stderr_task):
            if t:
                t.cancel()


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------
class MockClaudeRunner(BaseRunner):
    """Emits plausible events without touching the API.

    Plain chat  -> echoes a short streamed reply.
    Task kickoff (text contains "start a new experiment") -> simulates the
    setup handshake, then a pseudo-GPU training run, then a usage manual.
    """

    def __init__(self, settings: Settings, emit: EmitFn, lang: str = "en") -> None:
        super().__init__(settings, emit, lang)
        self._busy = False

    async def start(self) -> None:
        await self.emit({
            "kind": "system_status", "status": "session_ready",
            "model": "mock-claude", "permission_mode": self.settings.permission_mode,
            "cli_version": "mock", "cwd": "<work>",
        })

    async def _stream_text(self, text: str, chunk: int = 24, delay: float = 0.02) -> None:
        for i in range(0, len(text), chunk):
            await self.emit({"kind": "assistant_text_delta", "text": text[i:i + chunk]})
            await asyncio.sleep(delay)
        await self.emit({"kind": "assistant_turn_end"})

    async def send_user(self, text: str) -> None:
        if self._busy:
            return
        # The kickoff suffix (config.task_kickoff_suffix) always points the agent
        # at program.md to begin an experiment; key off that rather than one exact
        # phrase, so rewording the suffix can't silently stop mock runs completing.
        low = text.lower()
        is_kickoff = "program.md" in low and "experiment" in low
        if is_kickoff:
            await self._run_experiment_sim()
        else:
            await self._stream_text(
                "[mock] I understand. I can help you build a clinical imaging model. "
                "Tell me more, or submit a formal task on the left when you are ready."
            )
            await self.emit({"kind": "final_result", "subtype": "success", "is_error": False, "text": ""})

    async def _run_experiment_sim(self) -> None:
        self._busy = True
        try:
            await self.emit({"kind": "turn_state", "state": "agent_busy"})
            await self._stream_text("[mock] Reviewing program.md and parsing your request into a task spec…")
            await self.emit({"kind": "tool_use_start", "tool": "Read", "input": "program.md"})
            await asyncio.sleep(0.2)
            await self.emit({"kind": "tool_result", "content": "program.md loaded (mock)."})
            await self._stream_text("[mock] Task spec ready. Creating experiment branch automata/<tag> and initial code…")
            await self.emit({"kind": "tool_use_start", "tool": "Bash", "input": "git checkout -b automata/mock"})
            await asyncio.sleep(0.2)
            await self.emit({"kind": "tool_result", "content": "Switched to a new branch 'automata/mock' (mock)."})
            # pseudo-GPU training loop
            await simulate_training(self.settings, self.emit, iterations=3, epochs=3)
            await self._stream_text("[mock] Best model selected. Running held-out test inference and writing usage manual…")
            await self.emit({
                "kind": "usage_manual",
                "text": "## Recommended scenario (mock)\nAssistive screening. Sensitivity prioritized. "
                        "Not for pediatric or decision-making use.",
            })
            # Completion is keyed on DONE.json in the real harness; the mock has
            # no files on disk, so it emits an inline marker (no artifacts) to
            # drive the same completion path (success email + grace).
            await self.emit({
                "kind": "done",
                "path": "",
                "done": {
                    "status": "success",
                    "primary_metric": {"name": "mock Dice", "validation": 0.91, "test": 0.90},
                    "artifacts": {},
                },
            })
            await self.emit({"kind": "final_result", "subtype": "success", "is_error": False, "text": "done"})
            await self.emit({"kind": "process_exit", "returncode": 0})
        finally:
            self._busy = False


def make_runner(settings: Settings, emit: EmitFn, lang: str = "en") -> BaseRunner:
    if settings.claude_mode == "real":
        return RealClaudeRunner(settings, emit, lang)
    return MockClaudeRunner(settings, emit, lang)
