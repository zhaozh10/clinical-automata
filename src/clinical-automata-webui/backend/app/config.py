"""Runtime configuration, driven by environment variables.

Everything that differs between "skeleton dev on the login node" and "real
demo inside a Slurm allocation" is a setting here, so no paths or modes are
hard-coded.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


SITE_CONFIG = Path(__file__).resolve().parents[3] / "config.env"


class Settings(BaseSettings):
    # Site settings come from src/config.env (shared with the harness and the
    # deploy scripts); a local backend/.env, then real env vars, override it.
    model_config = SettingsConfigDict(
        env_prefix="CAWEBUI_", env_file=(SITE_CONFIG, ".env"), extra="ignore"
    )

    # --- access / auth ---------------------------------------------------
    # Per-user codes live in access_codes.txt (one per line, '#' comments).
    # A code IS the user's identity: same code -> same user (rejoin across
    # browsers/devices). Falls back to `access_code` if the file is absent.
    access_code: str = Field(default="CHANGE_ME_ACCESS_CODE", description="Fallback single passcode.")
    access_codes_file: str = Field(
        default=str(Path(__file__).resolve().parents[2] / "access_codes.txt")
    )
    secret_key: str = Field(default="CHANGE_ME_SECRET_KEY", description="Cookie signing secret.")
    # Dataset root (shared DATA_ROOT key, no prefix); scrubbed from anything sent to the browser.
    data_root: str = Field(default="/path/to/your/datasets", validation_alias="DATA_ROOT")
    cookie_name: str = "cawebui_session"
    cookie_max_age: int = 60 * 60 * 24  # 1 day

    def valid_codes(self) -> set[str]:
        codes: set[str] = set()
        try:
            with open(self.access_codes_file, "r", encoding="utf-8") as f:
                for line in f:
                    # Strip trailing "# who this code is for" annotations, not just
                    # whole-line comments — otherwise the annotation becomes part of
                    # the code and the invited user cannot log in at all.
                    s = line.split("#", 1)[0].strip()
                    if s:
                        codes.add(s)
        except OSError:
            pass
        if not codes and self.access_code:
            codes.add(self.access_code)
        return codes

    # --- claude runner ---------------------------------------------------
    # "mock"  -> canned stream-json events (no API cost); used for most dev.
    # "real"  -> spawn the actual `claude` CLI (bidirectional stream-json).
    claude_mode: str = Field(default="mock")
    claude_bin: str = Field(default="claude")
    # cwd for the claude subprocess. For the skeleton this defaults to a
    # scratch dir (plain chat). When we wire the real harness this becomes
    # BACKEND_DIR (../clinical-automata-backend) with bypassPermissions.
    claude_cwd: str = Field(default="")  # empty -> use work_dir/scratch
    permission_mode: str = Field(default="default")  # "bypassPermissions" for real harness
    # Backend harness location (sibling repo). Not used until real training.
    backend_dir: str = Field(default=str(Path(__file__).resolve().parents[3] / "clinical-automata-backend"))

    # --- dataset visibility ----------------------------------------------
    # "public"   -> datasets flagged `internal: true` in datasets.yaml are
    #               hidden from the cards AND the agent is instructed not to
    #               use them or reveal that they exist. This is the DEFAULT,
    #               so an external session can never accidentally expose them.
    # "internal" -> everything is shown (for our own use).
    visibility: str = Field(default="public")

    @property
    def show_internal(self) -> bool:
        return self.visibility == "internal"

    # --- gpu / training --------------------------------------------------
    # "pseudo" -> no Slurm, fake device + simulated training events.
    # "real"   -> the agent runs training itself; we tail run.log for progress.
    gpu_mode: str = Field(default="pseudo")
    pseudo_gpu_id: str = Field(default="pseudo:0")
    run_log_poll_interval: float = Field(default=1.0)  # seconds, real run.log tail

    reset_to_main: bool = Field(default=True)  # checkout a clean main before each session

    # --- session / lock --------------------------------------------------
    grace_seconds_after_success: int = Field(default=24 * 60 * 60)  # 24h after DONE.json, then auto-end
    # Watchdog: if the agent goes silent (no tool in flight) this long after
    # speaking without a `result` event, re-enable input (e.g. result stalled).
    idle_unlock_seconds: float = Field(default=15.0)

    # --- email -----------------------------------------------
    # Owner is notified on every session start; the user is notified on
    # success. Gmail SMTP (587) is open from the cluster. Provide an App
    # Password via CAWEBUI_smtp_password; if empty, email is logged (stub).
    owner_email: str = Field(default="your-email@example.com")
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_user: str = Field(default="your-email@example.com")
    smtp_password: str = Field(default="")  # Gmail App Password
    email_from: str = Field(default="")     # defaults to smtp_user

    # Optional IMAP: SMTP only *sends*; to also keep a copy in the mailbox's
    # "Sent" folder (like a normal mail client does), APPEND via IMAP after
    # sending. Empty imap_host disables it — the mail still goes out fine.
    imap_host: str = Field(default="")           # e.g. imap.example.com
    imap_port: int = Field(default=993)          # IMAPS
    imap_sent_folder: str = Field(default="Sent Items")

    # --- deliverables (completion email) ---------------------------------
    # Public base URL of the demo (the Cloudflare tunnel), used to build signed
    # download links for deliverables too large to attach. e.g.
    # "https://demo.example.com". Empty -> an oversized file is skipped (with a
    # log line) rather than mailed; set this in .env for real demos.
    public_url: str = Field(default="")
    # Attach a deliverable up to this size; larger ones (or product folders,
    # after zipping) become a download link instead. total_attach_budget caps
    # the summed attachment size so a multi-file mail stays under the SMTP limit.
    attach_max_bytes: int = Field(default=20 * 1024 * 1024)      # 20 MB / file
    total_attach_budget: int = Field(default=25 * 1024 * 1024)   # 25 MB total

    # LLM-refined completion summary: after a run finishes, a short-lived
    # `claude -p` opens the project folder (projects/YYYYMMDD_HH/), reads its
    # artifacts, and writes a clinician-facing description that replaces the
    # generic template intro in the success email. Best-effort — on any failure
    # (disabled, timeout, non-zero exit, empty output) the email falls back to
    # the template. Only runs in real mode (needs a real project directory).
    email_llm_refine: bool = Field(default=True)
    email_refine_timeout: int = Field(default=150)  # seconds, hard cap

    @property
    def from_addr(self) -> str:
        return self.email_from or self.smtp_user

    # --- misc ------------------------------------------------------------
    work_dir: str = Field(default=str(Path(__file__).resolve().parents[1] / "_work"))
    datasets_manifest: str = Field(
        default=str(Path(__file__).resolve().parents[2] / "datasets.yaml")
    )
    # Fixed suffix appended to the one-shot task submission.
    task_kickoff_suffix: str = (
        "\n\nPlease have a look at program.md and kick off a new experiment! "
        "let's do the setup first."
    )

    @property
    def work_path(self) -> Path:
        p = Path(self.work_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolved_claude_cwd(self) -> str:
        if self.claude_cwd:
            return self.claude_cwd
        # Real-harness profile: run claude inside the harness repo so it can
        # read program.md, branch, write train.py, and see the datasets.
        if self.gpu_mode == "real":
            return self.backend_dir
        scratch = self.work_path / "claude_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        return str(scratch)

    @property
    def projects_dir(self) -> Path:
        """Where the harness writes projects/YYYYMMDD_HH/run{i}/run.log."""
        return Path(self.resolved_claude_cwd()) / "projects"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if s.claude_mode == "real" and s.secret_key.startswith("CHANGE_ME"):
        raise RuntimeError(f"Set CAWEBUI_secret_key in {SITE_CONFIG} before running a real deployment.")
    return s
