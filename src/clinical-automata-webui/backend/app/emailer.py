"""Email notifications.

Gmail SMTP over STARTTLS (port 587, confirmed open from the cluster). Provide
a Gmail App Password via CAWEBUI_smtp_password; if it's empty we log instead of
sending (safe for local/mock testing). All sends are best-effort and never
crash the session.

Triggers:
  * owner  <- on every session start (age / experience / email)
  * user   <- on successful completion (usage manual generated)
  * user   <- on task interruption
"""
from __future__ import annotations

import asyncio
import imaplib
import logging
import mimetypes
import os
import smtplib
import ssl
import time
from email.message import EmailMessage

from .config import get_settings

log = logging.getLogger("cawebui.email")


def _save_to_sent(msg: EmailMessage) -> None:
    """APPEND a copy of a just-sent message to the mailbox's Sent folder, so it
    shows up like a normal mail client's "Sent Items". Best-effort: SMTP already
    delivered the mail, so any IMAP failure here is logged and swallowed."""
    s = get_settings()
    if not s.imap_host:
        return
    try:
        M = imaplib.IMAP4_SSL(s.imap_host, s.imap_port, timeout=20)
        M.login(s.smtp_user, s.smtp_password)
        M.append(
            f'"{s.imap_sent_folder}"', "\\Seen",
            imaplib.Time2Internaldate(time.time()), msg.as_bytes(),
        )
        M.logout()
        log.info("[email:SENT-COPY] saved to '%s'", s.imap_sent_folder)
    except Exception as e:  # never let the Sent-copy break anything
        log.warning("[email:SENT-COPY-FAILED] %s", e)


def deliver(msg: EmailMessage) -> None:
    """THE single send path. Sends via SMTP and always keeps a copy in the
    mailbox's Sent folder. Every email — the app's own notifications AND any
    ad-hoc/manual send — must go through here, so a "sent but not in Sent Items"
    gap cannot happen. Fills From from settings if the caller left it blank."""
    s = get_settings()
    if not msg["From"]:
        msg["From"] = s.from_addr
    if not s.smtp_password:
        log.info(
            "[email:STUB no smtp_password] to=%s subject=%s", msg["To"], msg["Subject"]
        )
        return
    ctx = ssl.create_default_context()
    with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=30) as server:
        server.starttls(context=ctx)
        server.login(s.smtp_user, s.smtp_password)
        server.send_message(msg)
    log.info("[email:SENT] to=%s subject=%s", msg["To"], msg["Subject"])
    _save_to_sent(msg)  # ALWAYS keep a copy in Sent (best-effort)


def _send_sync(to: str, subject: str, body: str, attachment: tuple[str, str] | None = None) -> None:
    s = get_settings()
    msg = EmailMessage()
    msg["From"] = s.from_addr
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    if attachment:
        filename, content = attachment
        # markdown as a plain-text attachment: opens anywhere, no client needed
        msg.add_attachment(
            content.encode("utf-8"), maintype="text", subtype="markdown", filename=filename
        )
    deliver(msg)


async def _send(to: str, subject: str, body: str, attachment: tuple[str, str] | None = None) -> None:
    if not to:
        return
    try:
        await asyncio.to_thread(_send_sync, to, subject, body, attachment)
    except Exception as e:  # never let email break a session
        log.warning("[email:FAILED] to=%s subject=%s err=%s", to, subject, e)


# User-facing mails follow the interface language chosen at login. The owner
# mail is always English (it is for the project owner, not the clinician).
#
# The completion mail is assembled from the parts below plus runtime data:
# attachments (small deliverables), download links (large ones), the primary
# metric, and the grace period. A plain-text and an HTML alternative are built
# so links render as clickable anchors while dumb clients still see the URL.
# `status` selects the opening line: success / partial / failed.
SUCCESS_TPL: dict[str, dict[str, str]] = {
    "en": {
        "subject": "[Clinical-Automata] Your model is ready",
        "subject_partial": "[Clinical-Automata] Your model — partial result",
        "subject_failed": "[Clinical-Automata] Your session did not produce a model",
        "open_success": "Your Clinical-Automata model development has finished, and your model is ready.",
        "open_partial": "Your Clinical-Automata model development has finished. Some steps did not complete fully, so the results below are partial.",
        "open_failed": "Your Clinical-Automata session has finished, but unfortunately it did not produce a usable model.",
        "intro": "The autonomous coding agent completed the experiment and prepared the deliverables below. You can return to the web interface at any time to review the results and ask the assistant for more details.",
        "metric": "Primary metric ({name}) on the independent test set: {test}.",
        "attach_header": "Attached to this email:",
        "links_header": "Larger files are provided as download links (valid for 7 days):",
        "grace": "IMPORTANT: your session stays open for another {h} hours. After that it is closed automatically and you can no longer return to the conversation.",
        "session": "Session ID: {sid}",
        "download": "Download",
    },
    "de": {
        "subject": "[Clinical-Automata] Ihr Modell ist fertig",
        "subject_partial": "[Clinical-Automata] Ihr Modell — Teilergebnis",
        "subject_failed": "[Clinical-Automata] Ihre Sitzung hat kein Modell erzeugt",
        "open_success": "Die Modellentwicklung in Clinical-Automata ist abgeschlossen, und Ihr Modell ist fertig.",
        "open_partial": "Die Modellentwicklung in Clinical-Automata ist abgeschlossen. Einige Schritte konnten nicht vollständig ausgeführt werden, daher sind die folgenden Ergebnisse unvollständig.",
        "open_failed": "Ihre Clinical-Automata-Sitzung ist abgeschlossen, hat aber leider kein verwendbares Modell erzeugt.",
        "intro": "Der autonome Coding-Agent hat das Experiment beendet und die unten aufgeführten Ergebnisse vorbereitet. Sie können jederzeit zur Weboberfläche zurückkehren, um die Ergebnisse anzusehen und dem Assistenten weitere Fragen zu stellen.",
        "metric": "Primäre Metrik ({name}) auf dem unabhängigen Testsatz: {test}.",
        "attach_header": "Dieser E-Mail beigefügt:",
        "links_header": "Größere Dateien stehen als Download-Links zur Verfügung (7 Tage gültig):",
        "grace": "WICHTIG: Ihre Sitzung bleibt noch {h} Stunden geöffnet. Danach wird sie automatisch geschlossen und Sie können nicht mehr zum Gespräch zurückkehren.",
        "session": "Session ID: {sid}",
        "download": "Herunterladen",
    },
    "ja": {
        "subject": "[Clinical-Automata] モデルが完成しました",
        "subject_partial": "[Clinical-Automata] モデル — 部分的な結果",
        "subject_failed": "[Clinical-Automata] 今回はモデルを生成できませんでした",
        "open_success": "Clinical-Automata でのモデル開発が完了し、モデルが完成しました。",
        "open_partial": "Clinical-Automata でのモデル開発が完了しました。一部の工程が完全には終わらなかったため、以下の結果は部分的なものです。",
        "open_failed": "Clinical-Automata のセッションは終了しましたが、残念ながら利用可能なモデルを生成できませんでした。",
        "intro": "自律型コーディング・エージェントが実験を終え、以下の成果物を用意しました。ウェブ画面にはいつでも戻ることができ、結果を確認したり、アシスタントにさらに詳しい説明を求めたりできます。",
        "metric": "独立したテストセットでの主要指標（{name}）: {test}。",
        "attach_header": "本メールに添付：",
        "links_header": "サイズの大きいファイルはダウンロードリンクで提供します（7日間有効）：",
        "grace": "【ご注意】セッションはあと {h} 時間だけ開いています。その後は自動的に終了し、会話に戻ることはできなくなります。",
        "session": "Session ID: {sid}",
        "download": "ダウンロード",
    },
    "zh": {
        "subject": "[Clinical-Automata] 您的模型已完成",
        "subject_partial": "[Clinical-Automata] 您的模型 — 部分结果",
        "subject_failed": "[Clinical-Automata] 本次会话未能生成模型",
        "open_success": "您在 Clinical-Automata 上的模型开发已经完成，模型已就绪。",
        "open_partial": "您在 Clinical-Automata 上的模型开发已经完成。部分环节未能完全完成，因此下面的结果是部分性的。",
        "open_failed": "您的 Clinical-Automata 会话已经结束，但很遗憾未能生成可用的模型。",
        "intro": "自主编码智能体已结束实验，并准备好了下列交付物。您可以随时回到网页界面查看结果，并向助手追问更多细节。",
        "metric": "在独立测试集上的主要指标（{name}）：{test}。",
        "attach_header": "本邮件附件：",
        "links_header": "较大的文件以下载链接形式提供（7 天内有效）：",
        "grace": "【请注意】您的会话还会保留 {h} 小时，之后将自动关闭，届时您将无法再回到该对话。",
        "session": "Session ID: {sid}",
        "download": "下载",
    },
}

INTERRUPTED: dict[str, tuple[str, str]] = {
    "en": (
        "[Clinical-Automata] Session interrupted",
        "Your Clinical-Automata session was interrupted before it finished.\n",
    ),
    "de": (
        "[Clinical-Automata] Sitzung unterbrochen",
        "Ihre Clinical-Automata-Sitzung wurde vor dem Abschluss unterbrochen.\n",
    ),
    "ja": (
        "[Clinical-Automata] セッションが中断されました",
        "Clinical-Automata のセッションは完了前に中断されました。\n",
    ),
    "zh": (
        "[Clinical-Automata] 会话已中断",
        "您的 Clinical-Automata 会话在完成之前被中断了。\n",
    ),
}


def _pick(table: dict, lang: str):
    return table.get(lang, table["en"])


async def send_owner_new_session(email: str, code: str, session_id: str, lang: str = "en") -> None:
    s = get_settings()
    body = (
        "A new user has started a Clinical-Automata session.\n\n"
        f"Session ID  : {session_id}\n"
        f"Access code : {code}\n"
        f"User email  : {email}\n"
        f"UI language : {lang}\n"
    )
    await _send(s.owner_email, "[Clinical-Automata] New session started", body)


async def send_owner_interruption(
    user_email: str, code: str, session_id: str, reason: str, task_submitted: bool
) -> None:
    s = get_settings()
    hint = (
        "The backend was stopped — most likely the slurm allocation expired, or\n"
        "uvicorn was killed. Any training run was killed with it.\n"
        if reason == "server_shutdown"
        else "The claude process exited on its own (crash, rate limit, or kill).\n"
    )
    # Only claim the user was told if there was somewhere to tell them; saying
    # it unconditionally leaves the owner believing a silent user is informed.
    notified = (
        "The user has been emailed as well."
        if user_email
        else "The user was NOT emailed — no address was given at login."
    )
    body = (
        "A Clinical-Automata session was INTERRUPTED before it finished.\n\n"
        f"Session ID     : {session_id}\n"
        f"Access code    : {code}\n"
        f"User email     : {user_email or '(none given)'}\n"
        f"Task submitted : {'yes' if task_submitted else 'no'}\n"
        f"Cause          : {reason}\n\n"
        f"{hint}\n"
        f"{notified} The lock is released.\n"
    )
    await _send(s.owner_email, "[Clinical-Automata] Session INTERRUPTED", body)


def _fmt_metric(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _hsize(n: int) -> str:
    """Human-readable file size for a link label, e.g. 6291456 -> '6 MB'."""
    if n < 1024:
        return f"{n} B"
    kb = n / 1024
    if kb < 1024:
        return f"{int(kb)} KB"
    mb = kb / 1024
    if mb < 1024:
        return f"{int(mb)} MB"
    return f"{mb / 1024:.1f} GB"


def _build_success(
    lang: str,
    session_id: str,
    grace_hours: int,
    attachment_names: list[str],
    links: list[tuple[str, str, int]],   # (filename, url, size_bytes)
    primary_metric: dict | None,   # accepted for API/kwargs symmetry; not shown
    status: str,
    descriptions: dict | None = None,    # {filename: {"label"?, "description"?}}
) -> tuple[str, str, str]:
    """Assemble (subject, plaintext, html) from the fixed per-language template +
    runtime data. The template (opening, intro, grace, session) is kept verbatim;
    the LLM only *adds* a friendly label + one-line `description` per delivered
    file. No performance/metric line is shown."""
    t = SUCCESS_TPL.get(lang, SUCCESS_TPL["en"])
    subject = t["subject"] if status == "success" else t.get(f"subject_{status}", t["subject"])
    opening = t.get(f"open_{status}", t["open_success"])
    d = descriptions or {}

    def _label(name: str) -> str:
        return (d.get(name) or {}).get("label") or name

    def _desc(name: str) -> str | None:
        return (d.get(name) or {}).get("description")

    grace_line = t["grace"].format(h=grace_hours)
    session_line = t["session"].format(sid=session_id)

    # -- plain text --
    p = [opening, "", t["intro"]]
    if attachment_names:
        p += ["", t["attach_header"]]
        for n in attachment_names:
            dt = _desc(n)
            p.append(f"  - {n} — {dt}" if dt else f"  - {n}")
    if links:
        p += ["", t["links_header"]]
        for fname, url, size in links:
            p.append(f"  - {_label(fname)}, {_hsize(size)} — {t['download']}:\n      {url}")
    p += ["", grace_line, "", session_line]
    plain = "\n".join(p) + "\n"

    # -- html --
    h = [f"<p>{_esc(opening)}</p>", f"<p>{_esc(t['intro'])}</p>"]
    if attachment_names:
        items = ""
        for n in attachment_names:
            dt = _desc(n)
            items += f"<li>{_esc(n)}{' — ' + _esc(dt) if dt else ''}</li>"
        h.append(f"<p>{_esc(t['attach_header'])}</p><ul>{items}</ul>")
    if links:
        items = "".join(
            f'<li>{_esc(_label(fname))}, {_hsize(size)} — '
            f'<a href="{_esc(url)}">{_esc(t["download"])}</a></li>'
            for fname, url, size in links
        )
        h.append(f"<p>{_esc(t['links_header'])}</p><ul>{items}</ul>")
    h.append(f"<p><b>{_esc(grace_line)}</b></p>")
    h.append(f'<p style="color:#888;font-size:12px">{_esc(session_line)}</p>')
    html = (
        '<div style="font-family:-apple-system,Segoe UI,Arial,sans-serif;'
        'font-size:14px;line-height:1.55;color:#222">' + "".join(h) + "</div>"
    )
    return subject, plain, html


def _send_success_sync(
    to_email: str,
    session_id: str,
    lang: str,
    grace_hours: int,
    attachments: list,          # each: file path (str) OR (filename, bytes)
    links: list,                # each: (filename, url, size_bytes)
    primary_metric: dict | None,
    status: str,
    descriptions: dict | None = None,
) -> None:
    s = get_settings()
    # normalize attachments to (filename, bytes)
    files: list[tuple[str, bytes]] = []
    for a in attachments or []:
        if isinstance(a, (tuple, list)):
            files.append((a[0], a[1]))
        else:
            files.append((os.path.basename(a), open(a, "rb").read()))
    subject, plain, html = _build_success(
        lang, session_id, grace_hours, [f[0] for f in files], links or [],
        primary_metric, status, descriptions,
    )
    msg = EmailMessage()
    msg["From"] = s.from_addr
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(plain)                 # plaintext alternative
    msg.add_alternative(html, subtype="html")  # HTML alternative (clickable links)
    for filename, content in files:
        maintype, subtype = _guess_mime(filename)
        msg.add_attachment(content, maintype=maintype, subtype=subtype, filename=filename)
    deliver(msg)


def _guess_mime(filename: str) -> tuple[str, str]:
    mt, _ = mimetypes.guess_type(filename)
    if mt and "/" in mt:
        a, b = mt.split("/", 1)
        return a, b
    return "application", "octet-stream"


async def send_success(
    to_email: str,
    session_id: str,
    lang: str = "en",
    grace_hours: int = 16,
    *,
    attachments: list | None = None,
    links: list | None = None,
    primary_metric: dict | None = None,
    status: str = "success",
    descriptions: dict | None = None,
) -> None:
    """Completion mail. Small deliverables ride as `attachments` (file paths or
    (name, bytes)); large ones as `links` [(filename, url, size_bytes)]. The
    template is fixed; `descriptions` ({filename: {label, description}}) is the
    only LLM-added part (a label + one-liner per file). Sends plaintext + HTML."""
    if not to_email:
        return
    try:
        await asyncio.to_thread(
            _send_success_sync, to_email, session_id, lang, grace_hours,
            attachments or [], links or [], primary_metric, status, descriptions,
        )
    except Exception as e:  # never let email break a session
        log.warning("[email:FAILED] success to=%s err=%s", to_email, e)


async def send_interruption(to_email: str, session_id: str, lang: str = "en") -> None:
    subject, body = _pick(INTERRUPTED, lang)
    await _send(to_email, subject, f"{body}\nSession ID: {session_id}\n")
