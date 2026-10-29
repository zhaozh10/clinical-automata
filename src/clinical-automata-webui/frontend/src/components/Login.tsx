import { useState } from "react";
import { api } from "../api";
import { LANGS, useT, type Lang } from "../i18n";

export default function Login({
  lang,
  onLang,
  onAuthed,
}: {
  lang: Lang;
  onLang: (l: Lang) => void;
  onAuthed: () => void;
}) {
  const t = useT();
  const [code, setCode] = useState("");
  const [email, setEmail] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const valid = code.trim() && email.includes("@");

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr("");
    try {
      // The chosen language also reaches the backend, so the notification
      // emails are written in the same language as the interface.
      await api.login(code.trim(), email.trim(), lang);
      onAuthed();
    } catch {
      setErr(t.badCode);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="center">
      <form className="card login" onSubmit={submit}>
        <h1>Clinical-Automata</h1>
        <p className="muted">{t.tagline}</p>

        <div className="login-label">
          {t.language}
          <div className="lang-picker">
            {LANGS.map((l) => (
              <button
                key={l.code}
                type="button"
                className={`lang-btn ${l.code === lang ? "on" : ""}`}
                onClick={() => onLang(l.code)}
              >
                {l.label}
              </button>
            ))}
          </div>
        </div>

        <label className="login-label">
          {t.accessCode}
          <input
            type="text"
            placeholder={t.accessCodePh}
            value={code}
            onChange={(e) => setCode(e.target.value)}
            autoFocus
          />
        </label>
        <label className="login-label">
          {t.email}
          <input
            type="email"
            placeholder={t.emailPh}
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </label>
        <button className="primary" disabled={busy || !valid}>
          {busy ? "…" : t.enter}
        </button>
        {err && <div className="error">{err}</div>}
      </form>
    </div>
  );
}
