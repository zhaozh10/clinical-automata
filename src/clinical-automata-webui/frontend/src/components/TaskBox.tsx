import { useEffect, useState } from "react";
import { useT } from "../i18n";

/** One-shot task submission. It locks into a read-only card ONLY
 *  once the server confirms it (the `disabled`/`submitted` props, driven by the
 *  backend `task_submitted` echo) — never optimistically on click, which used to
 *  make a dropped submit look successful while the run never started. */
export default function TaskBox({
  onSubmit,
  disabled,
  submitted,
}: {
  /** Returns whether the message actually went out on a live socket. */
  onSubmit: (text: string) => boolean;
  disabled: boolean;
  /** The task as the server has it. Local `text` is empty on a fresh mount —
   *  a reload or another device — so this is what actually
   *  survives; without it the box falls back to "your request is gone". */
  submitted?: string;
}) {
  const t = useT();
  const [text, setText] = useState("");
  const [pending, setPending] = useState(false);
  const [failed, setFailed] = useState(false);

  // Server confirmed the task -> clear any pending/error state.
  useEffect(() => {
    if (disabled) {
      setPending(false);
      setFailed(false);
    }
  }, [disabled]);

  // Sent, but no confirmation came back in time (dropped socket, or the server
  // ignored it): stop pretending it's in flight and let them retry.
  useEffect(() => {
    if (!pending) return;
    const id = setTimeout(() => {
      setPending(false);
      setFailed(true);
    }, 10000);
    return () => clearTimeout(id);
  }, [pending]);

  // Only the SERVER-confirmed state shows the read-only "submitted" card.
  if (disabled) {
    return (
      <div className="taskbox submitted">
        <div className="label">{t.taskSubmittedLabel}</div>
        <div className="submitted-text">{submitted || text || t.taskSubmittedFallback}</div>
      </div>
    );
  }

  const submit = () => {
    if (!text.trim() || pending) return;
    const ok = onSubmit(text);
    setFailed(!ok);
    setPending(ok); // wait for the backend `task_submitted` echo to lock
  };

  return (
    <div className="taskbox">
      <div className="label">{t.taskLabel}</div>
      <textarea
        placeholder={t.taskPh}
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={8}
        disabled={pending}
      />
      <button className="primary" disabled={!text.trim() || pending} onClick={submit}>
        {pending ? t.taskSending : t.taskSubmit}
      </button>
      {failed && <div className="send-failed">⚠ {t.taskSendFailed}</div>}
      {/* Set expectations before they commit: the setup phase is long and silent. */}
      <div className="setup-warn">⏳ {t.taskSetupWarn}</div>
    </div>
  );
}
