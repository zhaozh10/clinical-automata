import { useState } from "react";
import { useT } from "../i18n";

type Example = { title: string; hint: string; text: string };

export default function Examples() {
  const t = useT();
  const [openIdx, setOpenIdx] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);

  const open: Example | null = openIdx == null ? null : t.examples[openIdx];

  const copy = (text: string) => {
    navigator.clipboard
      ?.writeText(text)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {});
  };

  return (
    <div className="examples">
      <div className="section-title">{t.examplesTitle}</div>
      <div className="example-list">
        {t.examples.map((ex, i) => (
          <button key={i} className="example-card" onClick={() => setOpenIdx(i)}>
            <div className="ex-name">{ex.title}</div>
            <div className="ex-meta">{ex.hint}</div>
          </button>
        ))}
      </div>

      {open && (
        <div className="modal-backdrop" onClick={() => setOpenIdx(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setOpenIdx(null)}>
              ×
            </button>
            <h2>{open.title}</h2>
            <div className="ex-modal-meta">{open.hint}</div>
            <div className="ex-modal-text">{open.text}</div>
            <button className="primary" onClick={() => copy(open.text)}>
              {copied ? t.copied : t.copyRequest}
            </button>
            <div className="hint" style={{ marginTop: 8 }}>
              {t.exampleModalHint}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
