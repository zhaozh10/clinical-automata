import { useT } from "../i18n";

export default function Guidance() {
  const t = useT();
  return (
    <div className="guidance">
      <div className="guidance-title">{t.howItWorks}</div>
      <ol className="guidance-steps">
        {t.steps.map((s, i) => (
          <li key={i}>
            <span className="step-num">{i + 1}</span>
            <span className="step-text">
              <b>{s.title}.</b> {s.body}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
