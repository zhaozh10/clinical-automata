import { useState, type ReactNode } from "react";
import type { DatasetCard } from "../types";
import { useT } from "../i18n";
import Examples from "./Examples";

/** Only the *labels* around the dataset are localized. The dataset's own fields
 *  (display_name, dataset, modality, annotations, label_categories) are proper
 *  nouns from datasets.yaml and are shown verbatim in every language. */
export default function DatasetPanel({
  datasets,
  taskBox,
}: {
  datasets: DatasetCard[];
  taskBox: ReactNode;
}) {
  const t = useT();
  const [open, setOpen] = useState<DatasetCard | null>(null);

  return (
    <aside className="left-pane">
      <div className="section-title">{t.datasetsTitle}</div>
      <div className="dataset-list">
        {datasets.map((d) => (
          <button key={d.key} className="dataset-card" onClick={() => setOpen(d)}>
            <div className="dc-name">{d.display_name}</div>
            <div className="dc-meta">
              {d.dataset} · {d.modality}
            </div>
          </button>
        ))}
      </div>

      {taskBox && <div className="taskbox-slot">{taskBox}</div>}

      <Examples />

      {open && (
        <div className="modal-backdrop" onClick={() => setOpen(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setOpen(null)}>
              ×
            </button>
            <h2>{open.display_name}</h2>
            <div className="kv">
              <span>{t.dsDataset}</span>
              <b>{open.dataset}</b>
            </div>
            <div className="kv">
              <span>{t.dsModality}</span>
              <b>{open.modality}</b>
            </div>
            {open.num_images != null && (
              <div className="kv">
                <span>{t.dsImages}</span>
                <b>{open.num_images.toLocaleString()}</b>
              </div>
            )}
            {open.license && (
              <div className="kv">
                <span>{t.dsLicense}</span>
                <b>{open.license}</b>
              </div>
            )}

            <div className="modal-section">{t.dsAnnotations}</div>
            <ul className="ds-list">
              {open.annotations.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>

            <div className="modal-section">{t.dsCategories}</div>
            <div className="cat-chips">
              {open.label_categories.map((c) => (
                <span key={c} className="cat-chip">
                  {c}
                </span>
              ))}
            </div>

            {open.description && <p className="dc-desc">{open.description}</p>}
            {open.source_url && (
              <a className="source" href={open.source_url} target="_blank" rel="noreferrer">
                {t.dsSource}
              </a>
            )}
          </div>
        </div>
      )}
    </aside>
  );
}
