import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { SessionState, TurnState, WsEvent } from "../types";
import { useT } from "../i18n";

/** The agent points at previews it rendered on the cluster (`/tmp/preview.png`),
 *  which means nothing to a browser — it renders as a broken icon and the alt
 *  text. Route those through the endpoint that serves what the agent showed. */
function AgentImage({ src, alt }: { src?: string; alt?: string }) {
  const local = src && src.startsWith("/") && !src.startsWith("//");
  const url = local ? `/api/image?src=${encodeURIComponent(src!)}` : src;
  return <img className="agent-img" src={url} alt={alt ?? ""} loading="lazy" />;
}

const MD_COMPONENTS = { img: AgentImage };

type ToolItem = { kind: "tool"; tool?: string } | { kind: "result"; result: string };
type Segment = { caption?: string; items: ToolItem[] };

type Block =
  | { type: "user"; text: string }
  | { type: "assistant"; text: string }
  | { type: "toolgroup"; segments: Segment[] }
  | { type: "manual"; text: string }
  | { type: "notice"; text: string }
  | { type: "run"; run: RunAgg };

interface RunAgg {
  device?: string;
  gpuMode?: string;
  iteration?: number;
  epoch?: number;
  ofEpochs?: number;
  valAuc?: number;
  best?: number;
  done: boolean;
  milestones: { iteration: number; val: number; accepted: boolean }[];
  logLines: string[];
  lastMetricName?: string;
}

function fold(events: WsEvent[]): Block[] {
  const blocks: Block[] = [];
  let text: string | null = null;
  let run: RunAgg | null = null;
  let group: { type: "toolgroup"; segments: Segment[] } | null = null;

  const flushText = () => {
    // Text still pending at a turn boundary is a real, user-facing reply.
    if (text !== null && text.trim().length) blocks.push({ type: "assistant", text });
    text = null;
  };
  const closeGroup = () => {
    group = null;
  };
  const ensureGroup = () => {
    if (!group) {
      group = { type: "toolgroup", segments: [] };
      blocks.push(group);
    }
    return group;
  };
  // Current segment; a pending "Let me…" preamble opens a new one as its caption.
  const segmentFor = (g: { type: "toolgroup"; segments: Segment[] }) => {
    if (text && text.trim().length) {
      g.segments.push({ caption: text.trim(), items: [] });
      text = null;
    } else if (g.segments.length === 0) {
      g.segments.push({ items: [] });
    }
    return g.segments[g.segments.length - 1];
  };

  for (const e of events) {
    switch (e.kind) {
      case "user_message":
        flushText();
        closeGroup();
        blocks.push({ type: "user", text: e.text ?? "" });
        break;
      case "assistant_text_delta":
      case "assistant_text":
        // accumulate; keep the group open so consecutive activity stays MERGED
        text = (text ?? "") + (e.text ?? "");
        break;
      case "final_result":
        // End of the agent's whole turn. Prefer the streamed reply if there is
        // one; otherwise fall back to the CLI's own final result text. That text
        // is the last assistant message — the reply that would otherwise be lost
        // when it was consumed as a tool caption or never streamed as a delta
        // (R1/R2). Never both, so the reply is not duplicated.
        if (text !== null && text.trim().length) {
          flushText();
        } else if (e.text && e.text.trim().length) {
          blocks.push({ type: "assistant", text: e.text });
        }
        text = null;
        closeGroup();
        break;
      case "assistant_turn_end":
        // fires between tool rounds within a turn -> ignore, keep merging
        break;
      case "tool_use_start":
        segmentFor(ensureGroup()).items.push({ kind: "tool", tool: e.tool });
        break;
      case "tool_result":
        segmentFor(ensureGroup()).items.push({ kind: "result", result: e.content ?? "" });
        break;
      case "usage_manual":
        flushText();
        closeGroup();
        blocks.push({ type: "manual", text: e.text ?? "" });
        break;
      case "rate_limit":
        if (e.status && e.status !== "allowed")
          blocks.push({ type: "notice", text: `rate limit: ${e.status} (${e.rate_limit_type ?? ""})` });
        break;
      case "run_status": {
        flushText();
        closeGroup();
        if (!run) {
          run = { done: false, milestones: [], logLines: [] };
          blocks.push({ type: "run", run });
        }
        if (e.phase === "start") {
          run.device = e.device;
          run.gpuMode = e.gpu_mode;
        } else if (e.phase === "epoch") {
          run.iteration = e.iteration;
          run.epoch = e.epoch;
          run.ofEpochs = e.of_epochs;
          run.valAuc = e.val_auc;
          run.best = e.best_so_far;
        } else if (e.phase === "iteration_end") {
          run.milestones.push({ iteration: e.iteration ?? 0, val: e.val_auc ?? 0, accepted: !!e.accepted });
        } else if (e.phase === "done") {
          run.done = true;
          run.best = e.best_metric ?? run.best;
        } else if (e.phase === "metric") {
          run.valAuc = e.metric;
          run.lastMetricName = e.metric_name;
          run.best = Math.max(run.best ?? 0, e.metric ?? 0);
        }
        break;
      }
      case "run_log": {
        flushText();
        closeGroup();
        if (!run) {
          run = { done: false, milestones: [], logLines: [] };
          blocks.push({ type: "run", run });
        }
        run.logLines.push(...(e.lines ?? []));
        if (run.logLines.length > 400) run.logLines.splice(0, run.logLines.length - 400);
        break;
      }
      default:
        break;
    }
  }
  flushText();
  return blocks;
}

export default function Transcript({
  events,
  turn,
  onSend,
  sessionState,
  taskSubmitted,
  connected,
}: {
  events: WsEvent[];
  turn: TurnState;
  /** Returns whether the message actually went out (live socket). */
  onSend: (text: string) => boolean;
  sessionState: SessionState;
  taskSubmitted: boolean;
  connected: boolean;
}) {
  const t = useT();
  const blocks = useMemo(() => fold(events), [events]);
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [events]);

  const live = sessionState === "active" || sessionState === "completed";
  // No typing while the socket is down — otherwise the message silently no-ops.
  const canType = turn === "awaiting_user" && live && connected;

  // The long silent stretch the clinician needs warning about is *setup*: from
  // task submission until the first training output appears. After that the
  // optimization loop runs unattended, so the message changes to "you may leave".
  const training = useMemo(
    () => events.some((e) => e.kind === "run_status" || e.kind === "run_log"),
    [events]
  );
  const running = taskSubmitted && sessionState === "active";

  return (
    <div className="transcript-wrap">
      {live && !connected && <div className="leave-banner reconnect">🔌 {t.reconnecting}</div>}
      {running && !training && (
        <div className="leave-banner setup">
          ⏳ <b>{t.bannerSetupTitle}</b> {t.bannerSetupBody}
        </div>
      )}
      {running && training && (
        <div className="leave-banner">
          🧪 <b>{t.bannerOptTitle}</b> {t.bannerOptBody}
        </div>
      )}
      <div className="transcript" ref={scrollRef}>
        <div className="transcript-inner">
          {blocks.length === 0 && <div className="muted hello">{t.transcriptHello}</div>}
          {blocks.map((b, i) => (
            <BlockView key={i} b={b} />
          ))}
          {turn === "agent_busy" && live && <div className="busy">▍ {t.agentBusy}</div>}
        </div>
      </div>

      <div className="chat-encourage">💬 {t.chatEncourage}</div>
      <form
        className="chatbar"
        onSubmit={(e) => {
          e.preventDefault();
          // Clear the box only if the message actually went out — otherwise the
          // user would lose what they typed on a dropped socket.
          if (input.trim() && canType && onSend(input)) setInput("");
        }}
      >
        <input
          placeholder={canType ? t.typePh : t.waitingPh}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={!canType}
        />
        <button className="primary" disabled={!canType || !input.trim()}>
          {t.send}
        </button>
      </form>
    </div>
  );
}

function BlockView({ b }: { b: Block }) {
  const t = useT();
  switch (b.type) {
    case "user":
      return (
        <div className="row user-row">
          <div className="bubble user">{b.text}</div>
        </div>
      );
    case "assistant":
      return (
        <div className="row assistant-row">
          <div className="bubble assistant markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
              {b.text}
            </ReactMarkdown>
          </div>
        </div>
      );
    case "toolgroup":
      return <ToolGroupView segments={b.segments} />;
    case "manual":
      return (
        <div className="manual">
          <div className="manual-title">{t.manualTitle}</div>
          <div className="markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
              {b.text}
            </ReactMarkdown>
          </div>
        </div>
      );
    case "notice":
      return <div className="notice-line">{b.text}</div>;
    case "run":
      return <RunView run={b.run} />;
  }
}

function ToolGroupView({ segments }: { segments: Segment[] }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const stepCount = segments.reduce((n, s) => n + s.items.filter((i) => i.kind === "tool").length, 0);
  // The caption is the agent's own wording ("Let me…") — never translated.
  const lastCaption = [...segments].reverse().find((s) => s.caption)?.caption;
  return (
    <div className="toolgroup">
      <button className="toolgroup-head" onClick={() => setOpen((o) => !o)}>
        <span className="chev">{open ? "▾" : "▸"}</span>
        <span className="tool-badge sm">⚙</span>
        <span className="toolgroup-caption">{lastCaption ?? t.agentActivity}</span>
        <span className="muted tg-count">{t.stepCount(stepCount)}</span>
      </button>
      {open && (
        <div className="toolgroup-body">
          {segments.map((seg, si) => (
            <div key={si} className="tg-segment">
              {seg.caption && <div className="tg-seg-caption">{seg.caption}</div>}
              {seg.items.map((it, i) =>
                it.kind === "tool" ? (
                  <div key={i} className="tg-tool">
                    <span className="tool-badge sm">{it.tool}</span>
                  </div>
                ) : (
                  <pre key={i} className="tg-result">{truncate(it.result)}</pre>
                )
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function truncate(s: string, n = 1200): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function RunView({ run }: { run: RunAgg }) {
  const t = useT();
  return (
    <div className="run">
      <div className="run-head">
        <span className="tool-badge gpu">GPU {run.device ?? ""}</span>
        <span className="muted">{run.gpuMode === "pseudo" ? t.runPseudo : t.runTraining}</span>
      </div>
      {!run.done ? (
        <div className="run-live">
          {run.iteration != null && <>{t.runIteration} {run.iteration} · </>}
          {run.epoch != null && <>{t.runEpoch} {run.epoch}/{run.ofEpochs ?? "-"} · </>}
          {run.valAuc != null ? (
            <>
              {/* metric name comes from the run log — a proper noun, kept as-is */}
              {run.lastMetricName ?? "val_auc"} <b>{run.valAuc.toFixed(4)}</b>
              {run.best != null && <> · {t.runBest} <b>{run.best.toFixed(4)}</b></>}
            </>
          ) : (
            <span className="muted">{t.runRunning}</span>
          )}
        </div>
      ) : (
        <div className="run-live done">
          {t.runCompleted} · {t.runBest} {run.best?.toFixed(4)}
        </div>
      )}
      <div className="milestones">
        {run.milestones.map((m, i) => (
          <span key={i} className={`ms ${m.accepted ? "ok" : "no"}`}>
            it{m.iteration}:{m.val.toFixed(3)}
          </span>
        ))}
      </div>
      {run.logLines.length > 0 && (
        <pre className="run-logtail">{run.logLines.slice(-24).join("\n")}</pre>
      )}
    </div>
  );
}
