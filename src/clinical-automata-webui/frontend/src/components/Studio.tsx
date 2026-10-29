import { useCallback, useEffect, useRef, useState } from "react";
import { api, openWs } from "../api";
import type { DatasetCard, SessionState, TurnState, WsEvent } from "../types";
import { useT } from "../i18n";
import DatasetPanel from "./DatasetPanel";
import Transcript from "./Transcript";
import TaskBox from "./TaskBox";
import Guidance from "./Guidance";

type Phase = "preview" | "active";

export default function Studio({ onEnded }: { onEnded: () => void }) {
  const t = useT();
  const [phase, setPhase] = useState<Phase>("preview");
  const [datasets, setDatasets] = useState<DatasetCard[]>([]);
  const [occupied, setOccupied] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const [events, setEvents] = useState<WsEvent[]>([]);
  const [sessionState, setSessionState] = useState<SessionState>("onboarding");
  const [turn, setTurn] = useState<TurnState>("agent_busy");
  const [taskSubmitted, setTaskSubmitted] = useState(false);
  // Kept out of TaskBox's own state: the box is remounted on every rejoin.
  const [taskText, setTaskText] = useState("");
  // Live-connection health, surfaced to the UI so a dropped socket doesn't look
  // like a working one (which is what let a task submit vanish silently).
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  // Highest event seq applied. On reconnect the server replays the whole history
  // (replay_to), so we drop anything we've already seen to avoid duplicates.
  const lastSeqRef = useRef(0);

  useEffect(() => {
    api.datasets().then(setDatasets).catch(() => {});
    const loadStatus = () => {
      api.status().then(setOccupied).catch(() => {});
    };
    // Identity-based rejoin: if my code already owns a live session, go
    // straight back into it (works across browsers/devices).
    api
      .mine()
      .then((r) => {
        if (r.active && r.session_id) {
          setSessionId(r.session_id);
          setSessionState(r.state as SessionState);
          setTaskSubmitted(!!r.task_submitted);
          setTaskText(r.task_text ?? "");
          setPhase("active");
        } else {
          loadStatus();
        }
      })
      .catch(loadStatus);
  }, []);

  // --- websocket lifecycle (active phase) --------------------------------
  // Auto-reconnecting: a transient drop (network blip, tunnel bounce) must not
  // freeze the page or silently swallow sends. On every (re)connect the server
  // replays the full history; we dedupe by seq so the transcript isn't doubled.
  useEffect(() => {
    if (phase !== "active" || !sessionId) return;
    // New session (not a reconnect): start the transcript and seq cursor fresh.
    lastSeqRef.current = 0;
    setEvents([]);

    let stopped = false;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      const ws = openWs(sessionId);
      wsRef.current = ws;
      ws.onopen = () => {
        retry = 0;
        setConnected(true);
      };
      ws.onmessage = (m) => {
        const evt = JSON.parse(m.data) as WsEvent;
        // Already applied (replayed on reconnect) -> ignore entirely.
        if (evt.seq && evt.seq <= lastSeqRef.current) return;
        if (evt.seq) lastSeqRef.current = evt.seq;
        if (evt.kind === "session_state") setSessionState(evt.state as SessionState);
        else if (evt.kind === "turn_state") setTurn(evt.state as TurnState);
        else if (evt.kind === "task_submitted") {
          setTaskSubmitted(true);
          if (evt.text) setTaskText(evt.text);
        }
        if (evt.seq > 0) setEvents((prev) => [...prev, evt]);
      };
      ws.onerror = () => {
        try {
          ws.close();
        } catch {
          /* ignore */
        }
      };
      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null;
        setConnected(false);
        if (stopped) return; // intentional close (unmount / session change)
        const delay = Math.min(1000 * 2 ** retry, 10000); // backoff, capped at 10s
        retry += 1;
        timer = setTimeout(connect, delay);
      };
    };
    connect();

    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
      const ws = wsRef.current;
      wsRef.current = null;
      if (ws) {
        ws.onclose = null; // don't let the teardown close trigger a reconnect
        ws.close();
      }
    };
  }, [phase, sessionId]);

  // --- actions -----------------------------------------------------------
  const startUsing = useCallback(async () => {
    const sid = await api.startSession();
    if (!sid) {
      setOccupied(true);
      return;
    }
    setSessionId(sid);
    setPhase("active"); // email + language were captured at login; go straight in
  }, []);

  // Both return whether the message actually went out: only send on a live
  // socket, never pretend on a dead one (that is what lost tasks silently).
  const sendOn = (payload: object): boolean => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    ws.send(JSON.stringify(payload));
    return true;
  };
  const sendChat = useCallback((text: string): boolean => {
    return sendOn({ type: "user_message", text });
  }, []);

  const submitTask = useCallback((text: string): boolean => {
    return sendOn({ type: "task_submit", text });
  }, []);

  // Ending a session releases the lock and drops the identity cookie, so the
  // demo returns all the way to the access-code page — not to the preview,
  // which would still be logged in as the previous clinician.
  const complete = useCallback(async () => {
    wsRef.current?.close();
    if (sessionId) await api.complete(sessionId);
    await api.logout();
    onEnded();
  }, [sessionId, onEnded]);

  const stateLabel: Record<string, string> = {
    active: t.stateActive,
    completed: t.stateCompleted,
    interrupted: t.stateInterrupted,
  };

  // --- render ------------------------------------------------------------
  return (
    <div className="studio">
      <header className="topbar">
        <div className="brand">Clinical-Automata</div>
        <div className="subtitle">{t.subtitle}</div>
        {phase === "active" && (
          <div className="right">
            <span className={`pill ${sessionState}`}>{stateLabel[sessionState] ?? sessionState}</span>
            <button className="ghost" onClick={complete}>
              {sessionState === "completed" ? t.completeBtn : t.endSession}
            </button>
          </div>
        )}
      </header>

      <main className="panes">
        <DatasetPanel
          datasets={datasets}
          taskBox={
            phase === "active" ? (
              <TaskBox onSubmit={submitTask} disabled={taskSubmitted} submitted={taskText} />
            ) : null
          }
        />

        <section className="center-pane">
          {phase === "preview" && (
            <div className="preview">
              <pre className="welcome">{t.welcome}</pre>
              <Guidance />
              {occupied ? (
                <div className="occupied">{t.occupied}</div>
              ) : (
                <button className="primary big" onClick={startUsing}>
                  {t.startUsing}
                </button>
              )}
            </div>
          )}
          {phase === "active" && (
            <Transcript
              events={events}
              turn={turn}
              onSend={sendChat}
              sessionState={sessionState}
              taskSubmitted={taskSubmitted}
              connected={connected}
            />
          )}
        </section>
      </main>
    </div>
  );
}
