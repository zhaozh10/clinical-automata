import type { DatasetCard } from "./types";

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  async me(): Promise<boolean> {
    const r = await fetch("/api/me", { credentials: "include" });
    return (await j<{ authenticated: boolean }>(r)).authenticated;
  },
  async login(code: string, email: string, lang: string): Promise<void> {
    const r = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ code, email, lang }),
    });
    await j(r);
  },
  async datasets(): Promise<DatasetCard[]> {
    const r = await fetch("/api/datasets", { credentials: "include" });
    return (await j<{ datasets: DatasetCard[] }>(r)).datasets;
  },
  async status(): Promise<boolean> {
    const r = await fetch("/api/session/status", { credentials: "include" });
    return (await j<{ occupied: boolean }>(r)).occupied;
  },
  /** The session owned by my code, if any (identity-based rejoin). */
  async mine(): Promise<{
    active: boolean;
    session_id?: string;
    state?: string;
    task_submitted?: boolean;
    task_text?: string;
  }> {
    const r = await fetch("/api/session/mine", { credentials: "include" });
    return j(r);
  },
  /** Returns session_id, or null if the demo is already occupied. */
  async startSession(): Promise<string | null> {
    const r = await fetch("/api/session/start", { method: "POST", credentials: "include" });
    if (r.status === 409) return null;
    return (await j<{ session_id: string }>(r)).session_id;
  },
  async complete(sid: string): Promise<void> {
    await fetch(`/api/session/${sid}/complete`, { method: "POST", credentials: "include" });
  },
  /** Clears the identity cookie, so the next visitor is not seen as this user. */
  async logout(): Promise<void> {
    await fetch("/api/logout", { method: "POST", credentials: "include" });
  },
};

export function openWs(sid: string): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${proto}://${location.host}/ws/${sid}`);
}
