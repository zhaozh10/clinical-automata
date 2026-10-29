export interface DatasetCard {
  key: string;
  display_name: string;
  dataset: string;
  modality: string;
  annotations: string[];
  label_categories: string[];
  num_images?: number | null;
  license?: string;
  source_url?: string;
  description?: string;
}

export type SessionState = "onboarding" | "active" | "completed" | "interrupted";
export type TurnState = "awaiting_user" | "agent_busy";

// Normalized event coming over the websocket (backend events.py / session.py).
export interface WsEvent {
  session_id: string;
  seq: number;
  ts: number;
  kind: string;
  // payload fields (partial, depend on kind)
  text?: string;
  state?: string;
  status?: string;
  tool?: string;
  input?: string;
  content?: string;
  phase?: string;
  iteration?: number;
  epoch?: number;
  of?: number;
  of_epochs?: number;
  val_auc?: number;
  best_so_far?: number;
  best_metric?: number;
  max_loops?: number;
  // real run.log tailing
  run?: string;
  lines?: string[];
  metric?: number;
  metric_name?: string;
  raw?: string;
  device?: string;
  gpu_mode?: string;
  accepted?: boolean;
  subtype?: string;
  is_error?: boolean;
  returncode?: number;
  model?: string;
  cli_version?: string;
  resets_at?: number;
  rate_limit_type?: string;
}

export interface OnboardingData {
  age: string;
  experience: string;
  language: string;
  email: string;
}
