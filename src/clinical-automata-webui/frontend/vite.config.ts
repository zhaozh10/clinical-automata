import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { existsSync, readFileSync } from "node:fs";

// Site settings live in src/config.env (KEY=VALUE). Only the keys listed in
// `define` below reach the browser bundle; an env var of the same name wins.
const SITE_CONFIG = new URL("../../config.env", import.meta.url);
function siteConfig(): Record<string, string> {
  if (!existsSync(SITE_CONFIG)) return {};
  const out: Record<string, string> = {};
  for (const raw of readFileSync(SITE_CONFIG, "utf-8").split("\n")) {
    const line = raw.trim();
    const eq = line.indexOf("=");
    if (!line || line.startsWith("#") || eq < 0) continue;
    out[line.slice(0, eq).trim()] = line.slice(eq + 1).trim().replace(/^["']|["']$/g, "");
  }
  return out;
}
const site = siteConfig();
const contactEmail = process.env.VITE_CONTACT_EMAIL ?? site.VITE_CONTACT_EMAIL ?? "your-email@example.com";

// Dev server proxies API + websocket to the FastAPI backend on :8000 so the
// browser talks to a single origin (cookies + WS work without CORS fuss).
export default defineConfig({
  plugins: [react()],
  define: {
    "import.meta.env.VITE_CONTACT_EMAIL": JSON.stringify(contactEmail),
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
