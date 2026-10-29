"""Integration smoke test for RealClaudeRunner (spends a little API $).

Verifies the bidirectional stream-json path: spawn -> send_user -> receive
normalized streamed events -> final_result. Plain-chat, default permissions,
scratch cwd (does NOT touch the harness).
"""
import asyncio
import os

os.environ["CAWEBUI_claude_mode"] = "real"
os.environ["CAWEBUI_permission_mode"] = "default"

from app.config import get_settings  # noqa: E402
from app.claude_runner import RealClaudeRunner  # noqa: E402


async def main():
    settings = get_settings()
    events = []
    done = asyncio.Event()

    async def emit(evt):
        events.append(evt)
        kind = evt.get("kind")
        if kind == "assistant_text_delta":
            print(evt["text"], end="", flush=True)
        elif kind in ("final_result", "process_exit"):
            print(f"\n[{kind}] {evt}")
            if kind == "final_result":
                done.set()

    runner = RealClaudeRunner(settings, emit)
    await runner.start()
    await runner.send_user("Reply with exactly one short sentence: what is a chest radiograph?")
    try:
        await asyncio.wait_for(done.wait(), timeout=90)
    finally:
        await runner.close()

    kinds = [e.get("kind") for e in events]
    print("\n--- event kinds:", kinds)
    assert "system_status" in kinds, "no system init event"
    assert any(k in ("assistant_text_delta", "assistant_text") for k in kinds), "no assistant text"
    assert "final_result" in kinds, "no final result"
    print("REAL CLAUDE BIDIRECTIONAL: OK")


if __name__ == "__main__":
    asyncio.run(main())
