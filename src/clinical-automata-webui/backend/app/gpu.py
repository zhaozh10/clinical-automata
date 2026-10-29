"""Pseudo-GPU / training stub.

during framework development we do NOT request a real GPU from
Slurm. `GPU_MODE=pseudo` simulates the iterative optimization loop and emits
`run_status` events shaped like the real ones (which, in the real system,
will be parsed from projects/YYYYMMDD_HH/run{i}/run.log).

To wire real training later, implement `tail_real_training` and switch on
settings.gpu_mode; the event contract to the frontend stays identical.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from .config import Settings

EmitFn = Callable[[dict], Awaitable[None]]


async def simulate_training(
    settings: Settings,
    emit: EmitFn,
    iterations: int = 3,
    epochs: int = 3,
    step_delay: float = 0.15,
) -> None:
    """Emit a believable iterative-optimization trajectory (no real compute)."""
    best_metric = 0.58
    await emit({
        "kind": "run_status", "phase": "start",
        "device": settings.pseudo_gpu_id, "gpu_mode": settings.gpu_mode,
        "max_loops": iterations, "primary_metric": "val_auc",
    })
    for it in range(1, iterations + 1):
        await emit({"kind": "run_status", "phase": "iteration_start", "iteration": it, "of": iterations})
        cur = best_metric
        for ep in range(1, epochs + 1):
            cur = round(min(0.97, cur + 0.03 + 0.01 * ep), 4)
            await emit({
                "kind": "run_status", "phase": "epoch",
                "iteration": it, "epoch": ep, "of_epochs": epochs,
                "val_auc": cur, "best_so_far": max(best_metric, cur),
            })
            await asyncio.sleep(step_delay)
        improved = cur > best_metric
        if improved:
            best_metric = cur
        await emit({
            "kind": "run_status", "phase": "iteration_end",
            "iteration": it, "val_auc": cur, "best_so_far": best_metric,
            "accepted": improved,
        })
    await emit({"kind": "run_status", "phase": "done", "best_metric": best_metric})


async def tail_real_training(settings: Settings, emit: EmitFn, run_log_path: str) -> None:  # pragma: no cover
    """Placeholder for the real path: tail run.log and emit run_status.

    Not implemented in the skeleton — GPU_MODE=pseudo is enforced elsewhere.
    """
    raise NotImplementedError("real GPU training not wired in the skeleton")
