"""Police1 seeds — Utility | mean rep side by side (thesis half-page size)."""
from __future__ import annotations

import os

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

from harvest_socialjax.wandb_sweep_viz.fetch import fetch_run_history
from harvest_socialjax.wandb_sweep_viz.plot import (
    SWEEP_COLORS,
    THESIS_FONT,
    _apply_axis_formatters,
)
from harvest_socialjax.wandb_sweep_viz.series import EPISODE_XLABEL, metric_ylabel
from harvest_socialjax.wandb_sweep_viz.plot_runs import load_plot_runs, seed0 as plot_seed0

PROJ = f"{os.environ.get('WANDB_ENTITY', 'YOUR_ENTITY')}/{os.environ.get('WANDB_PROJECT', 'harvest_ippo_rep')}"
_PLOT_RUNS = load_plot_runs()
GROUP = "ippo_v2_w2_m1_scratch_police1_no_peer_tdur25_50M"
SEED0_ID = plot_seed0(_PLOT_RUNS, "police")
METRICS = ["mean_reward", "mean_rep"]
YLABELS = {
    "mean_reward": "Utility (U)",
    "mean_rep": metric_ylabel("mean_rep"),
}
OUT = Path("harvest_socialjax/plots/sweep_comparisons/police1_seed_divergence_small")


def main() -> None:
    api = wandb.Api(timeout=180)
    runs = list(api.runs(PROJ, filters={"group": GROUP}, order="-created_at"))
    by_seed: dict[int, object] = {}
    for r in runs:
        seed = int((r.config or {}).get("SEED", -1))
        if seed < 0:
            continue
        # Prefer finished; keep first seen (newest) if duplicate.
        if seed not in by_seed or (r.state == "finished" and by_seed[seed].state != "finished"):
            by_seed[seed] = r
    # Ensure seed0 reference is present when configured.
    if 0 not in by_seed and SEED0_ID:
        by_seed[0] = api.run(f"{PROJ}/{SEED0_ID}")

    seeds = sorted(by_seed)
    print(f"seeds={seeds}")
    series: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    finals: dict[int, float] = {}
    for seed in seeds:
        r = by_seed[seed]
        h = fetch_run_history(
            r,
            metrics=METRICS,
            x_axis="episode_count",
            x_fallbacks=("update", "env_steps", "_step"),
            full=False,
        )
        x_col = h.attrs.get("x_col", "episode_count") if h is not None else "episode_count"
        series[seed] = {}
        for m in METRICS:
            if h is None or h.empty or m not in h.columns:
                print(f"  seed{seed} {r.id}: missing {m}")
                continue
            df = h[[x_col, m]].dropna()
            x = df[x_col].to_numpy(dtype=float)
            y = df[m].to_numpy(dtype=float)
            series[seed][m] = (x, y)
            if m == "mean_reward" and len(y):
                finals[seed] = float(y[-1])
        print(
            f"  seed{seed} id={r.id} state={r.state} "
            f"final_R={finals.get(seed, float('nan')):.1f}"
        )

    lo_seed = min(finals, key=finals.get)
    hi_seed = max(finals, key=finals.get)
    focus = {lo_seed, hi_seed}
    print(f"focus seeds: low=s{lo_seed} high=s{hi_seed}")

    # Same panel height as other thesis comparison plots (~8.5×5.2 single).
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        # Background seeds first (shaded).
        for i, seed in enumerate(seeds):
            if seed in focus or metric not in series[seed]:
                continue
            x, y = series[seed][metric]
            mask = x <= 50_000
            ax.plot(
                x[mask],
                y[mask],
                color=SWEEP_COLORS[i % len(SWEEP_COLORS)],
                lw=1.6,
                alpha=0.22,
                zorder=1,
            )
        for i, seed in enumerate(seeds):
            if seed not in focus or metric not in series[seed]:
                continue
            x, y = series[seed][metric]
            mask = x <= 50_000
            ax.plot(
                x[mask],
                y[mask],
                color=SWEEP_COLORS[i % len(SWEEP_COLORS)],
                lw=2.4,
                alpha=0.95,
                zorder=3,
            )
        ax.set_xlim(0, 50_000)
        ax.set_xlabel(EPISODE_XLABEL, fontsize=THESIS_FONT["label"])
        ax.set_ylabel(YLABELS[metric], fontsize=THESIS_FONT["label"])
        ax.tick_params(labelsize=THESIS_FONT["tick"])
        ax.grid(True, alpha=0.25)
        _apply_axis_formatters(ax)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"mean_reward_mean_rep.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
