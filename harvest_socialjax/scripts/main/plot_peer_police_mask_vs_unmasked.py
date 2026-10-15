"""Peer ±5 and Police1: masked vs unmasked (8-seed mean±std), one plot.

Color grouping (same hue per family; masked = dashed):
  Peer ±5 / Peer masked   — meadow green
  Police / Police masked  — dark blue

  python harvest_socialjax/scripts/main/plot_peer_police_mask_vs_unmasked.py
"""
from __future__ import annotations

import os

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

from harvest_socialjax.wandb_sweep_viz.fetch import fetch_run_history
from harvest_socialjax.wandb_sweep_viz.plot import (
    ARM_COLORS,
    ARM_LABELS,
    THESIS_FONT,
    _apply_axis_formatters,
)
from harvest_socialjax.wandb_sweep_viz.series import (
    EPISODE_XLABEL,
    aggregate_across_runs,
    extract_series_from_history,
    metric_ylabel,
)
from harvest_socialjax.wandb_sweep_viz.plot_runs import load_plot_runs, seed0 as plot_seed0

PROJ = f"{os.environ.get('WANDB_ENTITY', 'YOUR_ENTITY')}/{os.environ.get('WANDB_PROJECT', 'harvest_ippo_rep')}"
_PLOT_RUNS = load_plot_runs()
METRICS = ["mean_reward", "peace", "sustainability", "equality"]
X_MAX = 50_000
OUT = Path("harvest_socialjax/plots/sweep_comparisons/peer_police_mask_vs_unmasked")

ARMS = [
    (
        plot_seed0(_PLOT_RUNS, "peer"),
        "ippo_v2_w2_m1_scratch_br5_g5_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        ARM_LABELS["peer"],
        ARM_COLORS["peer"],
        "-",
    ),
    (
        plot_seed0(_PLOT_RUNS, "peer_masked"),
        "ippo_v2_w2_m1_scratch_br5_g5_mask_rep_tdur25_ep4_mb25_g99_ent_floor0005_mask_rep_50kep",
        ARM_LABELS["peer_masked"],
        ARM_COLORS["peer"],
        "--",
    ),
    (
        plot_seed0(_PLOT_RUNS, "police"),
        "ippo_v2_w2_m1_scratch_police1_no_peer_tdur25_50M",
        ARM_LABELS["police"],
        ARM_COLORS["police"],
        "-",
    ),
    (
        plot_seed0(_PLOT_RUNS, "police_masked"),
        "ippo_v2_w2_m1_scratch_police1_no_peer_mask_rep_tdur25_50M",
        ARM_LABELS["police_masked"],
        ARM_COLORS["police"],
        "--",
    ),
]

api = wandb.Api(timeout=180)


def get_hist(run):
    return fetch_run_history(
        run,
        metrics=METRICS,
        x_axis="episode_count",
        x_fallbacks=("update", "env_steps", "_step"),
        full=False,
    )


def load_arm(seed0_id: str | None, group: str, finished_only: bool = True):
    by_seed: dict = {}
    if seed0_id:
        r0 = api.run(f"{PROJ}/{seed0_id}")
        by_seed[int((r0.config or {}).get("SEED", 0))] = r0
        print(f"  seed0 {seed0_id}: state={r0.state}")
    for r in api.runs(PROJ, filters={"group": group}, order="created_at"):
        seed = (r.config or {}).get("SEED")
        if seed is None:
            continue
        seed = int(seed)
        if finished_only and r.state != "finished":
            print(f"  skip s={seed} {r.id} state={r.state}")
            continue
        if seed0_id and seed == 0 and r.id != seed0_id:
            continue
        by_seed[seed] = r
    runs = [by_seed[s] for s in sorted(by_seed)]
    print(f"  seeds={sorted(by_seed)} n={len(runs)}")
    return [(r, get_hist(r)) for r in runs]


def agg_runs(run_hists, metric, label, mode="independent"):
    series_list = []
    for r, h in run_hists:
        if h is None or h.empty or metric not in h.columns:
            continue
        x_col = h.attrs.get("x_col", "episode_count")
        s = extract_series_from_history(
            h,
            metric=metric,
            x_col=x_col,
            label=label,
            mode=mode,
            run_id=r.id,
            meta={"SEED": (r.config or {}).get("SEED")},
        )
        if s is not None:
            series_list.append(s)
    if not series_list:
        return None
    return aggregate_across_runs(series_list, metric=metric, label=label, mode=mode)


def clip_xy(x, y, xmax=X_MAX):
    m = np.asarray(x, float) <= float(xmax)
    return np.asarray(x, float)[m], np.asarray(y, float)[m]


def plot_agg(ax, agg, color, label, *, ls="-", lw=2.5, z=4):
    if agg is None:
        return
    x, mean = clip_xy(agg.x, agg.mean)
    _, std = clip_xy(agg.x, agg.std)
    if len(x) == 0:
        return
    n = int(np.nanmax(agg.n)) if getattr(agg, "n", None) is not None else 1
    lab = label
    if n > 1:
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=z - 1,
        )
    ax.plot(
        x,
        mean,
        color=color,
        linewidth=lw,
        linestyle=ls,
        label=lab,
        alpha=0.95,
        zorder=z,
    )


def main() -> None:
    arms = []
    for seed0, group, label, color, ls in ARMS:
        print(f"\nloading {label}...")
        arms.append((label, load_arm(seed0, group), color, ls))

    OUT.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for label, run_hists, color, ls in arms:
            plot_agg(ax, agg_runs(run_hists, metric, label), color, label, ls=ls)
        ax.set_xlabel(EPISODE_XLABEL, fontsize=THESIS_FONT["label"])
        ax.set_ylabel(metric_ylabel(metric), fontsize=THESIS_FONT["label"])
        ax.set_xlim(0, X_MAX)
        ax.tick_params(axis="both", labelsize=THESIS_FONT["tick"])
        ax.grid(True, alpha=0.25)
        _apply_axis_formatters(ax)
        ax.legend(
            loc="best",
            fontsize=THESIS_FONT["legend"],
            framealpha=0.9,
            handlelength=2.8,
        )
        fig.tight_layout()
        out = OUT / f"{metric}.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
