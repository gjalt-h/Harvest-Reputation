"""Peer reward + Good immunity: zap attempts and mean zapped reputation.

Arms (mean±std over finished seeds):
  - Peer reward (±5) — br5_g5 g99 group (+ optional seed0)
  - Good immunity — br5_gimm g99 group (+ optional seed0)

Metrics:
  - zap_attempts_per_episode (from zap_attempt_rate)
  - zap_hit_rate
  - zap_tagged_rep_mean  (harvester shooters; hit-weighted victim rep)

  python harvest_socialjax/scripts/main/plot_peer_gimm_zap_behavior.py
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
    DERIVED_FROM_RATE,
    EPISODE_XLABEL,
    aggregate_across_runs,
    extract_series_from_history,
    metric_ylabel,
    resolve_history_metrics,
)
from harvest_socialjax.wandb_sweep_viz.plot_runs import load_plot_runs, seed0 as plot_seed0

PROJ = f"{os.environ.get('WANDB_ENTITY', 'YOUR_ENTITY')}/{os.environ.get('WANDB_PROJECT', 'harvest_ippo_rep')}"
_PLOT_RUNS = load_plot_runs()
X_MAX = 50_000
OUT = Path("harvest_socialjax/plots/sweep_comparisons/peer_gimm_zap_behavior")

METRICS = [
    "zap_attempts_per_episode",
    "zap_hit_rate",
    "zap_tagged_rep_mean",
]

METRIC_FILES = {
    "zap_attempts_per_episode": "zap_attempts",
    "zap_hit_rate": "zap_hit_rate",
    "zap_tagged_rep_mean": "zap_tagged_rep_mean",
}

ARMS = [
    (
        ARM_LABELS["peer"],
        "ippo_v2_w2_m1_scratch_br5_g5_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        plot_seed0(_PLOT_RUNS, "peer"),
        ARM_COLORS["peer"],
    ),
    (
        ARM_LABELS["gimm"],
        "ippo_v2_w2_m1_scratch_br5_gimm_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        plot_seed0(_PLOT_RUNS, "gimm"),
        ARM_COLORS["gimm"],
    ),
]

api = wandb.Api(timeout=180)


def get_hist(run):
    return fetch_run_history(
        run,
        metrics=resolve_history_metrics(METRICS),
        x_axis="episode_count",
        x_fallbacks=("update", "env_steps", "_step"),
        full=False,
    )


def load_sweep(group: str, seed0_id: str | None):
    by_seed: dict = {}
    if seed0_id:
        r0 = api.run(f"{PROJ}/{seed0_id}")
        by_seed[int((r0.config or {}).get("SEED", 0))] = r0
        print(f"  seed0 {seed0_id}: state={r0.state}")
    for r in api.runs(PROJ, filters={"group": group}, order="created_at"):
        if r.state != "finished":
            continue
        seed = (r.config or {}).get("SEED")
        if seed is None:
            continue
        seed = int(seed)
        if seed == 0 and seed0_id:
            continue
        by_seed[seed] = r
    runs = [by_seed[s] for s in sorted(by_seed)]
    print(f"  seeds={sorted(by_seed)} n={len(runs)}")
    return [(r, get_hist(r)) for r in runs]


def agg_runs(pairs, metric: str, label: str):
    series_list = []
    source = DERIVED_FROM_RATE.get(metric, metric)
    for r, h in pairs:
        if h is None or h.empty or source not in h.columns:
            print(f"    missing {source} on {r.id}")
            continue
        x_col = h.attrs.get("x_col", "episode_count")
        cfg = dict(r.config or {})
        s = extract_series_from_history(
            h,
            metric=metric,
            x_col=x_col,
            label=label,
            mode="independent",
            run_id=r.id,
            meta=cfg,
        )
        if s is not None:
            series_list.append(s)
    if not series_list:
        return None
    return aggregate_across_runs(
        series_list, metric=metric, label=label, mode="independent"
    )


def clip_xy(x, y, xmax=X_MAX):
    m = np.asarray(x, float) <= float(xmax)
    return np.asarray(x, float)[m], np.asarray(y, float)[m]


def plot_agg(ax, agg, color, label):
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
            x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0
        )
    ax.plot(x, mean, color=color, linewidth=2.4, label=lab, alpha=0.95)


def main() -> None:
    loaded = []
    for label, group, seed0, color in ARMS:
        print(f"\nloading {label}...")
        try:
            pairs = load_sweep(group, seed0)
        except Exception as exc:
            print(f"  skip {label}: {exc}")
            continue
        loaded.append((label, color, pairs))

    label_fs = THESIS_FONT["label"]
    tick_fs = THESIS_FONT["tick"]
    legend_fs = THESIS_FONT["legend"]
    OUT.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for label, color, pairs in loaded:
            plot_agg(ax, agg_runs(pairs, metric, label), color, label)
        ax.set_xlabel(EPISODE_XLABEL, fontsize=label_fs)
        ax.set_ylabel(metric_ylabel(metric), fontsize=label_fs)
        ax.set_xlim(0, X_MAX)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.25)
        _apply_axis_formatters(ax)
        ax.legend(
            loc="best",
            fontsize=legend_fs,
            framealpha=0.9,
            handlelength=2.4,
        )
        fig.tight_layout()
        out = OUT / f"{METRIC_FILES[metric]}.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
