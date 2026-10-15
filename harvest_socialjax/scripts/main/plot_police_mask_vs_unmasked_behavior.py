"""Police masked vs unmasked: zap behavior + Utility + total peace.

Metrics:
  - police zap attempts / hit rate / tagged-rep mean / police reward
  - mean_reward (Utility) / total_peace — with γ=0.995 Individual/Shared baselines
    (matched to police learning: ENT 0.01 flat)

Style: same color; solid=Police, dashed=Police mask (matches other mask plots).
"""
from __future__ import annotations

import os

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from harvest_socialjax.wandb_sweep_viz.fetch import fetch_run_history
from harvest_socialjax.wandb_sweep_viz.plot import (
    ARM_COLORS,
    ARM_LABELS,
    BASELINE_COLOR,
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
N_HARVESTERS = 4
X_MAX = 50_000
OUT = Path("harvest_socialjax/plots/sweep_comparisons/police_mask_vs_unmasked_behavior")

# Matched to police: γ=0.995, ENT 0.01 flat (not g99 twin).
BASELINE_GROUPS = {
    "independent": "ippo_v2_w2_m1_no_peer_indep_50kep",
    "shared": "ippo_v2_w2_m1_no_peer_shared_50kep",
}
BASELINE_METRICS = frozenset({"mean_reward", "total_peace"})

ZAP_METRICS = [
    "enforcement/police_zap_attempts_per_episode",
    "enforcement/police_zap_tagged_rep_mean",
    "enforcement/police_reward",
    "enforcement/police_zap_hit_rate",
]
TOC_METRICS = ["mean_reward", "total_peace"]
FETCH_METRICS = resolve_history_metrics(ZAP_METRICS) + [
    "mean_reward",
    "peace",
    "system_peace",
]

METRIC_FILES = {
    "enforcement/police_zap_attempts_per_episode": "police_zap_attempts",
    "enforcement/police_zap_tagged_rep_mean": "police_zap_tagged_rep_mean",
    "enforcement/police_reward": "police_reward",
    "enforcement/police_zap_hit_rate": "police_zap_hit_rate",
    "mean_reward": "mean_reward",
    "total_peace": "total_peace",
}

METRIC_YLABEL_OVERRIDE = {
    "total_peace": "Peace (P)",
}

ARMS = [
    (
        ARM_LABELS["police"],
        "ippo_v2_w2_m1_scratch_police1_no_peer_tdur25_50M",
        plot_seed0(_PLOT_RUNS, "police"),
        ARM_COLORS["police"],
        "-",
    ),
    (
        ARM_LABELS["police_masked"],
        "ippo_v2_w2_m1_scratch_police1_no_peer_mask_rep_tdur25_50M",
        plot_seed0(_PLOT_RUNS, "police_masked"),
        ARM_COLORS["police"],
        "--",
    ),
]

api = wandb.Api(timeout=180)


def _add_total_peace(h: pd.DataFrame) -> pd.DataFrame:
    if h is None or h.empty or "peace" not in h.columns:
        return h
    peace = pd.to_numeric(h["peace"], errors="coerce")
    if "system_peace" in h.columns:
        sys_p = pd.to_numeric(h["system_peace"], errors="coerce").fillna(
            float(N_HARVESTERS)
        )
    else:
        sys_p = float(N_HARVESTERS)
    h = h.copy()
    h["total_peace"] = peace + sys_p - float(N_HARVESTERS)
    return h


def get_hist(run, *, metrics: list[str] | None = None):
    h = fetch_run_history(
        run,
        metrics=metrics or FETCH_METRICS,
        x_axis="episode_count",
        x_fallbacks=("update", "env_steps", "_step"),
        full=False,
    )
    return _add_total_peace(h)


def load_group(group: str, finished_only: bool = True):
    """Baselines: fetch only ToC metrics (no police keys)."""
    base_metrics = ["mean_reward", "peace", "system_peace"]
    by_seed: dict = {}
    for r in api.runs(PROJ, filters={"group": group}, order="created_at"):
        seed = (r.config or {}).get("SEED")
        if seed is None:
            continue
        if finished_only and r.state != "finished":
            continue
        by_seed[int(seed)] = r
    runs = [by_seed[s] for s in sorted(by_seed)]
    print(f"  {group}: seeds={sorted(by_seed)} n={len(runs)}")
    return [(r, get_hist(r, metrics=base_metrics)) for r in runs]


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
        if seed == 0:
            continue
        by_seed[seed] = r
    runs = [by_seed[s] for s in sorted(by_seed)]
    print(f"  seeds={sorted(by_seed)} n={len(runs)}")
    return [(r, get_hist(r)) for r in runs]


def agg_runs(pairs, metric: str, label: str, mode: str = "independent"):
    series_list = []
    source = DERIVED_FROM_RATE.get(metric, metric)
    for r, h in pairs:
        if h is None or h.empty or source not in h.columns:
            print(f"    missing {source} on {r.id}")
            continue
        x_col = h.attrs.get("x_col", "episode_count")
        s = extract_series_from_history(
            h,
            metric=metric,
            x_col=x_col,
            label=label,
            mode=mode,
            run_id=r.id,
            meta=dict(r.config or {}),
        )
        if s is not None:
            series_list.append(s)
    if not series_list:
        return None
    return aggregate_across_runs(series_list, metric=metric, label=label, mode=mode)


def clip_xy(x, y, xmax=X_MAX):
    m = np.asarray(x, float) <= float(xmax)
    return np.asarray(x, float)[m], np.asarray(y, float)[m]


def plot_baselines(ax, base_hists, metric: str):
    """Light-grey γ=0.995 overlays; omitted from legend."""
    for mode, ls in [("independent", "-"), ("shared", "--")]:
        pairs = base_hists.get(mode)
        if not pairs:
            continue
        agg = agg_runs(pairs, metric, mode, mode=mode)
        if agg is None:
            continue
        x, mean = clip_xy(agg.x, agg.mean)
        _, std = clip_xy(agg.x, agg.std)
        if len(x) == 0:
            continue
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=BASELINE_COLOR,
            alpha=0.12 if mode == "independent" else 0.08,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            mean,
            color=BASELINE_COLOR,
            linestyle=ls,
            linewidth=1.8,
            label="_nolegend_",
            alpha=0.85,
            zorder=2,
        )


def plot_agg(ax, agg, color, label, ls="-"):
    if agg is None:
        return
    x, mean = clip_xy(agg.x, agg.mean)
    _, std = clip_xy(agg.x, agg.std)
    if len(x) == 0:
        return
    n = int(np.nanmax(agg.n)) if getattr(agg, "n", None) is not None else 1
    if n > 1:
        ax.fill_between(
            x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0
        )
    ax.plot(
        x,
        mean,
        color=color,
        linewidth=2.4,
        linestyle=ls,
        label=label,
        alpha=0.95,
    )


def ylabel_for(metric: str) -> str:
    if metric in METRIC_YLABEL_OVERRIDE:
        return METRIC_YLABEL_OVERRIDE[metric]
    return metric_ylabel(metric)


def main() -> None:
    print("loading γ=0.995 baselines (police-matched)...")
    base_hists = {
        mode: load_group(grp) for mode, grp in BASELINE_GROUPS.items()
    }

    loaded = []
    for label, group, seed0, color, ls in ARMS:
        print(f"\nloading {label}...")
        pairs = load_sweep(group, seed0)
        loaded.append((label, color, ls, pairs))

    OUT.mkdir(parents=True, exist_ok=True)
    for metric in ZAP_METRICS + TOC_METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        if metric in BASELINE_METRICS:
            plot_baselines(ax, base_hists, metric)
        for label, color, ls, pairs in loaded:
            plot_agg(ax, agg_runs(pairs, metric, label), color, label, ls=ls)
        ax.set_xlabel(EPISODE_XLABEL, fontsize=THESIS_FONT["label"])
        ax.set_ylabel(ylabel_for(metric), fontsize=THESIS_FONT["label"])
        ax.set_xlim(0, X_MAX)
        ax.tick_params(axis="both", labelsize=THESIS_FONT["tick"])
        ax.grid(True, alpha=0.25)
        _apply_axis_formatters(ax)
        ax.legend(
            loc="best",
            fontsize=THESIS_FONT["legend"],
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
