"""Peer + Good-imm masked vs unmasked on one panel (zap / Utility / total peace).

γ=0.99 ENT-anneal twins; light-grey Individual/Shared baselines.
Style: Peer green / Good imm sand; solid=unmasked, dashed=mask.
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

G99_BASELINE_GROUPS = {
    "independent": "ippo_v2_w2_m1_no_peer_indep_ep4_mb25_g99_ent005_floor0005_50kep",
    "shared": "ippo_v2_w2_m1_no_peer_shared_ep4_mb25_g99_ent005_floor0005_50kep",
}

ZAP_METRICS = [
    "zap_attempts_per_episode",
    "zap_hit_rate",
    "zap_tagged_rep_mean",
]
TOC_METRICS = ["mean_reward", "total_peace"]
ALL_METRICS = ZAP_METRICS + TOC_METRICS

FETCH_METRICS = resolve_history_metrics(ZAP_METRICS) + [
    "mean_reward",
    "peace",
    "system_peace",
]

METRIC_FILES = {
    "zap_attempts_per_episode": "zap_attempts",
    "zap_hit_rate": "zap_hit_rate",
    "zap_tagged_rep_mean": "zap_tagged_rep_mean",
    "mean_reward": "mean_reward",
    "total_peace": "total_peace",
}

METRIC_YLABEL_OVERRIDE = {
    "total_peace": "Peace (P)",
}

OUT = Path("harvest_socialjax/plots/sweep_comparisons/peer_gimm_mask_vs_unmasked_behavior")

# Peer + Good imm on one panel: solid=unmasked, dashed=mask; family colors.
ARMS = [
    (
        ARM_LABELS["peer"],
        "ippo_v2_w2_m1_scratch_br5_g5_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        plot_seed0(_PLOT_RUNS, "peer"),
        ARM_COLORS["peer"],
        "-",
    ),
    (
        ARM_LABELS["peer_masked"],
        "ippo_v2_w2_m1_scratch_br5_g5_mask_rep_tdur25_ep4_mb25_g99_ent_floor0005_mask_rep_50kep",
        plot_seed0(_PLOT_RUNS, "peer_masked"),
        ARM_COLORS["peer"],
        "--",
    ),
    (
        ARM_LABELS["gimm"],
        "ippo_v2_w2_m1_scratch_br5_gimm_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        plot_seed0(_PLOT_RUNS, "gimm"),
        ARM_COLORS["gimm"],
        "-",
    ),
    (
        "Good imm mask",
        "ippo_v2_w2_m1_scratch_br5_gimm_mask_rep_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        plot_seed0(_PLOT_RUNS, "gimm_masked"),
        ARM_COLORS["gimm"],
        "--",
    ),
]

api = wandb.Api(timeout=180)


def get_hist(run):
    h = fetch_run_history(
        run,
        metrics=FETCH_METRICS,
        x_axis="episode_count",
        x_fallbacks=("update", "env_steps", "_step"),
        full=False,
    )
    if h is None or h.empty:
        return h
    if "peace" in h.columns:
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


def load_group(group: str, finished_only: bool = True):
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
    return [(r, get_hist(r)) for r in runs]


def load_sweep(group: str, seed0_id: str | None, finished_only: bool = True):
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
        if seed == 0 and seed0_id and r.id != seed0_id:
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
    for mode, leg, ls in [
        ("independent", ARM_LABELS["independent"], "-"),
        ("shared", ARM_LABELS["shared"], "--"),
    ]:
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


def plot_agg(ax, agg, color, label, ls="-", *, alpha_line=0.95, alpha_band=0.18, lw=2.4, z=4):
    if agg is None:
        return
    x, mean = clip_xy(agg.x, agg.mean)
    _, std = clip_xy(agg.x, agg.std)
    if len(x) == 0:
        return
    n = int(np.nanmax(agg.n)) if getattr(agg, "n", None) is not None else 1
    if n > 1:
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=alpha_band,
            linewidth=0,
            zorder=z - 1,
        )
    ax.plot(
        x,
        mean,
        color=color,
        linewidth=lw,
        linestyle=ls,
        label=label,
        alpha=alpha_line,
        zorder=z,
    )


def ylabel_for(metric: str) -> str:
    if metric in METRIC_YLABEL_OVERRIDE:
        return METRIC_YLABEL_OVERRIDE[metric]
    return metric_ylabel(metric)


def main() -> None:
    print("loading γ=0.99 baselines...")
    base_hists = {
        mode: load_group(grp) for mode, grp in G99_BASELINE_GROUPS.items()
    }

    loaded = []
    for label, group, seed0, color, ls in ARMS:
        print(f"\nloading {label}...")
        pairs = load_sweep(group, seed0)
        loaded.append((label, color, ls, pairs))

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {OUT.name} ===")
    for metric in ALL_METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
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
