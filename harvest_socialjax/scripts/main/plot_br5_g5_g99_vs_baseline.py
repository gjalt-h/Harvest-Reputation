"""Peer ±5 + Good immunity (gimm) g99/ENT anneal vs γ=0.99 no-peer baselines.

Arms (mean±std over finished seeds):
  - Peer +/-5 (W&B group + optional seed0 from plot_runs.yaml)
  - Good immunity / gimm g99 (group + optional seed0)
  - Baselines: no-peer indep/shared with matching γ=0.99 + ENT anneal

Writes both folder names the thesis has used:
  - br5_g5_g99_ent0005_vs_baseline
  - br5_g5_g99_ent005to0005_vs_baseline
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
    BASELINE_COLOR,
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
X_MAX_EP = 50_000
METRICS = ["mean_reward", "peace", "sustainability", "equality"]

PEER_GROUP = "ippo_v2_w2_m1_scratch_br5_g5_tdur25_ep4_mb25_g99_ent005_floor0005_50kep"
PEER_SEED0 = plot_seed0(_PLOT_RUNS, "peer")
PEER_LABEL = ARM_LABELS["peer"]
PEER_COLOR = ARM_COLORS["peer"]

GIMM_GROUP = "ippo_v2_w2_m1_scratch_br5_gimm_tdur25_ep4_mb25_g99_ent005_floor0005_50kep"
GIMM_SEED0 = plot_seed0(_PLOT_RUNS, "gimm")
GIMM_LABEL = ARM_LABELS["gimm"]
GIMM_COLOR = ARM_COLORS["gimm"]

OUT_DIRS = [
    Path("harvest_socialjax/plots/sweep_comparisons/br5_g5_g99_ent0005_vs_baseline"),
    Path("harvest_socialjax/plots/sweep_comparisons/br5_g5_g99_ent005to0005_vs_baseline"),
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


def load_sweep(group: str, seed0_id: str | None):
    by_seed: dict = {}
    if seed0_id:
        r0 = api.run(f"{PROJ}/{seed0_id}")
        by_seed[int((r0.config or {}).get("SEED", 0))] = r0
        print(f"  seed0 {seed0_id}: state={r0.state} group={r0.group}")
    for r in api.runs(PROJ, filters={"group": group}, order="created_at"):
        if r.state != "finished":
            print(f"  skip seed={(r.config or {}).get('SEED')} id={r.id} state={r.state}")
            continue
        seed = (r.config or {}).get("SEED")
        if seed is None:
            continue
        seed = int(seed)
        if seed0_id and seed == 0:
            continue  # keep canonical seed0 id
        by_seed[seed] = r
    runs = [by_seed[s] for s in sorted(by_seed)]
    print(f"  using seeds={sorted(by_seed)} n={len(runs)}")
    return [(r, get_hist(r)) for r in runs]


# Matched learning twin of peer/gimm g99 (not the older default-γ baselines).
G99_BASELINE_GROUPS = {
    "independent": "ippo_v2_w2_m1_no_peer_indep_ep4_mb25_g99_ent005_floor0005_50kep",
    "shared": "ippo_v2_w2_m1_no_peer_shared_ep4_mb25_g99_ent005_floor0005_50kep",
}


def load_baselines():
    out = {}
    for mode, grp in G99_BASELINE_GROUPS.items():
        runs = list(
            api.runs(PROJ, filters={"group": grp, "state": "finished"}, order="-created_at")
        )
        by_seed = {}
        for r in runs:
            by_seed[int((r.config or {}).get("SEED", 0))] = r
        sel = [by_seed[s] for s in sorted(by_seed)]
        print(f"g99 baseline {mode}: {len(sel)} seeds  group={grp}")
        out[mode] = [(r, get_hist(r)) for r in sel]
    return out


def agg_runs(pairs, metric: str, label: str, mode: str = "independent"):
    series_list = []
    for r, h in pairs:
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


def clip_xy(x, y, xmax=X_MAX_EP):
    m = np.asarray(x, float) <= float(xmax)
    return np.asarray(x, float)[m], np.asarray(y, float)[m]


def plot_agg(ax, agg, color, label, ls="-", *, in_legend=True, lw=2.4, alpha=0.95):
    if agg is None:
        return
    x, mean = clip_xy(agg.x, agg.mean)
    _, std = clip_xy(agg.x, agg.std)
    if len(x) == 0:
        return
    n = int(np.nanmax(agg.n)) if getattr(agg, "n", None) is not None else 1
    lab = label if in_legend else "_nolegend_"
    if n > 1:
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.12 if color == BASELINE_COLOR else 0.16,
            linewidth=0,
        )
    ax.plot(x, mean, color=color, linestyle=ls, linewidth=lw, label=lab, alpha=alpha)


def main() -> None:
    print("loading baselines...")
    base = load_baselines()

    print("\nloading peer +/-5 g99...")
    peer = load_sweep(PEER_GROUP, PEER_SEED0)

    print("\nloading good immunity (gimm g99)...")
    gimm = load_sweep(GIMM_GROUP, GIMM_SEED0)

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    label_fs, tick_fs, legend_fs = (
        THESIS_FONT["label"],
        THESIS_FONT["tick"],
        THESIS_FONT["legend"],
    )
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        plot_agg(
            ax,
            agg_runs(
                base["independent"],
                metric,
                ARM_LABELS["independent"],
                mode="independent",
            ),
            BASELINE_COLOR,
            ARM_LABELS["independent"],
            ls="-",
            in_legend=True,
            lw=1.8,
            alpha=0.85,
        )
        plot_agg(
            ax,
            agg_runs(
                base["shared"], metric, ARM_LABELS["shared"], mode="shared"
            ),
            BASELINE_COLOR,
            ARM_LABELS["shared"],
            ls="--",
            in_legend=True,
            lw=1.8,
            alpha=0.85,
        )
        plot_agg(ax, agg_runs(peer, metric, PEER_LABEL), PEER_COLOR, PEER_LABEL)
        plot_agg(ax, agg_runs(gimm, metric, GIMM_LABEL), GIMM_COLOR, GIMM_LABEL)

        ax.set_xlabel(EPISODE_XLABEL, fontsize=label_fs)
        ax.set_ylabel(metric_ylabel(metric), fontsize=label_fs)
        ax.set_xlim(0, X_MAX_EP)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.25)
        _apply_axis_formatters(ax)
        ax.legend(
            loc="best",
            fontsize=legend_fs,
            framealpha=0.9,
            handlelength=2.0,
            borderpad=0.3,
            labelspacing=0.3,
        )
        fig.tight_layout()
        for out_dir in OUT_DIRS:
            out = out_dir / f"{metric}.png"
            fig.savefig(out, dpi=220, bbox_inches="tight")
            print(f"wrote {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
