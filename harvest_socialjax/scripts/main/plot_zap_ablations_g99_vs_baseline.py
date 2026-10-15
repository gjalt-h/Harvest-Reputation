"""Zap ablations (γ=0.99 ENT anneal) vs Peer ±5 + matched baselines.

Peer learning twin: γ=0.99, ENT 0.05→0.005.
Outputs to zap_ablations_g99_vs_baseline/ (separate from old γ=0.995 plots).

Legend: Peer, Zap Any, Target Bad, Avoid Good.
Baselines: Individual/Shared γ=0.99 overlays (not in legend).
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
METRICS = ["mean_reward", "peace", "sustainability", "equality"]
X_MAX = 50_000
OUT_DIR = Path("harvest_socialjax/plots/sweep_comparisons/zap_ablations_g99_vs_baseline")

G99_BASELINE_GROUPS = {
    "independent": "ippo_v2_w2_m1_no_peer_indep_ep4_mb25_g99_ent005_floor0005_50kep",
    "shared": "ippo_v2_w2_m1_no_peer_shared_ep4_mb25_g99_ent005_floor0005_50kep",
}

PEER_SEED0_ID = plot_seed0(_PLOT_RUNS, "peer")
PEER_SEEDS_GROUP = "ippo_v2_w2_m1_scratch_br5_g5_tdur25_ep4_mb25_g99_ent005_floor0005_50kep"

ZAP_ARM_COLORS = {
    "Peer": ARM_COLORS["peer"],
    "Zap Any": "#8fb8c9",
    "Target Bad": "#a67c52",
    "Avoid Good": "#c4b59a",
}

# (group or None for peer, legend label, color)
ZAP_ARMS = [
    (None, "Peer", ZAP_ARM_COLORS["Peer"]),
    (
        "ippo_v2_w2_m1_scratch_zap5_any_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        "Zap Any",
        ZAP_ARM_COLORS["Zap Any"],
    ),
    (
        "ippo_v2_w2_m1_scratch_zap_bad5_only_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        "Target Bad",
        ZAP_ARM_COLORS["Target Bad"],
    ),
    (
        "ippo_v2_w2_m1_scratch_zap_goodm5_only_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        "Avoid Good",
        ZAP_ARM_COLORS["Avoid Good"],
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


def load_group(group: str, finished_only: bool = True):
    by_seed: dict = {}
    for r in api.runs(PROJ, filters={"group": group}, order="created_at"):
        seed = (r.config or {}).get("SEED")
        if seed is None:
            continue
        if finished_only and r.state != "finished":
            print(f"  skip s={seed} {r.id} state={r.state}")
            continue
        by_seed[int(seed)] = r
    runs = [by_seed[s] for s in sorted(by_seed)]
    print(f"  {group}: seeds={sorted(by_seed)} n={len(runs)}")
    return [(r, get_hist(r)) for r in runs]


def load_peer_pm5(finished_only: bool = True):
    by_seed: dict = {}
    if PEER_SEED0_ID:
        r0 = api.run(f"{PROJ}/{PEER_SEED0_ID}")
        by_seed[int((r0.config or {}).get("SEED", 0))] = r0
        print(f"  peer seed0 {PEER_SEED0_ID}: state={r0.state}")
    for r in api.runs(PROJ, filters={"group": PEER_SEEDS_GROUP}, order="created_at"):
        seed = (r.config or {}).get("SEED")
        if seed is None:
            continue
        seed = int(seed)
        if finished_only and r.state != "finished":
            print(f"  skip peer s={seed} {r.id} state={r.state}")
            continue
        if PEER_SEED0_ID and seed == 0:
            continue
        by_seed[seed] = r
    runs = [by_seed[s] for s in sorted(by_seed)]
    print(f"  peer +/-5: seeds={sorted(by_seed)} n={len(runs)}")
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
    m = x <= float(xmax)
    return x[m], y[m]


def plot_baselines_muted(ax, base_hists, metric):
    for mode, ls in [("independent", "-"), ("shared", "--")]:
        pairs = base_hists.get(mode)
        if not pairs:
            continue
        agg = agg_runs(pairs, metric, mode, mode=mode)
        if agg is None:
            continue
        x, mean = clip_xy(np.asarray(agg.x, float), np.asarray(agg.mean, float))
        _, std = clip_xy(np.asarray(agg.x, float), np.asarray(agg.std, float))
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


def plot_agg(ax, agg, color, label, alpha_line=0.95, alpha_band=0.18, lw=2.4, z=4):
    if agg is None:
        return
    x, mean = clip_xy(np.asarray(agg.x, float), np.asarray(agg.mean, float))
    _, std = clip_xy(np.asarray(agg.x, float), np.asarray(agg.std, float))
    if len(x) == 0:
        return
    n = int(np.nanmax(agg.n)) if getattr(agg, "n", None) is not None else 1
    if n > 1:
        ax.fill_between(
            x, mean - std, mean + std, color=color, alpha=alpha_band, linewidth=0, zorder=z - 1
        )
    ax.plot(x, mean, color=color, linewidth=lw, label=label, alpha=alpha_line, zorder=z)


def main():
    print("loading γ=0.99 baselines...")
    base_hists = {
        mode: load_group(grp) for mode, grp in G99_BASELINE_GROUPS.items()
    }

    arms = []
    for group, label, color in ZAP_ARMS:
        print(f"\nloading {label}...")
        if group is None:
            run_hists = load_peer_pm5()
        else:
            run_hists = load_group(group)
        arms.append((label, run_hists, color))

    # Utility / peace: bottom-right (away from early trajectories).
    LEGEND_LOC = {
        "mean_reward": "lower right",
        "peace": "lower right",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        plot_baselines_muted(ax, base_hists, metric)
        for label, run_hists, color in arms:
            plot_agg(ax, agg_runs(run_hists, metric, label), color, label)
        ax.set_xlabel(EPISODE_XLABEL, fontsize=THESIS_FONT["label"])
        ax.set_ylabel(metric_ylabel(metric), fontsize=THESIS_FONT["label"])
        ax.set_xlim(0, X_MAX)
        ax.tick_params(axis="both", labelsize=THESIS_FONT["tick"])
        ax.grid(True, alpha=0.25)
        _apply_axis_formatters(ax)
        ax.legend(
            loc=LEGEND_LOC.get(metric, "best"),
            fontsize=THESIS_FONT["legend"],
            framealpha=0.9,
            handlelength=2.4,
        )
        fig.tight_layout()
        out = OUT_DIR / f"{metric}.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
