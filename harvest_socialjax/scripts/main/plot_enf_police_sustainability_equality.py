"""Plot sustainability + equality: baselines + police1 + enf period p1.

Loads finished W&B *groups* (and optional seed0 from ``configs/plot_runs.yaml``).
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
from harvest_socialjax.wandb_sweep_viz.plot_runs import load_plot_runs, seed0 as plot_seed0
from harvest_socialjax.wandb_sweep_viz.series import (
    EPISODE_XLABEL,
    aggregate_across_runs,
    extract_series_from_history,
    metric_ylabel,
)

PROJ = f"{os.environ.get('WANDB_ENTITY', 'YOUR_ENTITY')}/{os.environ.get('WANDB_PROJECT', 'harvest_ippo_rep')}"
_PLOT_RUNS = load_plot_runs()
api = wandb.Api(timeout=120)

METRICS = ["sustainability", "equality"]
OUT_DIR = Path("harvest_socialjax/plots/sweep_comparisons/enf_p1_police_vs_baseline")

BASE_GROUPS = {
    "independent": "ippo_v2_w2_m1_no_peer_indep_50kep",
    "shared": "ippo_v2_w2_m1_no_peer_shared_50kep",
}
ARM_SPECS = [
    (
        ARM_LABELS["police"],
        "ippo_v2_w2_m1_scratch_police1_no_peer_tdur25_50M",
        plot_seed0(_PLOT_RUNS, "police"),
        ARM_COLORS["police"],
    ),
    (
        ARM_LABELS["enf"],
        "ippo_v2_w2_m1_enf_period_p1_50m",
        plot_seed0(_PLOT_RUNS, "enf"),
        ARM_COLORS["enf"],
    ),
]


def get_hist(run, metric_list):
    return fetch_run_history(
        run,
        metrics=metric_list,
        x_axis="episode_count",
        x_fallbacks=("update", "env_steps", "_step"),
        full=False,
    )


def load_group(group: str, seed0_id: str | None = None):
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
        if seed == 0 and seed0_id and r.id != seed0_id:
            continue
        by_seed[seed] = r
    runs = [by_seed[s] for s in sorted(by_seed)]
    print(f"  {group}: seeds={sorted(by_seed)} n={len(runs)}")
    return runs


def aggregate_runs(runs, metric, label, mode="independent"):
    series_list = []
    for r in runs:
        h = get_hist(r, METRICS)
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


def main() -> None:
    print("loading baselines...")
    base_hists = {}
    for mode, grp in BASE_GROUPS.items():
        runs = load_group(grp)
        base_hists[mode] = [(r, get_hist(r, METRICS)) for r in runs]

    print("loading police / enf arms...")
    arm_aggs: dict[str, dict] = {m: {} for m in METRICS}
    for label, group, seed0_id, color in ARM_SPECS:
        runs = load_group(group, seed0_id)
        for metric in METRICS:
            arm_aggs[metric][label] = (
                aggregate_runs(runs, metric, label),
                color,
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for mode, ls in [("independent", "-"), ("shared", "--")]:
            series_list = []
            for r, h in base_hists[mode]:
                if h is None or h.empty or metric not in h.columns:
                    continue
                x_col = h.attrs.get("x_col", "episode_count")
                s = extract_series_from_history(
                    h,
                    metric=metric,
                    x_col=x_col,
                    label=mode,
                    mode=mode,
                    run_id=r.id,
                    meta={"SEED": (r.config or {}).get("SEED")},
                )
                if s is not None:
                    series_list.append(s)
            if not series_list:
                continue
            agg = aggregate_across_runs(
                series_list, metric=metric, label=mode, mode=mode
            )
            x = np.asarray(agg.x, float)
            mean = np.asarray(agg.mean, float)
            std = np.asarray(agg.std, float)
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=BASELINE_COLOR,
                alpha=0.12 if mode == "independent" else 0.08,
                linewidth=0,
            )
            ax.plot(
                x,
                mean,
                color=BASELINE_COLOR,
                linestyle=ls,
                linewidth=1.8,
                label=ARM_LABELS.get(mode, mode),
            )
        for label, (agg, color) in arm_aggs[metric].items():
            if agg is None:
                continue
            x = np.asarray(agg.x, float)
            mean = np.asarray(agg.mean, float)
            std = np.asarray(agg.std, float)
            ax.fill_between(
                x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0
            )
            ax.plot(x, mean, color=color, linewidth=2.4, label=label, alpha=0.95)
        ax.set_xlabel(EPISODE_XLABEL, fontsize=THESIS_FONT["label"])
        ax.set_ylabel(metric_ylabel(metric), fontsize=THESIS_FONT["label"])
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
        out = OUT_DIR / f"{metric}.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
