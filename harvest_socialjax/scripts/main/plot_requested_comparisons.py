"""Generate the requested comparison plot sets vs indep/shared baselines.

Single-run overlays come from ``configs/plot_runs.yaml`` → ``requested.*``
(see ``plot_runs.example.yaml``). Sets with no filled IDs are skipped.
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
    BASELINE_COLOR,
    SWEEP_COLORS,
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
BASE_GROUPS = {
    "independent": "ippo_v2_w2_m1_no_peer_indep_50kep",
    "shared": "ippo_v2_w2_m1_no_peer_shared_50kep",
}

# Run IDs come from configs/plot_runs.yaml → requested.*; null entries are skipped.
def _plot_sets():
    req = _PLOT_RUNS.get("requested") or {}
    return [
        (
            "br5_g5_g99_ent005to0005_100k_vs_baseline",
            100_000,
            [(req.get("peer_g99_100k"), "Peer +/-5 (g=0.99, ENT 0.05->0.005, 100k)", 2)],
            True,
        ),
        (
            "br5_g5_g995_ent001_vs_baseline",
            50_000,
            [(req.get("peer_g995"), "Peer +/-5 (g=0.995, ENT=0.01)", 2)],
            True,
        ),
        (
            "br5_gimm_vs_enf_mask_norew",
            50_000,
            [
                (req.get("gimm"), "Good immunity (+/-5)", 2),
                (req.get("enf_p1"), "Strict enf (period=1)", 3),
                (req.get("gimm_masked"), "Good immunity + masked obs", 4),
                (req.get("gimm_no_rew"), "Good immunity, no peer reward", 5),
            ],
            True,
        ),
        (
            "police1_mask_vs_unmasked",
            50_000,
            [
                (req.get("police"), "Police1 unmasked", 3),
                (req.get("police_masked"), "Police1 masked obs", 4),
            ],
            True,
        ),
    ]


PLOT_SETS = _plot_sets()


def _clip_xy(x, y, xmax):
    if xmax is None:
        return x, y
    mask = x <= float(xmax)
    return x[mask], y[mask]


def get_hist(run, metric_list):
    return fetch_run_history(
        run,
        metrics=metric_list,
        x_axis="episode_count",
        x_fallbacks=("update", "env_steps", "_step"),
        full=False,
    )


def load_baselines(api):
    base_runs: dict[str, list] = {}
    for mode, grp in BASE_GROUPS.items():
        runs = list(
            api.runs(
                PROJ,
                filters={"group": grp, "state": "finished"},
                order="-created_at",
            )
        )
        by_seed: dict = {}
        for r in runs:
            seed = (r.config or {}).get("SEED", 0)
            by_seed[seed] = r
        base_runs[mode] = [by_seed[s] for s in sorted(by_seed)]
        print(f"baseline {mode}: {len(base_runs[mode])} seeds")
    base_hists = {}
    for mode, runs in base_runs.items():
        base_hists[mode] = [(r, get_hist(r, METRICS)) for r in runs]
    return base_hists


def aggregate_mode(base_hists, mode, metric):
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
        return None
    return aggregate_across_runs(series_list, metric=metric, label=mode, mode=mode)


def plot_set(out_name, x_max, singles, include_baselines, base_hists):
    out_dir = Path("harvest_socialjax/plots/sweep_comparisons") / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {out_name} (x_max={x_max}) ===")
    for label, color, hist in singles:
        cols = [m for m in METRICS if hist is not None and m in hist.columns]
        print(f"  {label}: rows={0 if hist is None else len(hist)} cols={cols}")

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        if include_baselines:
            for mode, leg, ls in [
                ("independent", "Individual", "-"),
                ("shared", "Shared", "--"),
            ]:
                agg = aggregate_mode(base_hists, mode, metric)
                if agg is None:
                    print(f"  no baseline agg for {mode}/{metric}")
                    continue
                color = BASELINE_COLOR
                x, mean = _clip_xy(
                    np.asarray(agg.x, dtype=float),
                    np.asarray(agg.mean, dtype=float),
                    x_max,
                )
                _, std = _clip_xy(
                    np.asarray(agg.x, dtype=float),
                    np.asarray(agg.std, dtype=float),
                    x_max,
                )
                ax.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=0.12 if mode == "independent" else 0.08,
                    linewidth=0,
                    zorder=1,
                )
                ax.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=ls,
                    linewidth=1.8,
                    label=leg,
                    alpha=0.85,
                    zorder=2,
                )

        for label, color, hist in singles:
            if hist is None or metric not in hist.columns:
                print(f"  skip {label}/{metric}")
                continue
            x_col = hist.attrs.get("x_col", "episode_count")
            x = pd.to_numeric(hist[x_col], errors="coerce")
            y = pd.to_numeric(hist[metric], errors="coerce")
            mask = x.notna() & y.notna()
            x = x[mask].to_numpy(dtype=float)
            y = y[mask].to_numpy(dtype=float)
            order = np.argsort(x)
            x, y = _clip_xy(x[order], y[order], x_max)
            ax.plot(x, y, color=color, linewidth=2.6, label=label)

        ax.set_xlabel(EPISODE_XLABEL, fontsize=14)
        ax.set_ylabel(metric_ylabel(metric), fontsize=14)
        if x_max is not None:
            ax.set_xlim(0, x_max)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.25)
        _apply_axis_formatters(ax)
        ax.legend(loc="best", fontsize=9, framealpha=0.9, handlelength=2.6)
        fig.tight_layout()
        out = out_dir / f"{metric}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


def main():
    api = wandb.Api(timeout=120)
    print("loading baselines...")
    base_hists = load_baselines(api)
    plot_sets = _plot_sets()

    # Cache histories by run id (skip unset / null IDs from plot_runs.yaml)
    hist_cache: dict[str, object] = {}
    needed_ids = {
        rid
        for _, _, specs, _ in plot_sets
        for rid, _, _ in specs
        if rid
    }
    if not needed_ids:
        raise SystemExit(
            "No run IDs in configs/plot_runs.yaml (requested.*). "
            "Copy plot_runs.example.yaml → plot_runs.yaml and fill them in."
        )
    for rid in sorted(needed_ids):
        run = api.run(f"{PROJ}/{str(rid)}")
        print(f"fetch {rid}: {run.name} state={run.state}")
        hist_cache[str(rid)] = get_hist(run, METRICS)

    for out_name, x_max, specs, include_baselines in plot_sets:
        singles = []
        for rid, label, cidx in specs:
            rid_s = str(rid) if rid else ""
            if not rid_s or rid_s not in hist_cache:
                print(f"  skip unset run for {label!r}")
                continue
            color = SWEEP_COLORS[cidx % len(SWEEP_COLORS)]
            singles.append((label, color, hist_cache[rid_s]))
        if not singles:
            print(f"skip plot set {out_name}: no configured run IDs")
            continue
        plot_set(out_name, x_max, singles, include_baselines, base_hists)


if __name__ == "__main__":
    main()
