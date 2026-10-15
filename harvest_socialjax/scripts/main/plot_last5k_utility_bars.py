"""Bar chart: last-5k mean Utility (±std across seeds).

Arms: Individual / Shared baselines + Direct enf / Police.
"""
from __future__ import annotations

import os

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from harvest_socialjax.wandb_sweep_viz.fetch import fetch_run_history
from harvest_socialjax.wandb_sweep_viz.plot import ARM_COLORS, ARM_LABELS, THESIS_FONT
from harvest_socialjax.wandb_sweep_viz.plot_runs import load_plot_runs, seed0 as plot_seed0

PROJ = f"{os.environ.get('WANDB_ENTITY', 'YOUR_ENTITY')}/{os.environ.get('WANDB_PROJECT', 'harvest_ippo_rep')}"
_PLOT_RUNS = load_plot_runs()
LAST_EP = 5_000
X_MAX = 50_000
OUT_DIR = Path("harvest_socialjax/plots/sweep_comparisons/last5k_utility_bars")

# Same family as line-plot ARM_COLORS, lightened for filled bars.
BAR_COLORS = {
    "independent": "#9bc492",  # soft orchard green
    "shared": "#d2c2a4",  # light sand
    "enf": "#d0b08a",  # light terracotta
    "police": "#6a8fbf",  # soft mid blue (police family)
}

# (label, color_key, wandb_group, seed0_id)
ARMS = [
    (
        ARM_LABELS["independent"],
        "independent",
        "ippo_v2_w2_m1_no_peer_indep_50kep",
        None,
    ),
    (
        ARM_LABELS["shared"],
        "shared",
        "ippo_v2_w2_m1_no_peer_shared_50kep",
        None,
    ),
    (
        ARM_LABELS["enf"],
        "enf",
        "ippo_v2_w2_m1_enf_period_p1_50m",
        plot_seed0(_PLOT_RUNS, "enf"),
    ),
    (
        ARM_LABELS["police"],
        "police",
        "ippo_v2_w2_m1_scratch_police1_no_peer_tdur25_50M",
        plot_seed0(_PLOT_RUNS, "police"),
    ),
]

api = wandb.Api(timeout=180)


def get_hist(run):
    return fetch_run_history(
        run,
        metrics=["mean_reward"],
        x_axis="episode_count",
        x_fallbacks=("update", "env_steps", "_step"),
        full=False,
    )


def load_runs(group: str, seed0_id: str | None) -> list:
    by_seed: dict = {}
    if seed0_id:
        r0 = api.run(f"{PROJ}/{seed0_id}")
        by_seed[int((r0.config or {}).get("SEED", 0))] = r0
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


def last5k_mean_utility(run) -> float | None:
    h = get_hist(run)
    if h is None or h.empty or "mean_reward" not in h.columns:
        return None
    x_col = h.attrs.get("x_col", "episode_count")
    x = pd.to_numeric(h[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(h["mean_reward"], errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(x) & ~np.isnan(y) & (x <= X_MAX)
    if not valid.any():
        return None
    x, y = x[valid], y[valid]
    xmin = max(float(np.nanmax(x)) - LAST_EP, 0.0)
    m = x >= xmin
    if not m.any():
        return None
    return float(np.mean(y[m]))


def _collect() -> tuple[list, list, list, list, list]:
    labels, colors, means, stds, ns = [], [], [], [], []
    rows = []
    for label, color_key, group, seed0 in ARMS:
        color = BAR_COLORS[color_key]
        print(f"\nloading {label}...")
        runs = load_runs(group, seed0)
        vals = []
        for r in runs:
            u = last5k_mean_utility(r)
            seed = int((r.config or {}).get("SEED", -1))
            if u is None:
                print(f"  skip seed={seed} id={r.id}: no last-{LAST_EP} utility")
                continue
            vals.append(u)
            rows.append({"arm": label, "seed": seed, "run_id": r.id, "U_last5k": u})
            print(f"  seed={seed}: U_last5k={u:.1f}")
        arr = np.asarray(vals, dtype=float)
        labels.append(label)
        colors.append(color)
        means.append(float(np.mean(arr)) if arr.size else float("nan"))
        stds.append(float(np.std(arr, ddof=0)) if arr.size > 1 else 0.0)
        ns.append(int(arr.size))
        print(f"  mean±std = {means[-1]:.1f} ± {stds[-1]:.1f}  (n={ns[-1]})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "last5k_utility_per_seed.csv", index=False)
    pd.DataFrame({"arm": labels, "mean": means, "std": stds, "n": ns}).to_csv(
        OUT_DIR / "last5k_utility_summary.csv", index=False
    )
    return labels, colors, means, stds, ns


def main() -> None:
    summary_path = OUT_DIR / "last5k_utility_summary.csv"
    if summary_path.exists():
        print(f"using cached {summary_path}")
        summary = pd.read_csv(summary_path)
        labels = summary["arm"].tolist()
        means = summary["mean"].astype(float).tolist()
        stds = summary["std"].astype(float).tolist()
        color_by_label = {lab: BAR_COLORS[k] for lab, k, _, _ in ARMS}
        colors = [color_by_label[lab] for lab in labels]
    else:
        labels, colors, means, stds, _ns = _collect()

    # Narrow canvas; keep height + text size; pack bars tightly.
    n = len(labels)
    step = 0.58
    width = 0.52
    span = (n - 1) * step
    x = np.linspace(-span / 2, span / 2, n)
    fig, ax = plt.subplots(figsize=(3.2, 3.4))
    key_by_label = {lab: k for lab, k, _, _ in ARMS}
    edge = [ARM_COLORS[key_by_label[lab]] for lab in labels]
    label_fs, tick_fs = 12, 10
    ax.bar(
        x,
        means,
        yerr=stds,
        color=colors,
        edgecolor=edge,
        linewidth=0.9,
        width=width,
        capsize=2.5,
        error_kw={
            "elinewidth": 1.1,
            "capthick": 1.0,
            "ecolor": "#555555",
        },
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        fontsize=tick_fs,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Final Utility (U)", fontsize=label_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.set_xlim(-span / 2 - 0.40, span / 2 + 0.40)
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.10)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.2)
    out = OUT_DIR / "last5k_utility_bars.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
