"""Last-10% mean_reward vs total peace for all 8-seed sweeps.

total_peace = peace + system_peace - N_h
  (peer timeouts + enforcement timeouts; N_h=4 harvesters on test grid)
When system_peace is missing, treat as N_h (no enforcement).

Color = experimental family; marker = arm within family.
  - baseline (grey)
  - external enforcement (soft blue)
  - peer enforcement (meadow green)
  - ablations (warm sand)
"""
from __future__ import annotations

import os

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.lines import Line2D
from harvest_socialjax.wandb_sweep_viz.fetch import fetch_run_history
from harvest_socialjax.wandb_sweep_viz.plot import ARM_LABELS, THESIS_FONT
from harvest_socialjax.wandb_sweep_viz.plot_runs import (
    cfg_get,
    load_plot_runs,
    seed0 as plot_seed0,
)

PROJ = f"{os.environ.get('WANDB_ENTITY', 'YOUR_ENTITY')}/{os.environ.get('WANDB_PROJECT', 'harvest_ippo_rep')}"
_PLOT_RUNS = load_plot_runs()
N_HARVESTERS = 4
METRICS = ["mean_reward", "peace", "system_peace"]
OUT_DIR = Path("harvest_socialjax/plots/sweep_comparisons/last10pct_reward_vs_total_peace")

# Light orchard tones — easier to tell apart on white.
FAMILY_COLORS = {
    "baseline": "#b0b0b5",  # light grey
    "external": "#5b8fbf",  # soft mid blue
    "peer": "#7ec28a",  # light meadow green
    "ablation": "#d4a574",  # light sand / terracotta
}
FAMILY_LABELS = {
    "baseline": "Baselines",
    "external": "External Enforcement",
    "peer": "Peer Enforcement",
    "ablation": "Ablations",
}

# (label, family, marker, wandb_group, seed0_id, drop_name_substr)
SWEEPS: list[tuple[str, str, str, str | None, str | None, str | None]] = [
    (ARM_LABELS["independent"], "baseline", "o", "ippo_v2_w2_m1_no_peer_indep_50kep", None, None),
    (ARM_LABELS["shared"], "baseline", "s", "ippo_v2_w2_m1_no_peer_shared_50kep", None, None),
    # Learning twins of peer/gimm (γ=0.99, ENT 0.05→0.005, ep4/mb25).
    (
        ARM_LABELS["independent"] + " γ=0.99",
        "baseline",
        "^",
        "ippo_v2_w2_m1_no_peer_indep_ep4_mb25_g99_ent005_floor0005_50kep",
        None,
        None,
    ),
    (
        ARM_LABELS["shared"] + " γ=0.99",
        "baseline",
        "v",
        "ippo_v2_w2_m1_no_peer_shared_ep4_mb25_g99_ent005_floor0005_50kep",
        None,
        None,
    ),
    (ARM_LABELS["enf"], "external", "o", "ippo_v2_w2_m1_enf_period_p1_50m", plot_seed0(_PLOT_RUNS, "enf"), None),
    (ARM_LABELS["police"], "external", "^", "ippo_v2_w2_m1_scratch_police1_no_peer_tdur25_50M", plot_seed0(_PLOT_RUNS, "police"), None),
    (ARM_LABELS["police_masked"], "external", "v", "ippo_v2_w2_m1_scratch_police1_no_peer_mask_rep_tdur25_50M", plot_seed0(_PLOT_RUNS, "police_masked"), None),
    (
        ARM_LABELS["enf"] + " drop@u200",
        "external",
        "D",
        "ippo_v2_w2_m1_drop_enf_continue_50M",
        cfg_get(_PLOT_RUNS, "drop", "enf_seed0"),
        "drop_enf_from_u200_enfp1",
    ),
    (
        ARM_LABELS["police"] + " drop@u200",
        "external",
        "P",
        "ippo_v2_w2_m1_drop_enf_continue_50M",
        cfg_get(_PLOT_RUNS, "drop", "police_seed0"),
        "drop_enf_from_u200_police1",
    ),
    (ARM_LABELS["peer"], "peer", "o", "ippo_v2_w2_m1_scratch_br5_g5_tdur25_ep4_mb25_g99_ent005_floor0005_50kep", plot_seed0(_PLOT_RUNS, "peer"), None),
    (ARM_LABELS["gimm"], "peer", "s", "ippo_v2_w2_m1_scratch_br5_gimm_tdur25_ep4_mb25_g99_ent005_floor0005_50kep", plot_seed0(_PLOT_RUNS, "gimm"), None),
    (ARM_LABELS["peer_masked"], "peer", "^", "ippo_v2_w2_m1_scratch_br5_g5_mask_rep_tdur25_ep4_mb25_g99_ent_floor0005_mask_rep_50kep", plot_seed0(_PLOT_RUNS, "peer_masked"), None),
    (
        ARM_LABELS["peer"] + " drop@u200",
        "peer",
        "D",
        "ippo_v2_w2_m1_br5g5_ent005_g99_drop_rew_from_u200_50M",
        cfg_get(_PLOT_RUNS, "drop", "peer_seed0"),
        "drop_rew_from_u200",
    ),
    ("Zap any", "ablation", "o", "ippo_v2_w2_m1_scratch_zap5_any_tdur25_50M", None, None),
    ("Zap Bad only", "ablation", "^", "ippo_v2_w2_m1_scratch_zap_bad5_only_tdur25_50M", None, None),
    ("Zap Good only", "ablation", "v", "ippo_v2_w2_m1_scratch_zap_goodm5_only_tdur25_50M", None, None),
]

SWEEP_META = {label: (family, marker) for label, family, marker, *_ in SWEEPS}

api = wandb.Api(timeout=180)


def get_hist(run):
    return fetch_run_history(
        run,
        metrics=METRICS,
        x_axis="episode_count",
        x_fallbacks=("update", "env_steps", "_step"),
        full=False,
    )


def load_runs(
    group: str | None,
    seed0_id: str | None,
    name_substr: str | None,
) -> list:
    by_seed: dict = {}
    if seed0_id:
        r0 = api.run(f"{PROJ}/{seed0_id}")
        by_seed[int((r0.config or {}).get("SEED", 0))] = r0
    if group is None:
        return [by_seed[s] for s in sorted(by_seed)]
    for r in api.runs(PROJ, filters={"group": group}, order="created_at"):
        if r.state != "finished":
            continue
        if name_substr and name_substr.lower() not in (r.name or "").lower():
            continue
        seed = (r.config or {}).get("SEED")
        if seed is None:
            continue
        seed = int(seed)
        if seed == 0 and seed0_id and r.id != seed0_id:
            continue
        by_seed[seed] = r
    return [by_seed[s] for s in sorted(by_seed)]


def last10_means(hist: pd.DataFrame) -> tuple[float, float, float, float] | None:
    if hist is None or hist.empty or "mean_reward" not in hist.columns:
        return None
    x_col = hist.attrs.get("x_col", "episode_count")
    if x_col not in hist.columns or "peace" not in hist.columns:
        return None
    x = pd.to_numeric(hist[x_col], errors="coerce")
    rew = pd.to_numeric(hist["mean_reward"], errors="coerce")
    peace = pd.to_numeric(hist["peace"], errors="coerce")
    if "system_peace" in hist.columns:
        sys_p = pd.to_numeric(hist["system_peace"], errors="coerce")
    else:
        sys_p = pd.Series(np.full(len(hist), float(N_HARVESTERS)), index=hist.index)
    m = x.notna() & rew.notna() & peace.notna()
    if m.sum() < 2:
        return None
    x = x[m].to_numpy(float)
    rew = rew[m].to_numpy(float)
    peace = peace[m].to_numpy(float)
    sys_p = sys_p[m].fillna(float(N_HARVESTERS)).to_numpy(float)
    x_max = float(np.nanmax(x))
    if not np.isfinite(x_max) or x_max <= 0:
        return None
    thr = 0.9 * x_max
    sel = x >= thr
    if sel.sum() < 1:
        return None
    r = float(np.nanmean(rew[sel]))
    p = float(np.nanmean(peace[sel]))
    s = float(np.nanmean(sys_p[sel]))
    total = p + s - float(N_HARVESTERS)
    return r, p, s, total


# Ablations: masks, drops, zap variants (excluded from the main plot).
ABLATION_LABELS = frozenset(
    {
        "Zap any",
        "Zap Bad only",
        "Zap Good only",
        ARM_LABELS["police_masked"],
        ARM_LABELS["peer_masked"],
        ARM_LABELS["enf"] + " drop@u200",
        ARM_LABELS["police"] + " drop@u200",
        ARM_LABELS["peer"] + " drop@u200",
    }
)
EXCLUDE_LABELS = ABLATION_LABELS  # alias for main plot


def _plot_family(label: str, family: str, *, ablations_as_family: bool) -> str:
    if ablations_as_family and label in ABLATION_LABELS:
        return "ablation"
    return family


def plot_mean_dots(
    ax,
    summary: pd.DataFrame,
    *,
    exclude_labels: frozenset[str] = frozenset(),
    ablations_as_family: bool = False,
) -> None:
    """Mean ± std circles; color = family."""
    arms = [
        (lab, _plot_family(lab, fam, ablations_as_family=ablations_as_family))
        for lab, fam, *_ in SWEEPS
        if lab not in exclude_labels
    ]
    family_order = ("baseline", "external", "peer", "ablation")
    labeled_families: set[str] = set()

    for label, family in arms:
        row = summary[summary["label"] == label]
        if row.empty:
            continue
        color = FAMILY_COLORS[family]
        first = family not in labeled_families
        if first:
            labeled_families.add(family)
        ax.errorbar(
            float(row["total_peace_mean"].iloc[0]),
            float(row["reward_mean"].iloc[0]),
            xerr=float(row["total_peace_std"].iloc[0]),
            yerr=float(row["reward_std"].iloc[0]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3.5,
            markersize=12,
            markeredgecolor="#222222",
            markeredgewidth=0.9,
            alpha=0.95,
            # One legend entry per family.
            label="_nolegend_",
            zorder=3,
        )

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FAMILY_COLORS[f],
            markeredgecolor="#222222",
            markersize=10,
            label=FAMILY_LABELS[f],
        )
        for f in family_order
        if f in labeled_families
    ]
    ax.legend(
        handles=family_handles,
        loc="upper left",
        fontsize=THESIS_FONT["legend"],
        framealpha=0.92,
        handlelength=1.4,
    )


def save_dot_plot(
    summary: pd.DataFrame,
    *,
    out_name: str,
    exclude_labels: frozenset[str],
    ablations_as_family: bool,
    xlim: tuple[float, float],
) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    plot_mean_dots(
        ax,
        summary,
        exclude_labels=exclude_labels,
        ablations_as_family=ablations_as_family,
    )
    ax.set_xlabel("Peace (P)", fontsize=THESIS_FONT["label"])
    ax.set_ylabel("Utility (U)", fontsize=THESIS_FONT["label"])
    ax.tick_params(axis="both", labelsize=THESIS_FONT["tick"])
    ax.grid(True, alpha=0.25)
    ax.set_xlim(*xlim)
    fig.tight_layout()
    out_png = OUT_DIR / out_name
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    return out_png


def main(*, refetch: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "last10pct_summary.csv"
    seeds_path = OUT_DIR / "last10pct_per_seed.csv"

    if refetch or not summary_path.exists() or not seeds_path.exists():
        rows = []
        seed_rows = []
        for label, family, marker, group, seed0, substr in SWEEPS:
            color = FAMILY_COLORS[family]
            print(f"\n=== {label} ({family}) ===")
            runs = load_runs(group, seed0, substr)
            print(f"  n={len(runs)} seeds={[ (r.config or {}).get('SEED') for r in runs ]}")
            rews, totals, peaces, sysps = [], [], [], []
            for r in runs:
                seed = int((r.config or {}).get("SEED", -1))
                hist = get_hist(r)
                vals = last10_means(hist)
                if vals is None:
                    print(f"  skip seed={seed} id={r.id}")
                    continue
                rew, peace, sys_p, total = vals
                rews.append(rew)
                peaces.append(peace)
                sysps.append(sys_p)
                totals.append(total)
                seed_rows.append(
                    {
                        "label": label,
                        "family": family,
                        "marker": marker,
                        "color": color,
                        "seed": seed,
                        "run_id": r.id,
                        "reward": rew,
                        "peace": peace,
                        "system_peace": sys_p,
                        "total_peace": total,
                    }
                )
                print(
                    f"  seed={seed} reward={rew:.2f} peace={peace:.3f} "
                    f"sys={sys_p:.3f} total={total:.3f}"
                )
            if not rews:
                continue
            rows.append(
                {
                    "label": label,
                    "family": family,
                    "marker": marker,
                    "color": color,
                    "n": len(rews),
                    "group": group or "",
                    "seed0": seed0 or "",
                    "reward_mean": float(np.mean(rews)),
                    "reward_std": float(np.std(rews, ddof=1)) if len(rews) > 1 else 0.0,
                    "peace_mean": float(np.mean(peaces)),
                    "system_peace_mean": float(np.mean(sysps)),
                    "total_peace_mean": float(np.mean(totals)),
                    "total_peace_std": float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0,
                }
            )
        summary = pd.DataFrame(rows)
        seeds_df = pd.DataFrame(seed_rows)
        summary.to_csv(summary_path, index=False)
        seeds_df.to_csv(seeds_path, index=False)
        print(f"\nwrote {summary_path}")
        print(f"wrote {seeds_path}")
    else:
        summary = pd.read_csv(summary_path)
        seeds_df = pd.read_csv(seeds_path)
        # Refresh family/marker/color from current SWEEP_META if replotting old CSV.
        if "family" not in seeds_df.columns:
            seeds_df["family"] = seeds_df["label"].map(lambda L: SWEEP_META[L][0])
            seeds_df["marker"] = seeds_df["label"].map(lambda L: SWEEP_META[L][1])
            seeds_df["color"] = seeds_df["family"].map(FAMILY_COLORS)
        if "family" not in summary.columns:
            summary["family"] = summary["label"].map(lambda L: SWEEP_META[L][0])
            summary["marker"] = summary["label"].map(lambda L: SWEEP_META[L][1])
            summary["color"] = summary["family"].map(FAMILY_COLORS)

    # Main thesis plot: no masks / drops / zap.
    save_dot_plot(
        summary,
        out_name="reward_vs_total_peace_last10pct.png",
        exclude_labels=EXCLUDE_LABELS,
        ablations_as_family=False,
        xlim=(2.2, 4.2),
    )
    # Full plot: include all ablations (mask, drop, zap) in Ablations color.
    save_dot_plot(
        summary,
        out_name="reward_vs_total_peace_last10pct_with_ablations.png",
        exclude_labels=frozenset(),
        ablations_as_family=True,
        xlim=(1.5, 4.2),
    )

    kept = summary[~summary["label"].isin(EXCLUDE_LABELS)].copy()
    payload = {
        "n_harvesters": N_HARVESTERS,
        "total_peace_def": "peace + system_peace - N_h",
        "window": "last 10% of episode_count per run; mean±std over seeds",
        "excluded_from_main": sorted(EXCLUDE_LABELS),
        "families": FAMILY_LABELS,
        "family_colors": FAMILY_COLORS,
        "sweeps_main": kept.to_dict(orient="records"),
        "sweeps_all": summary.to_dict(orient="records"),
    }
    json_path = OUT_DIR / "last10pct_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    import sys

    # Default: replot from CSV if present (family styling only); --refetch to hit W&B.
    refetch = "--refetch" in sys.argv
    main(refetch=refetch)
