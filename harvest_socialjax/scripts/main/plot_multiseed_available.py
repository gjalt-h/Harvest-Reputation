"""Redo key comparison plots with multi-seed mean±std where available.

Drop u200 multi-seed arms (enf/police/peer) + continued originals.
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
    aggregate_across_runs,
    extract_series_from_history,
    EPISODE_XLABEL,
    metric_ylabel,
)
from harvest_socialjax.wandb_sweep_viz.plot_runs import (
    cfg_get,
    load_plot_runs,
    seed0 as plot_seed0,
)

PROJ = f"{os.environ.get('WANDB_ENTITY', 'YOUR_ENTITY')}/{os.environ.get('WANDB_PROJECT', 'harvest_ippo_rep')}"
_PLOT_RUNS = load_plot_runs()
METRICS = ["mean_reward", "peace", "sustainability", "equality"]
# Fetched with every history; only plotted where requested.
EXTRA_METRICS = ["system_peace"]
FETCH_METRICS = METRICS + EXTRA_METRICS
X_MAX = 50_000
DROP_EP = 19_200
# Metrics where indep/shared baselines are flat / irrelevant (no enforcement).
SKIP_BASELINE_METRICS = frozenset({"system_peace"})

# Shared W&B group holds several drop arms; filter by name substring.
DROP_ENF_POLICE_GROUP = "ippo_v2_w2_m1_drop_enf_continue_50M"
DROP_PEER_GROUP = "ippo_v2_w2_m1_br5g5_ent005_g99_drop_rew_from_u200_50M"

api = wandb.Api(timeout=180)


def get_hist(run):
    return fetch_run_history(
        run,
        metrics=FETCH_METRICS,
        x_axis="episode_count",
        x_fallbacks=("update", "env_steps", "_step"),
        full=False,
    )


BASELINE_GROUPS = {
    "independent": "ippo_v2_w2_m1_no_peer_indep_50kep",
    "shared": "ippo_v2_w2_m1_no_peer_shared_50kep",
}
# Matched twin of peer ±5 / gimm g99 (γ=0.99, ENT 0.05→0.005).
G99_BASELINE_GROUPS = {
    "independent": "ippo_v2_w2_m1_no_peer_indep_ep4_mb25_g99_ent005_floor0005_50kep",
    "shared": "ippo_v2_w2_m1_no_peer_shared_ep4_mb25_g99_ent005_floor0005_50kep",
}


def load_baseline_hists(groups: dict[str, str] | None = None, *, label: str = "baseline"):
    base_groups = groups or BASELINE_GROUPS
    out = {}
    for mode, grp in base_groups.items():
        runs = list(
            api.runs(PROJ, filters={"group": grp, "state": "finished"}, order="-created_at")
        )
        by_seed = {}
        for r in runs:
            by_seed[(r.config or {}).get("SEED", 0)] = r
        sel = [by_seed[s] for s in sorted(by_seed)]
        print(f"{label} {mode}: {len(sel)} seeds  group={grp}")
        out[mode] = [(r, get_hist(r)) for r in sel]
    return out


def load_sweep(group: str, seed0_id: str | None, finished_only: bool = True):
    """Return list[(run, hist)] including seed0 + group seeds."""
    by_seed: dict = {}
    if seed0_id:
        r0 = api.run(f"{PROJ}/{seed0_id}")
        by_seed[(r0.config or {}).get("SEED", 0)] = r0
        print(f"  seed0 {seed0_id}: state={r0.state} ep={r0.summary.get('episode_count')}")
    for r in api.runs(PROJ, filters={"group": group}, order="created_at"):
        seed = (r.config or {}).get("SEED")
        if finished_only and r.state != "finished":
            print(f"  skip seed={seed} id={r.id} state={r.state} ep={r.summary.get('episode_count')}")
            continue
        # Prefer group run over seed0 if same seed and finished in group
        if seed in by_seed and seed0_id and r.id != seed0_id:
            # keep seed0 for seed 0; for others prefer group
            if seed != 0:
                by_seed[seed] = r
        else:
            by_seed[seed] = r
    runs = [by_seed[s] for s in sorted(by_seed) if s is not None]
    print(f"  using seeds={sorted(by_seed)} n={len(runs)}")
    return [(r, get_hist(r)) for r in runs]


def load_drop_arm(
    group: str,
    name_substr: str,
    seed0_id: str | None = None,
    finished_only: bool = True,
    exclude_substr: str | None = None,
):
    """Load one drop arm from a (possibly mixed) W&B group, keyed by SEED."""
    by_seed: dict = {}
    if seed0_id:
        r0 = api.run(f"{PROJ}/{seed0_id}")
        by_seed[int((r0.config or {}).get("SEED", 0))] = r0
        print(f"  drop seed0 {seed0_id}: state={r0.state}")
    for r in api.runs(PROJ, filters={"group": group}, order="created_at"):
        name = (r.name or "").lower()
        if name_substr.lower() not in name:
            continue
        if exclude_substr and exclude_substr.lower() in name:
            continue
        seed = (r.config or {}).get("SEED")
        if seed is None:
            continue
        seed = int(seed)
        if finished_only and r.state != "finished":
            print(f"  skip drop s={seed} id={r.id} state={r.state}")
            continue
        # Prefer newer group members for seeds 1+; keep explicit seed0_id for 0
        if seed == 0 and seed0_id and r.id != seed0_id:
            continue
        by_seed[seed] = r
    runs = [by_seed[s] for s in sorted(by_seed)]
    print(
        f"  drop '{name_substr}' seeds={sorted(by_seed)} n={len(runs)} "
        f"ids={[r.id for r in runs]}"
    )
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


def plot_baselines(ax, base_hists, metric):
    """Light-grey reference curves (solid=indep, dashed=shared)."""
    if metric in SKIP_BASELINE_METRICS:
        return
    for mode, leg, ls in [
        ("independent", ARM_LABELS["independent"], "-"),
        ("shared", ARM_LABELS["shared"], "--"),
    ]:
        agg = agg_runs(base_hists[mode], metric, mode, mode=mode)
        if agg is None:
            continue
        color = BASELINE_COLOR
        x, mean = clip_xy(np.asarray(agg.x, float), np.asarray(agg.mean, float))
        _, std = clip_xy(np.asarray(agg.x, float), np.asarray(agg.std, float))
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


def plot_agg_series(
    ax, agg, color, label, alpha_line=0.95, alpha_band=0.18, lw=2.4, z=4, ls="-"
):
    if agg is None:
        return
    x, mean = clip_xy(np.asarray(agg.x, float), np.asarray(agg.mean, float))
    _, std = clip_xy(np.asarray(agg.x, float), np.asarray(agg.std, float))
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
        label=lab,
        alpha=alpha_line,
        zorder=z,
    )


def plot_single_metric(ax, hist, metric, color, label, alpha=0.95, lw=2.4, z=4):
    if hist is None or metric not in hist.columns:
        return
    x_col = hist.attrs.get("x_col", "episode_count")
    x = pd.to_numeric(hist[x_col], errors="coerce")
    y = pd.to_numeric(hist[metric], errors="coerce")
    m = x.notna() & y.notna()
    x = x[m].to_numpy(float)
    y = y[m].to_numpy(float)
    o = np.argsort(x)
    x, y = clip_xy(x[o], y[o])
    ax.plot(x, y, color=color, linewidth=lw, label=label, alpha=alpha, zorder=z)


def save_set(out_name, plot_fn, metrics=None):
    out_dir = Path("harvest_socialjax/plots/sweep_comparisons") / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {out_name} ===")
    for metric in metrics or METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        plot_fn(ax, metric)
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
            handlelength=2.4,
        )
        fig.tight_layout()
        out = out_dir / f"{metric}.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


def _drop_vline(ax):
    ax.axvline(
        DROP_EP,
        color="#333333",
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        zorder=5,
        label="Drop @u200",
    )


def main():
    print("loading baselines (default γ)...")
    base_hists = load_baseline_hists(label="baseline")

    print("\nloading baselines (γ=0.99 twin of peer/gimm)...")
    base_hists_g99 = load_baseline_hists(G99_BASELINE_GROUPS, label="g99 baseline")

    print("\nloading enf p1 sweep...")
    enf = load_sweep(
        "ippo_v2_w2_m1_enf_period_p1_50m",
        plot_seed0(_PLOT_RUNS, "enf"),
        finished_only=True,
    )

    print("\nloading police sweep...")
    police = load_sweep(
        "ippo_v2_w2_m1_scratch_police1_no_peer_tdur25_50M",
        plot_seed0(_PLOT_RUNS, "police"),
        finished_only=True,
    )

    print("\nloading peer g99 anneal...")
    peer = load_sweep(
        "ippo_v2_w2_m1_scratch_br5_g5_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        plot_seed0(_PLOT_RUNS, "peer"),
        finished_only=True,
    )

    print("\nloading police MASK sweep...")
    police_mask = load_sweep(
        "ippo_v2_w2_m1_scratch_police1_no_peer_mask_rep_tdur25_50M",
        plot_seed0(_PLOT_RUNS, "police_masked"),
        finished_only=True,
    )

    print("\nloading peer MASK sweep...")
    peer_mask = load_sweep(
        "ippo_v2_w2_m1_scratch_br5_g5_mask_rep_tdur25_ep4_mb25_g99_ent_floor0005_mask_rep_50kep",
        plot_seed0(_PLOT_RUNS, "peer_masked"),
        finished_only=True,
    )

    print("\nloading good immunity gimm g99 (multi-seed)...")
    gimm = load_sweep(
        "ippo_v2_w2_m1_scratch_br5_gimm_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        plot_seed0(_PLOT_RUNS, "gimm"),
        finished_only=True,
    )

    print("\nloading drop enf u200 (multi-seed)...")
    drop_enf = load_drop_arm(
        DROP_ENF_POLICE_GROUP,
        name_substr="drop_enf_from_u200_enfp1",
        seed0_id=cfg_get(_PLOT_RUNS, "drop", "enf_seed0"),
        finished_only=True,
    )

    print("\nloading drop police u200 (multi-seed)...")
    drop_police = load_drop_arm(
        DROP_ENF_POLICE_GROUP,
        name_substr="drop_enf_from_u200_police1",
        seed0_id=cfg_get(_PLOT_RUNS, "drop", "police_seed0"),
        finished_only=True,
    )

    print("\nloading drop peer u200 (multi-seed, finished only)...")
    drop_peer = load_drop_arm(
        DROP_PEER_GROUP,
        name_substr="drop_rew_from_u200",
        seed0_id=cfg_get(_PLOT_RUNS, "drop", "peer_seed0"),
        finished_only=True,
    )

    # 1a) enf + police vs baselines (no peer)
    def plot_enf_police(ax, metric):
        plot_baselines(ax, base_hists, metric)
        plot_agg_series(
            ax,
            agg_runs(enf, metric, ARM_LABELS["enf"]),
            ARM_COLORS["enf"],
            ARM_LABELS["enf"],
        )
        plot_agg_series(
            ax,
            agg_runs(police, metric, ARM_LABELS["police"]),
            ARM_COLORS["police"],
            ARM_LABELS["police"],
        )

    enf_police_metrics = METRICS + ["system_peace"]
    save_set(
        "enf_police_multiseed_vs_baseline",
        plot_enf_police,
        metrics=enf_police_metrics,
    )
    # Keep legacy folder name used in thesis drafts (system_peace without baselines).
    save_set(
        "enf_p1_police_vs_baseline",
        plot_enf_police,
        metrics=enf_police_metrics,
    )

    # 1b) enf + police + peer vs baselines
    def plot_main(ax, metric):
        plot_baselines(ax, base_hists, metric)
        plot_agg_series(
            ax,
            agg_runs(enf, metric, ARM_LABELS["enf"]),
            ARM_COLORS["enf"],
            ARM_LABELS["enf"],
        )
        plot_agg_series(
            ax,
            agg_runs(police, metric, ARM_LABELS["police"]),
            ARM_COLORS["police"],
            ARM_LABELS["police"],
        )
        plot_agg_series(
            ax,
            agg_runs(peer, metric, ARM_LABELS["peer"]),
            ARM_COLORS["peer"],
            ARM_LABELS["peer"],
        )

    save_set(
        "enf_police_peer_multiseed_vs_baseline",
        plot_main,
        metrics=enf_police_metrics,
    )

    # 2) peer (+ newest good-immunity gimm g99) vs matched γ=0.99 baselines
    def plot_peer_gimm(ax, metric):
        plot_baselines(ax, base_hists_g99, metric)
        plot_agg_series(
            ax,
            agg_runs(peer, metric, ARM_LABELS["peer"]),
            ARM_COLORS["peer"],
            ARM_LABELS["peer"],
        )
        plot_agg_series(
            ax,
            agg_runs(gimm, metric, ARM_LABELS["gimm"]),
            ARM_COLORS["gimm"],
            ARM_LABELS["gimm"],
        )

    save_set("br5_g5_g99_ent005to0005_vs_baseline", plot_peer_gimm)
    save_set("br5_g5_g99_ent0005_vs_baseline", plot_peer_gimm)

    def _plot_drops_core(ax, metric, with_baselines: bool):
        if with_baselines:
            plot_baselines(ax, base_hists, metric)
        plot_agg_series(
            ax,
            agg_runs(enf, metric, ARM_LABELS["enf"] + " cont."),
            ARM_COLORS["enf"],
            ARM_LABELS["enf"] + " cont.",
            alpha_line=0.45,
            alpha_band=0.12,
            lw=2.0,
            z=3,
        )
        plot_agg_series(
            ax,
            agg_runs(police, metric, ARM_LABELS["police"] + " cont."),
            ARM_COLORS["police"],
            ARM_LABELS["police"] + " cont.",
            alpha_line=0.45,
            alpha_band=0.12,
            lw=2.0,
            z=3,
        )
        plot_agg_series(
            ax,
            agg_runs(peer, metric, ARM_LABELS["peer"] + " cont."),
            ARM_COLORS["peer"],
            ARM_LABELS["peer"] + " cont.",
            alpha_line=0.40,
            alpha_band=0.10,
            lw=2.0,
            z=3,
        )
        plot_agg_series(
            ax,
            agg_runs(drop_enf, metric, "Drop enf"),
            ARM_COLORS["enf"],
            "Drop enf",
            alpha_line=0.95,
            alpha_band=0.22,
            lw=2.7,
            z=4,
        )
        plot_agg_series(
            ax,
            agg_runs(drop_police, metric, "Drop police"),
            ARM_COLORS["police"],
            "Drop police",
            alpha_line=0.95,
            alpha_band=0.22,
            lw=2.7,
            z=4,
        )
        plot_agg_series(
            ax,
            agg_runs(drop_peer, metric, "Drop peer"),
            ARM_COLORS["gimm"],
            "Drop peer",
            alpha_line=0.95,
            alpha_band=0.22,
            lw=2.7,
            z=4,
        )
        _drop_vline(ax)

    save_set(
        "drop_u200_vs_original_vs_baseline",
        lambda ax, metric: _plot_drops_core(ax, metric, with_baselines=True),
    )
    save_set(
        "drop_u200_vs_original",
        lambda ax, metric: _plot_drops_core(ax, metric, with_baselines=False),
    )

    # mask vs unmasked police / peer (multi-seed)
    def plot_police_mask(ax, metric):
        plot_baselines(ax, base_hists, metric)
        plot_agg_series(
            ax,
            agg_runs(police, metric, ARM_LABELS["police"]),
            ARM_COLORS["police"],
            ARM_LABELS["police"],
        )
        plot_agg_series(
            ax,
            agg_runs(police_mask, metric, ARM_LABELS["police_masked"]),
            ARM_COLORS["police"],
            ARM_LABELS["police_masked"],
            ls="--",
        )

    save_set("police1_mask_vs_unmasked", plot_police_mask)

    def plot_peer_mask(ax, metric):
        plot_baselines(ax, base_hists_g99, metric)
        plot_agg_series(
            ax,
            agg_runs(peer, metric, ARM_LABELS["peer"]),
            ARM_COLORS["peer"],
            ARM_LABELS["peer"],
        )
        plot_agg_series(
            ax,
            agg_runs(peer_mask, metric, ARM_LABELS["peer_masked"]),
            ARM_COLORS["peer"],
            ARM_LABELS["peer_masked"],
            ls="--",
        )

    save_set("br5_g5_mask_vs_unmasked", plot_peer_mask)

    print("\n=== COMPLETION STATUS ===")
    print(f"Enf continued: n={len(enf)}")
    print(f"Police continued: n={len(police)}")
    print(f"Peer continued: n={len(peer)}")
    print(f"Police MASK: n={len(police_mask)}")
    print(f"Peer MASK: n={len(peer_mask)}")
    print(f"Drop enf u200: n={len(drop_enf)}")
    print(f"Drop police u200: n={len(drop_police)}")
    print(f"Drop peer u200 (finished): n={len(drop_peer)}")
    print("Good immunity gimm g99: multi-seed included in br5_g5_g99_*_vs_baseline")


if __name__ == "__main__":
    main()

