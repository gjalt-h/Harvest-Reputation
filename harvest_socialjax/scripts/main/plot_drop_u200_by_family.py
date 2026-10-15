"""Drop @u200 Utility / Peace plots by family with matched baselines.

1) External (Direct enf + Police) vs γ=0.995 Individual/Shared baselines
2) Peer + Good imm vs γ=0.99 twin baselines

Legend / colors match prior drop_u200 plots (cont. muted, Drop bold, Drop @u200 vline).
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
from harvest_socialjax.wandb_sweep_viz.plot_runs import (
    cfg_get,
    load_plot_runs,
    seed0 as plot_seed0,
)

PROJ = f"{os.environ.get('WANDB_ENTITY', 'YOUR_ENTITY')}/{os.environ.get('WANDB_PROJECT', 'harvest_ippo_rep')}"
_PLOT_RUNS = load_plot_runs()
METRICS = ["mean_reward", "peace"]
X_MAX = 50_000
DROP_EP = 19_200

DROP_ENF_POLICE_GROUP = "ippo_v2_w2_m1_drop_enf_continue_50M"
DROP_PEER_GROUP = "ippo_v2_w2_m1_br5g5_ent005_g99_drop_rew_from_u200_50M"
DROP_GIMM_GROUP = "ippo_v2_w2_m1_br5_gimm_g99_ent005_drop_rew_keepimm_from_u200_50M"

BASELINE_GROUPS = {
    "independent": "ippo_v2_w2_m1_no_peer_indep_50kep",
    "shared": "ippo_v2_w2_m1_no_peer_shared_50kep",
}
G99_BASELINE_GROUPS = {
    "independent": "ippo_v2_w2_m1_no_peer_indep_ep4_mb25_g99_ent005_floor0005_50kep",
    "shared": "ippo_v2_w2_m1_no_peer_shared_ep4_mb25_g99_ent005_floor0005_50kep",
}

api = wandb.Api(timeout=180)


def get_hist(run):
    return fetch_run_history(
        run,
        metrics=METRICS,
        x_axis="episode_count",
        x_fallbacks=("update", "env_steps", "_step"),
        full=False,
    )


def load_baseline_hists(groups: dict[str, str], *, label: str):
    out = {}
    for mode, grp in groups.items():
        by_seed: dict = {}
        for r in api.runs(PROJ, filters={"group": grp, "state": "finished"}, order="-created_at"):
            seed = (r.config or {}).get("SEED")
            if seed is None:
                continue
            by_seed[int(seed)] = r
        sel = [by_seed[s] for s in sorted(by_seed)]
        print(f"{label} {mode}: n={len(sel)} group={grp}")
        out[mode] = [(r, get_hist(r)) for r in sel]
    return out


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
    print(f"  {group}: seeds={sorted(by_seed)} n={len(runs)}")
    return [(r, get_hist(r)) for r in runs]


def load_drop_arm(
    group: str,
    name_substr: str,
    seed0_id: str | None = None,
    finished_only: bool = True,
):
    by_seed: dict = {}
    if seed0_id:
        r0 = api.run(f"{PROJ}/{seed0_id}")
        by_seed[int((r0.config or {}).get("SEED", 0))] = r0
        print(f"  drop seed0 {seed0_id}: state={r0.state}")
    for r in api.runs(PROJ, filters={"group": group}, order="created_at"):
        if name_substr.lower() not in (r.name or "").lower():
            continue
        seed = (r.config or {}).get("SEED")
        if seed is None:
            continue
        seed = int(seed)
        if finished_only and r.state != "finished":
            print(f"  skip drop s={seed} {r.id} state={r.state}")
            continue
        if seed == 0 and seed0_id and r.id != seed0_id:
            continue
        by_seed[seed] = r
    runs = [by_seed[s] for s in sorted(by_seed)]
    print(f"  drop '{name_substr}': seeds={sorted(by_seed)} n={len(runs)}")
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


def plot_agg(
    ax,
    agg,
    color,
    label,
    *,
    alpha_line=0.95,
    alpha_band=0.18,
    lw=2.4,
    z=4,
):
    if agg is None:
        return
    x, mean = clip_xy(np.asarray(agg.x, float), np.asarray(agg.mean, float))
    _, std = clip_xy(np.asarray(agg.x, float), np.asarray(agg.std, float))
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
        label=label,
        alpha=alpha_line,
        zorder=z,
    )


def drop_vline(ax):
    ax.axvline(
        DROP_EP,
        color="#333333",
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        zorder=5,
        label="_nolegend_",
    )


def save_set(out_name: str, plot_fn) -> None:
    out_dir = Path("harvest_socialjax/plots/sweep_comparisons") / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {out_name} ===")
    for metric in METRICS:
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


def main() -> None:
    print("loading γ=0.995 baselines...")
    base_hists = load_baseline_hists(BASELINE_GROUPS, label="baseline")

    print("\nloading γ=0.99 baselines...")
    base_hists_g99 = load_baseline_hists(G99_BASELINE_GROUPS, label="g99 baseline")

    print("\nloading Direct enf...")
    enf = load_sweep(
        "ippo_v2_w2_m1_enf_period_p1_50m", plot_seed0(_PLOT_RUNS, "enf")
    )

    print("\nloading Police...")
    police = load_sweep(
        "ippo_v2_w2_m1_scratch_police1_no_peer_tdur25_50M",
        plot_seed0(_PLOT_RUNS, "police"),
    )

    print("\nloading Peer...")
    peer = load_sweep(
        "ippo_v2_w2_m1_scratch_br5_g5_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        plot_seed0(_PLOT_RUNS, "peer"),
    )

    print("\nloading Good imm...")
    gimm = load_sweep(
        "ippo_v2_w2_m1_scratch_br5_gimm_tdur25_ep4_mb25_g99_ent005_floor0005_50kep",
        plot_seed0(_PLOT_RUNS, "gimm"),
    )

    print("\nloading Drop Direct enf...")
    drop_enf = load_drop_arm(
        DROP_ENF_POLICE_GROUP,
        name_substr="drop_enf_from_u200_enfp1",
        seed0_id=cfg_get(_PLOT_RUNS, "drop", "enf_seed0"),
    )

    print("\nloading Drop Police...")
    drop_police = load_drop_arm(
        DROP_ENF_POLICE_GROUP,
        name_substr="drop_enf_from_u200_police1",
        seed0_id=cfg_get(_PLOT_RUNS, "drop", "police_seed0"),
    )

    print("\nloading Drop Peer...")
    drop_peer = load_drop_arm(
        DROP_PEER_GROUP,
        name_substr="drop_rew_from_u200",
        seed0_id=cfg_get(_PLOT_RUNS, "drop", "peer_seed0"),
    )

    print("\nloading Drop Good imm (keepimm)...")
    drop_gimm = load_sweep(
        DROP_GIMM_GROUP, cfg_get(_PLOT_RUNS, "drop", "gimm_seed0")
    )

    def plot_external(ax, metric):
        plot_baselines(ax, base_hists, metric)
        plot_agg(
            ax,
            agg_runs(enf, metric, ARM_LABELS["enf"] + " cont."),
            ARM_COLORS["enf"],
            "_nolegend_",
            alpha_line=0.45,
            alpha_band=0.12,
            lw=2.0,
            z=3,
        )
        plot_agg(
            ax,
            agg_runs(police, metric, ARM_LABELS["police"] + " cont."),
            ARM_COLORS["police"],
            "_nolegend_",
            alpha_line=0.45,
            alpha_band=0.12,
            lw=2.0,
            z=3,
        )
        plot_agg(
            ax,
            agg_runs(drop_enf, metric, "Drop enf"),
            ARM_COLORS["enf"],
            "Drop enf",
            alpha_line=0.95,
            alpha_band=0.22,
            lw=2.7,
            z=4,
        )
        plot_agg(
            ax,
            agg_runs(drop_police, metric, "Drop police"),
            ARM_COLORS["police"],
            "Drop police",
            alpha_line=0.95,
            alpha_band=0.22,
            lw=2.7,
            z=4,
        )
        drop_vline(ax)

    def plot_peer_gimm(ax, metric):
        plot_baselines(ax, base_hists_g99, metric)
        plot_agg(
            ax,
            agg_runs(peer, metric, ARM_LABELS["peer"] + " cont."),
            ARM_COLORS["peer"],
            "_nolegend_",
            alpha_line=0.45,
            alpha_band=0.12,
            lw=2.0,
            z=3,
        )
        plot_agg(
            ax,
            agg_runs(gimm, metric, ARM_LABELS["gimm"] + " cont."),
            ARM_COLORS["gimm"],
            "_nolegend_",
            alpha_line=0.45,
            alpha_band=0.12,
            lw=2.0,
            z=3,
        )
        plot_agg(
            ax,
            agg_runs(drop_peer, metric, "Drop peer"),
            ARM_COLORS["peer"],
            "Drop peer",
            alpha_line=0.95,
            alpha_band=0.22,
            lw=2.7,
            z=4,
        )
        plot_agg(
            ax,
            agg_runs(drop_gimm, metric, "Drop " + ARM_LABELS["gimm"]),
            ARM_COLORS["gimm"],
            "Drop " + ARM_LABELS["gimm"],
            alpha_line=0.95,
            alpha_band=0.22,
            lw=2.7,
            z=4,
        )
        drop_vline(ax)

    save_set("drop_u200_external_vs_baseline", plot_external)
    save_set("drop_u200_peer_gimm_vs_baseline", plot_peer_gimm)


if __name__ == "__main__":
    main()
