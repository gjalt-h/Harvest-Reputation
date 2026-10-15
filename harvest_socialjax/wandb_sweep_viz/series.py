from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# Derived W&B plot metrics: total zap attempts/hits per episode from rates.
# Harvester rate = sum(events) / (ep_len * N_h) → events/ep ≈ rate * ep_len * N_h.
# Police rate = sum(events) / (ep_len * N_police) → events/ep ≈ rate * ep_len * N_p.
DERIVED_FROM_RATE: dict[str, str] = {
    "zap_attempts_per_episode": "zap_attempt_rate",
    "zap_hits_per_episode": "zap_success_rate",
    "enforcement/police_zap_attempts_per_episode": "enforcement/police_zap_attempt_rate",
}

# Display names for axis labels, titles, and HTML figcaptions.
# W&B / history keys stay unchanged (e.g. mean_reward); only the visible label changes.
METRIC_YLABELS: dict[str, str] = {
    "mean_reward": "Utility (U)",
    "peace": "Peace (P)",
    "sustainability": "Sustainability (S)",
    "equality": "Equality (E)",
    "system_peace": "System Peace (SP)",

    "tagged_out_frac": "Tagged-out fraction",

    "zap_attempt_rate": "Zap attempt rate",
    "zap_success_rate": "Zap success rate",
    "zap_hit_rate": "Zap hit rate (given attempt)",
    "zap_attempts_per_episode": "Zap attempts",
    "zap_hits_per_episode": "Zap hits / episode",
    "zap_tagged_rep_mean": "Mean zapped reputation",
    "enforcement/police_zap_tagged_rep_mean": "Police zapped rep mean",
    "enforcement/police_zap_attempt_rate": "Police zap attempt rate",
    "enforcement/police_zap_attempts_per_episode": "Police zap attempts",
    "enforcement/police_zap_hit_rate": "Police zap hit rate",
    "enforcement/police_reward": "Police reward",
    "mean_rep": "Mean reputation",
}


def resolve_history_metrics(metrics: list[str]) -> list[str]:
    """Expand derived metric names to the raw W&B keys that must be fetched."""
    out: list[str] = []
    for metric in metrics:
        src = DERIVED_FROM_RATE.get(metric, metric)
        if src not in out:
            out.append(src)
        if metric not in DERIVED_FROM_RATE and metric not in out:
            out.append(metric)
    return out


def _episode_length(meta: dict[str, Any] | None) -> float:
    meta = meta or {}
    t = meta.get("num_inner_steps")
    if t is None:
        env_kw = meta.get("ENV_KWARGS")
        if isinstance(env_kw, dict):
            t = env_kw.get("num_inner_steps")
        else:
            t = meta.get("ENV_KWARGS.num_inner_steps")
    try:
        return float(t) if t is not None else 1000.0
    except (TypeError, ValueError):
        return 1000.0


def episode_zap_scale(meta: dict[str, Any] | None) -> float:
    """Scale harvester zap rate → total attempts/hits per episode."""
    meta = meta or {}
    try:
        n = float(meta.get("ENV_NUM_AGENTS") or 4)
    except (TypeError, ValueError):
        n = 4.0
    # Police agent is excluded from harvester zap-rate denominators.
    try:
        if meta.get("ENABLE_POLICE_AGENT") or meta.get("NUM_POLICE_AGENTS"):
            n_police = float(meta.get("NUM_POLICE_AGENTS") or 1)
            n = max(n - n_police, 1.0)
    except (TypeError, ValueError):
        pass
    return max(n * _episode_length(meta), 1.0)


def episode_police_zap_scale(meta: dict[str, Any] | None) -> float:
    """Scale police zap rate → total police attempts per episode."""
    meta = meta or {}
    try:
        n_police = float(meta.get("NUM_POLICE_AGENTS") or 1)
    except (TypeError, ValueError):
        n_police = 1.0
    return max(n_police * _episode_length(meta), 1.0)


@dataclass(frozen=True)
class AggregatedSeries:
    x: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n: np.ndarray
    metric: str
    label: str
    mode: str  # independent | shared
    run_ids: tuple[str, ...]


def _seed_columns(df: pd.DataFrame, metric: str) -> list[str]:
    cols = sorted(
        [c for c in df.columns if c.startswith(f"{metric}/seed_")],
        key=lambda s: int(s.rsplit("_", 1)[-1]),
    )
    return cols


def extract_series_from_history(
    df: pd.DataFrame,
    *,
    metric: str,
    x_col: str,
    label: str,
    mode: str,
    run_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> AggregatedSeries | None:
    if df.empty:
        return None

    source_metric = DERIVED_FROM_RATE.get(metric, metric)
    if metric in DERIVED_FROM_RATE:
        if metric.startswith("enforcement/police"):
            scale = episode_police_zap_scale(meta)
        else:
            scale = episode_zap_scale(meta)
    else:
        scale = 1.0

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    seed_cols = _seed_columns(df, source_metric)

    if seed_cols:
        values = df[seed_cols].apply(pd.to_numeric, errors="coerce")
        mean = values.mean(axis=1).to_numpy(dtype=float) * scale
        std = values.std(axis=1, ddof=0).fillna(0.0).to_numpy(dtype=float) * scale
        n = values.notna().sum(axis=1).to_numpy(dtype=int)
    elif source_metric in df.columns:
        mean = pd.to_numeric(df[source_metric], errors="coerce").to_numpy(dtype=float)
        mean = mean * scale
        std_col = f"{source_metric}_std"
        if std_col in df.columns:
            std = (
                pd.to_numeric(df[std_col], errors="coerce").fillna(0.0).to_numpy(
                    dtype=float
                )
                * scale
            )
        else:
            std = np.zeros_like(mean)
        n = np.where(np.isnan(mean), 0, 1).astype(int)
    else:
        return None

    valid = ~np.isnan(mean) & ~np.isnan(x)
    if not valid.any():
        return None

    return AggregatedSeries(
        x=x[valid],
        mean=mean[valid],
        std=std[valid],
        n=n[valid],
        metric=metric,
        label=label,
        mode=mode,
        run_ids=(run_id,) if run_id else (),
    )


def aggregate_across_runs(
    series_list: list[AggregatedSeries],
    *,
    metric: str,
    label: str,
    mode: str,
) -> AggregatedSeries | None:
    """Combine multiple runs of the same setup (e.g. different seeds as separate jobs)."""
    if not series_list:
        return None
    if len(series_list) == 1:
        return series_list[0]

    x_union = np.unique(np.concatenate([s.x for s in series_list]))
    x_union.sort()
    means = []
    stds = []
    counts = []

    for x_val in x_union:
        vals = []
        for series in series_list:
            idx = np.where(np.isclose(series.x, x_val, rtol=0, atol=0))[0]
            if idx.size:
                vals.append(float(series.mean[idx[0]]))
        if not vals:
            means.append(np.nan)
            stds.append(0.0)
            counts.append(0)
        else:
            arr = np.asarray(vals, dtype=float)
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr, ddof=0)) if arr.size > 1 else 0.0)
            counts.append(arr.size)

    mean_arr = np.asarray(means, dtype=float)
    valid = ~np.isnan(mean_arr)
    if not valid.any():
        return None

    run_ids = tuple(dict.fromkeys(rid for s in series_list for rid in s.run_ids))
    return AggregatedSeries(
        x=x_union[valid],
        mean=mean_arr[valid],
        std=np.asarray(stds, dtype=float)[valid],
        n=np.asarray(counts, dtype=int)[valid],
        metric=metric,
        label=label,
        mode=mode,
        run_ids=run_ids,
    )


# Shared x-axis caption for thesis training curves.
EPISODE_XLABEL = "Episode count"


def metric_ylabel(metric: str) -> str:
    """Human-readable caption for a metric key (axis label / title / figcaption)."""
    if metric in METRIC_YLABELS:
        return METRIC_YLABELS[metric]
    # Fallback: title-case path segments (e.g. enforcement/police_peace).
    parts = metric.replace("-", " ").split("/")
    return " / ".join(p.replace("_", " ").strip().title() for p in parts if p)
