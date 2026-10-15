from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import ModeDef, SweepCatalog, SweepDef
from .metadata import enrich_run_metadata, match_key_tuple, passes_filters


def _ensure_real_wandb_importable() -> None:
    """Avoid shadowing by repo-local wandb_offline_* folders."""
    here = Path(__file__).resolve()
    blocked = {Path.cwd().resolve()}
    for parent in here.parents[:5]:
        blocked.add(parent)
    cleaned = []
    for entry in sys.path:
        if entry in ("", "."):
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            cleaned.append(entry)
            continue
        if resolved in blocked and (resolved / "wandb").is_dir():
            if not (resolved / "wandb" / "__init__.py").exists():
                continue
        cleaned.append(entry)
    sys.path[:] = cleaned


@dataclass
class RunRecord:
    run: Any
    meta: dict[str, Any]
    mode: str  # "independent" | "shared"
    match_key: tuple[Any, ...]
    sweep_value: Any
    sweep_label: str
    sort_key: float | None = None
    is_baseline_reuse: bool = False
    layout_arm: str | None = None  # old | new for layout_compare overlays


def _project_path(catalog: SweepCatalog) -> str:
    entity = catalog.entity or os.environ.get("WANDB_ENTITY")
    if not entity:
        raise ValueError(
            "W&B entity is required. Set WANDB_ENTITY or entity: in sweeps.yaml"
        )
    return f"{entity}/{catalog.project}"


def fetch_runs_for_mode(
    api: Any,
    catalog: SweepCatalog,
    mode_def: ModeDef,
    *,
    state: str | None = "finished",
) -> list[Any]:
    """Fetch runs for a mode.

    New ``harvest_ippo`` jobs use seed-family W&B ``group`` strings and stash the
    sweep label in ``jobType`` / ``sweep_family:*`` tags. Match any of those so
    both legacy and new runs are plottable.
    """
    project = _project_path(catalog)
    seen: dict[str, Any] = {}

    def _ingest(filters: dict[str, Any]) -> None:
        query = dict(filters)
        if state:
            query["state"] = state
        try:
            for run in api.runs(project, filters=query):
                rid = str(getattr(run, "id", "") or id(run))
                seen[rid] = run
        except Exception:  # noqa: BLE001 — W&B filter variants may 400
            return

    _ingest({"group": mode_def.group})
    _ingest({"jobType": mode_def.group})
    _ingest({"tags": f"sweep_family:{mode_def.group}"})
    for extra in mode_def.also_groups:
        _ingest({"group": extra})
    for job in mode_def.also_job_types:
        _ingest({"jobType": job})
        _ingest({"tags": f"sweep_family:{job}"})
    for tag in mode_def.fetch_tags:
        _ingest({"tags": tag})
    return list(seen.values())


def _choose_runs_for_key(
    items: list[tuple[Any, dict[str, Any]]],
    *,
    latest_only: bool,
) -> list[tuple[Any, dict[str, Any]]]:
    """Pick runs for one match key.

    ``latest_only`` keeps the newest finished run **per SEED** so mean±std across
    replicate seed jobs still works (``--all-runs`` keeps every duplicate).
    """
    items = sorted(items, key=lambda item: str(item[1].get("created_at", "")))
    if not latest_only:
        return items
    by_seed: dict[Any, tuple[Any, dict[str, Any]]] = {}
    for run, meta in items:
        seed = meta.get("SEED", 0)
        by_seed[seed] = (run, meta)
    return list(by_seed.values())


def select_runs_for_sweep(
    catalog: SweepCatalog,
    sweep: SweepDef,
    *,
    api: Any | None = None,
    latest_only: bool = True,
    include_running: bool = False,
) -> list[RunRecord]:
    if api is None:
        _ensure_real_wandb_importable()
        import wandb

        api = wandb.Api()

    state = None if include_running else "finished"
    selected: list[RunRecord] = []

    for mode_name, mode_def in (
        ("independent", sweep.independent),
        ("shared", sweep.shared),
    ):
        runs = fetch_runs_for_mode(api, catalog, mode_def, state=state)
        bucket: dict[tuple[Any, ...], list[tuple[Any, dict[str, Any]]]] = {}

        for run in runs:
            meta = enrich_run_metadata(run)
            if mode_name == "independent" and meta.get("shared"):
                continue
            if mode_name == "shared" and not meta.get("shared"):
                continue
            if not passes_filters(meta, sweep.filters):
                continue

            key = match_key_tuple(meta, sweep.match_keys)
            if any(v is None for v in key):
                continue

            if mode_name == "shared" and sweep.shared_pair_variants is not None:
                variant = meta.get("variant") or key[0]
                if str(variant) not in sweep.shared_pair_variants:
                    continue

            bucket.setdefault(key, []).append((run, meta))

        for key, items in bucket.items():
            chosen = _choose_runs_for_key(items, latest_only=latest_only)
            for run, meta in chosen:
                from .metadata import (
                    composite_sort_key,
                    format_composite_sweep_label,
                    format_sweep_label,
                    variant_sort_key,
                )

                if len(sweep.match_keys) > 1:
                    sweep_label = format_composite_sweep_label(meta, sweep.match_keys)
                    default_sort = composite_sort_key(meta, sweep.match_keys)
                else:
                    sweep_value = meta.get(sweep.sweep_param, key[0] if key else None)
                    sweep_label = format_sweep_label(sweep.sweep_param, sweep_value)
                    default_sort = None
                    if sweep.sweep_param == "variant":
                        default_sort = variant_sort_key(sweep_value)
                    if default_sort is None:
                        try:
                            default_sort = float(sweep_value)
                        except (TypeError, ValueError):
                            default_sort = None

                if len(sweep.match_keys) == 1:
                    sweep_value = meta.get(sweep.sweep_param, key[0] if key else None)
                else:
                    sweep_value = key

                selected.append(
                    RunRecord(
                        run=run,
                        meta=meta,
                        mode=mode_name,
                        match_key=key,
                        sweep_value=sweep_value,
                        sweep_label=sweep_label,
                        sort_key=default_sort,
                    )
                )

    return selected


def _resolve_base_keys(
    run: Any,
    metrics: list[str],
    x_axis: str,
    x_fallbacks: Iterable[str],
) -> list[str]:
    available = set()
    try:
        available |= set(getattr(run, "summary", {}).keys())
    except Exception:  # noqa: BLE001
        pass
    try:
        available |= set((getattr(run, "config", {}) or {}).keys())
    except Exception:  # noqa: BLE001
        pass

    candidates = [*metrics, x_axis, *x_fallbacks, "_step"]
    if available:
        base_keys = [k for k in dict.fromkeys(candidates) if k in available or k == "_step"]
        for m in metrics:
            if m not in base_keys:
                base_keys.append(m)
        if x_axis not in base_keys:
            base_keys.append(x_axis)
    else:
        base_keys = list(dict.fromkeys(candidates))
    return base_keys or [*metrics, "_step"]


def fetch_run_history(
    run: Any,
    *,
    metrics: Iterable[str],
    x_axis: str,
    x_fallbacks: Iterable[str],
    samples: int = 5000,
    full: bool = False,
) -> pd.DataFrame:
    """Pull scalar history for plotting.

    ``full=True`` → ``scan_history`` (every logged point).
    Otherwise → ``history(samples=...)`` (fast, downsampled).
    Never request missing keys (W&B returns an empty frame).
    """
    metric_list = list(dict.fromkeys(metrics))
    base_keys = _resolve_base_keys(run, metric_list, x_axis, x_fallbacks)
    df = pd.DataFrame()

    if full and hasattr(run, "scan_history"):
        rows = list(run.scan_history(keys=base_keys))
        df = pd.DataFrame(rows)
    elif hasattr(run, "history"):
        try:
            df = run.history(keys=base_keys, samples=samples, pandas=True)
        except TypeError:
            df = run.history(keys=base_keys, pandas=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  history() failed for {getattr(run, 'name', '?')}: {exc}")

    num_seeds = 1
    try:
        cfg = getattr(run, "config", {}) or {}
        num_seeds = cfg.get("NUM_SEEDS")
        if isinstance(num_seeds, dict) and "value" in num_seeds:
            num_seeds = num_seeds["value"]
        num_seeds = int(num_seeds) if num_seeds is not None else 1
    except (TypeError, ValueError):
        num_seeds = 1

    if num_seeds > 1 and df is not None and not df.empty and metric_list:
        try:
            seed_keys = [f"{m}/seed_{i}" for m in metric_list for i in range(num_seeds)]
            std_keys = [f"{m}_std" for m in metric_list]
            extra_keys = list(dict.fromkeys([*seed_keys, *std_keys, "_step"]))
            if full and hasattr(run, "scan_history"):
                extra = pd.DataFrame(list(run.scan_history(keys=extra_keys)))
            else:
                extra = run.history(keys=extra_keys, samples=samples, pandas=True)
            if extra is not None and not extra.empty:
                merge_on = "_step" if "_step" in df.columns and "_step" in extra.columns else None
                if merge_on:
                    df = df.merge(extra, on=merge_on, how="outer", suffixes=("", "_dup"))
                    drop_dup = [c for c in df.columns if c.endswith("_dup")]
                    if drop_dup:
                        df = df.drop(columns=drop_dup)
                else:
                    for col in extra.columns:
                        if col not in df.columns:
                            df[col] = extra[col]
        except Exception:  # noqa: BLE001
            pass

    if (df is None or df.empty) and not full and hasattr(run, "scan_history"):
        rows = list(run.scan_history(keys=base_keys))
        df = pd.DataFrame(rows)

    if df is None or df.empty:
        return pd.DataFrame()

    x_col = None
    for candidate in (x_axis, *x_fallbacks, "_step"):
        if candidate in df.columns and pd.to_numeric(df[candidate], errors="coerce").notna().any():
            x_col = candidate
            break
    if x_col is None:
        df = df.copy()
        df["_x"] = range(len(df))
        x_col = "_x"
    else:
        df = df.sort_values(x_col).reset_index(drop=True)

    df.attrs["x_col"] = x_col
    return df
