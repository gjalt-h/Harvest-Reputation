from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SWEEPS_FILE = PACKAGE_DIR / "sweeps.yaml"


@dataclass(frozen=True)
class ModeDef:
    group: str
    mode_tag: str
    # Extra W&B identities for seed-family / multi-sbatch sweeps.
    also_job_types: tuple[str, ...] = ()
    also_groups: tuple[str, ...] = ()
    fetch_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaselineMatchDef:
    match: dict[str, Any]
    source_groups: tuple[str, ...]
    also_job_types: tuple[str, ...] = ()
    fetch_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class N4BaselineDef:
    independent: BaselineMatchDef | None = None
    shared: BaselineMatchDef | None = None


@dataclass(frozen=True)
class BaselineInsertDef:
    label: str
    sort_key: float
    param: str | None = None
    value: Any = None
    position: str = "middle"  # middle | always
    when_variants: tuple[str, ...] | None = None


@dataclass(frozen=True)
class BaselineMarkDef:
    """Relabel an already-selected sweep arm so it draws as the black baseline."""

    match: dict[str, Any]
    label: str


@dataclass(frozen=True)
class CompareArmDef:
    id: str
    label_suffix: str
    source_sweep: str
    mode: str = "independent"
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LayoutCompareDef:
    match_keys: tuple[str, ...]
    arms: tuple[CompareArmDef, ...]


@dataclass(frozen=True)
class BaselineEqCompareDef:
    eq_values: tuple[float, ...]
    n_agents: int
    baseline_group: str
    source_sweep: str = "n_sweep"


@dataclass(frozen=True)
class SweepDef:
    sweep_id: str
    title: str
    subtitle: str
    description: str
    independent: ModeDef
    shared: ModeDef
    sweep_param: str
    sweep_label: str
    match_keys: tuple[str, ...]
    filters: dict[str, Any] = field(default_factory=dict)
    shared_pair_variants: tuple[str, ...] | None = None
    sbatch: dict[str, str] = field(default_factory=dict)
    baseline: dict[str, str] = field(default_factory=dict)
    baseline_inserts: tuple[BaselineInsertDef, ...] = ()
    baseline_marks: tuple[BaselineMarkDef, ...] = ()
    baseline_profile: str = "canonical"  # canonical | plus98
    plot_mode: str = "standard"  # standard | layout_compare | baseline_eq_compare
    compare: LayoutCompareDef | None = None
    baseline_eq: BaselineEqCompareDef | None = None
    metrics: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SweepCatalog:
    project: str
    entity: str | None
    default_metrics: tuple[str, ...]
    x_axis: str
    x_axis_fallbacks: tuple[str, ...]
    sweeps: dict[str, SweepDef]
    n4_baseline: N4BaselineDef | None = None
    n4_plus98_baseline: N4BaselineDef | None = None


def _parse_mode(raw: dict[str, Any] | None) -> ModeDef:
    if not raw:
        raise ValueError("mode definition requires group and mode_tag")
    return ModeDef(
        group=str(raw["group"]),
        mode_tag=str(raw["mode_tag"]),
        also_job_types=tuple(raw.get("also_job_types") or ()),
        also_groups=tuple(raw.get("also_groups") or ()),
        fetch_tags=tuple(raw.get("fetch_tags") or ()),
    )


def _parse_baseline_match(raw: dict[str, Any] | None) -> BaselineMatchDef | None:
    if not raw:
        return None
    return BaselineMatchDef(
        match=dict(raw.get("match") or {}),
        source_groups=tuple(raw.get("source_groups") or ()),
        also_job_types=tuple(raw.get("also_job_types") or ()),
        fetch_tags=tuple(raw.get("fetch_tags") or ()),
    )


def _parse_baseline_marks(raw: list[dict[str, Any]] | None) -> tuple[BaselineMarkDef, ...]:
    if not raw:
        return ()
    return tuple(
        BaselineMarkDef(
            match=dict(item.get("match") or {}),
            label=str(item.get("label") or "baseline"),
        )
        for item in raw
    )


def _parse_n4_baseline_block(raw: dict[str, Any] | None) -> N4BaselineDef | None:
    if not raw:
        return None
    return N4BaselineDef(
        independent=_parse_baseline_match(raw.get("independent")),
        shared=_parse_baseline_match(raw.get("shared")),
    )


def _parse_compare(raw: dict[str, Any] | None) -> LayoutCompareDef | None:
    if not raw:
        return None
    arms = []
    for item in raw.get("arms") or []:
        arms.append(
            CompareArmDef(
                id=str(item["id"]),
                label_suffix=str(item.get("label_suffix", item["id"])),
                source_sweep=str(item["source_sweep"]),
                mode=str(item.get("mode", "independent")),
                filters=dict(item.get("filters") or {}),
            )
        )
    return LayoutCompareDef(
        match_keys=tuple(raw.get("match_keys") or ()),
        arms=tuple(arms),
    )


def _parse_baseline_eq(raw: dict[str, Any] | None) -> BaselineEqCompareDef | None:
    if not raw:
        return None
    eq_raw = raw.get("eq_values") or [1, 10, 100]
    return BaselineEqCompareDef(
        eq_values=tuple(float(v) for v in eq_raw),
        n_agents=int(raw.get("n_agents", 4)),
        baseline_group=str(raw["baseline_group"]),
        source_sweep=str(raw.get("source_sweep", "n_sweep")),
    )


def _parse_baseline_inserts(raw: list[dict[str, Any]] | None) -> tuple[BaselineInsertDef, ...]:
    if not raw:
        return ()
    inserts = []
    for item in raw:
        when = item.get("when_variants")
        inserts.append(
            BaselineInsertDef(
                label=str(item.get("label", "")),
                sort_key=float(item["sort_key"]),
                param=item.get("param"),
                value=item.get("value"),
                position=str(item.get("position", "middle")),
                when_variants=tuple(when) if when else None,
            )
        )
    return tuple(inserts)


def load_sweep_catalog(path: Path | None = None) -> SweepCatalog:
    cfg_path = path or DEFAULT_SWEEPS_FILE
    with cfg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    entity = raw.get("entity") or os.environ.get("WANDB_ENTITY")
    n4_baseline = _parse_n4_baseline_block(raw.get("n4_canonical_baseline"))
    n4_plus98_baseline = _parse_n4_baseline_block(raw.get("n4_plus98_baseline"))

    sweeps: dict[str, SweepDef] = {}
    for sweep_id, spec in (raw.get("sweeps") or {}).items():
        inserts_raw = spec.get("baseline_inserts")
        if inserts_raw is None and spec.get("baseline_insert"):
            inserts_raw = [spec["baseline_insert"]]
        marks_raw = spec.get("baseline_marks")
        if marks_raw is None and spec.get("baseline_mark"):
            marks_raw = [spec["baseline_mark"]]
        sweeps[sweep_id] = SweepDef(
            sweep_id=sweep_id,
            title=str(spec["title"]),
            subtitle=str(spec.get("subtitle", "")),
            description=str(spec.get("description", "")).strip(),
            independent=_parse_mode(spec["independent"]),
            shared=_parse_mode(spec["shared"]),
            sweep_param=str(spec["sweep_param"]),
            sweep_label=str(spec.get("sweep_label", spec["sweep_param"])),
            match_keys=tuple(spec.get("match_keys", [spec["sweep_param"]])),
            filters=dict(spec.get("filters") or {}),
            shared_pair_variants=tuple(spec["shared_pair_variants"])
            if spec.get("shared_pair_variants")
            else None,
            sbatch=dict(spec.get("sbatch") or {}),
            baseline=dict(spec.get("baseline") or {}),
            baseline_inserts=_parse_baseline_inserts(inserts_raw),
            baseline_marks=_parse_baseline_marks(marks_raw),
            baseline_profile=str(spec.get("baseline_profile", "canonical")),
            plot_mode=str(spec.get("plot_mode", "standard")),
            compare=_parse_compare(spec.get("compare")),
            baseline_eq=_parse_baseline_eq(spec.get("baseline_eq")),
            metrics=tuple(spec["metrics"]) if spec.get("metrics") else None,
        )

    return SweepCatalog(
        project=str(raw.get("project", "harvest_ippo")),
        entity=entity,
        default_metrics=tuple(raw.get("default_metrics") or ["mean_reward"]),
        x_axis=str(raw.get("x_axis", "episode_count")),
        x_axis_fallbacks=tuple(
            raw.get("x_axis_fallbacks") or ("update", "env_steps", "_step")
        ),
        sweeps=sweeps,
        n4_baseline=n4_baseline,
        n4_plus98_baseline=n4_plus98_baseline,
    )
