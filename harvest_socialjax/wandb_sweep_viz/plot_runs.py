"""Load optional W&B run IDs for thesis plot scripts.

Copy ``configs/plot_runs.example.yaml`` → ``configs/plot_runs.yaml`` (repo root
or ``harvest_socialjax/configs/``) and fill in your entity's run IDs.
``plot_runs.yaml`` is gitignored. Scripts default every seed0 id to ``None``
and can still resolve finished runs from W&B *groups*.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # harvest_socialjax/
_REPO_ROOT = _PACKAGE_ROOT.parent
DEFAULT_CANDIDATES = (
    _REPO_ROOT / "configs" / "plot_runs.yaml",
    _PACKAGE_ROOT / "configs" / "plot_runs.yaml",
)
EXAMPLE_PLOT_RUNS = _REPO_ROOT / "configs" / "plot_runs.example.yaml"
EXAMPLE_PLOT_RUNS_PKG = _PACKAGE_ROOT / "configs" / "plot_runs.example.yaml"


def load_plot_runs(path: Path | str | None = None) -> dict[str, Any]:
    """Return nested dict from YAML, or ``{}`` if the file is missing."""
    if path is not None:
        candidates = (Path(path),)
    else:
        candidates = DEFAULT_CANDIDATES
    for cfg_path in candidates:
        if not cfg_path.is_file():
            continue
        with cfg_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"plot runs config must be a mapping: {cfg_path}")
        return data
    return {}


def dig(runs: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Dig ``runs[k1][k2]...``; return ``default`` if any key is missing."""
    node: Any = runs
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def cfg_get(runs: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Like ``dig``, but empty strings / False become ``default`` (for run IDs)."""
    val = dig(runs, *keys, default=default)
    if val is None or val == "" or val is False:
        return default
    return val


def seed0(runs: dict[str, Any], *keys: str, default: str | None = None) -> str | None:
    """Dig ``runs[k1][k2]...['seed0']`` (or a leaf string); usually ``None``."""
    node = dig(runs, *keys, default=None)
    if node is None:
        return default
    if isinstance(node, dict):
        val = node.get("seed0", default)
    else:
        val = node
    if val is None or val == "" or val is False:
        return default
    return str(val)


def add_plot_runs_arg(parser: argparse.ArgumentParser) -> None:
    example = EXAMPLE_PLOT_RUNS if EXAMPLE_PLOT_RUNS.is_file() else EXAMPLE_PLOT_RUNS_PKG
    parser.add_argument(
        "--plot-runs",
        type=Path,
        default=None,
        help=(
            "YAML with W&B seed0 run IDs "
            f"(default: configs/plot_runs.yaml if present; see {example.name})"
        ),
    )


def require_seed0(val: str | None, label: str) -> str:
    """Raise a clear error when a script truly needs an explicit seed0 id."""
    if val:
        return val
    example = EXAMPLE_PLOT_RUNS if EXAMPLE_PLOT_RUNS.is_file() else EXAMPLE_PLOT_RUNS_PKG
    raise SystemExit(
        f"Missing W&B seed0 run id for {label!r}. "
        f"Copy {example.name} → plot_runs.yaml and fill it in, "
        "or pass --plot-runs / CLI overrides."
    )
