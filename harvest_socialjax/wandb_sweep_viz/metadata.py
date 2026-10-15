from __future__ import annotations

import json
import re
from typing import Any


def _unwrap(value: Any) -> Any:
    """Unwrap wandb config Value objects and JSON-ish strings."""
    if value is None:
        return None
    if isinstance(value, dict):
        if set(value.keys()) == {"value"}:
            return _unwrap(value["value"])
        return {str(k): _unwrap(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_unwrap(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        return value
    return value


def flatten_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten nested Hydra config from wandb.run.config."""
    if not config:
        return {}

    flat: dict[str, Any] = {}

    def _walk(prefix: str, obj: Any) -> None:
        obj = _unwrap(obj)
        if isinstance(obj, dict):
            for key, val in obj.items():
                key_s = str(key)
                path = f"{prefix}.{key_s}" if prefix else key_s
                _walk(path, val)
            return
        flat[prefix] = obj

    for key, val in config.items():
        _walk(str(key), val)
    return flat


def _tags(run: Any) -> set[str]:
    tags = getattr(run, "tags", None) or []
    return {str(t) for t in tags}


def is_shared_run(run: Any, flat: dict[str, Any] | None = None) -> bool:
    flat = flat if flat is not None else flatten_config(getattr(run, "config", {}) or {})
    if flat.get("OPTIMIZE_SHARED_REWARDS") is True:
        return True
    name = str(getattr(run, "name", "") or "")
    if name.startswith("ippo_shared"):
        return True
    tags = _tags(run)
    if "shared_rewards" in tags or "shared" in tags:
        return True
    group = str(getattr(run, "group", "") or "")
    if "shared" in group.lower():
        return True
    return False


def _get_nested(flat: dict[str, Any], key: str) -> Any:
    if key in flat:
        return flat[key]
    alt = key.replace(".", "_")
    if alt in flat:
        return flat[alt]
    return None


def _parse_n_from_tags(tags: set[str]) -> int | None:
    for tag in tags:
        m = re.fullmatch(r"n(\d+)", tag)
        if m:
            return int(m.group(1))
    return None


def _parse_n_from_name(name: str) -> int | None:
    m = re.search(r"_n(\d+)_", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"_n(\d+)$", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _parse_apa_token(token: str) -> float:
    return float(token.replace("p", "."))


def _parse_regrowth_apa_from_name(
    name: str,
) -> tuple[float | None, float | None, str | None]:
    rg = None
    apa = None
    axis = None
    m = re.search(r"_regrowth_rg([\dp]+)_", name)
    if m:
        axis = "regrowth"
        rg = float(m.group(1).replace("p", "."))
    m = re.search(r"_density_rg([\dp]+)_apa([\dp]+)_", name)
    if m:
        axis = "density"
        rg = float(m.group(1).replace("p", "."))
        apa = _parse_apa_token(m.group(2))
    m = re.search(r"rg_([\dp]+)", name)
    if rg is None and m:
        rg = float(m.group(1).replace("p", "."))
    # Prefer fractional tokens (24p5) before bare integers (apa24).
    m = re.search(r"apa_?(\d+p\d+)", name)
    if apa is None and m:
        apa = _parse_apa_token(m.group(1))
    m = re.search(r"apa_?(\d+)(?!\d|p)", name)
    if apa is None and m:
        apa = float(m.group(1))
    if "regrowth_sweep" in name or "_regrowth_" in name:
        axis = axis or "regrowth"
    if (
        "density_sweep" in name
        or "_density_" in name
        or "n_and_density" in name
    ):
        axis = axis or "density"
    return rg, apa, axis


def _parse_variant_from_name(name: str) -> str | None:
    # ippo_independent_n4_tag10_25983089 / ippo_shared_n4_beam24x7_shared_25983084
    m = re.search(r"_n4_([A-Za-z0-9]+(?:_shared)?)_\d+$", name)
    if m:
        return m.group(1)
    m = re.search(r"_n4_([A-Za-z0-9]+(?:_shared)?)$", name)
    if m:
        return m.group(1)
    m = re.search(r"_n4_([^_]+)_\d+$", name)
    if m:
        return m.group(1)
    return None


def variant_sort_key(variant: Any) -> float | None:
    """Order tag timeouts numerically; beam variants by length then width."""
    if variant is None:
        return None
    text = str(variant).replace("_shared", "")
    m = re.fullmatch(r"tag(\d+)s?", text)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"beam(\d+)x(\d+)", text)
    if m:
        return 100.0 + float(m.group(1)) + float(m.group(2)) * 0.01
    beam_order = {
        "beam6x1": 106.01,
        "beam8x3": 108.03,
        "beam12x3": 112.03,
        "beam18x5": 118.05,
        "beam24x7": 124.07,
    }
    if text in beam_order:
        return beam_order[text]
    return None


def enrich_run_metadata(run: Any) -> dict[str, Any]:
    """Extract sweep-matching fields from a W&B run."""
    flat = flatten_config(getattr(run, "config", {}) or {})
    tags = _tags(run)
    name = str(getattr(run, "name", "") or "")

    meta: dict[str, Any] = {
        "run_id": getattr(run, "id", None),
        "name": name,
        "group": getattr(run, "group", None),
        "tags": sorted(tags),
        "url": getattr(run, "url", None),
        "state": getattr(run, "state", None),
        "created_at": str(getattr(run, "created_at", "")),
        "shared": is_shared_run(run, flat),
    }

    for key in (
        "ENV_NUM_AGENTS",
        "ENV_KWARGS.regrowth_scale",
        "ENV_KWARGS.apple_density",
        "ENV_KWARGS.apple_layout_mode",
        "ENV_KWARGS.tag_timeout",
        "ENV_KWARGS.beam_length",
        "ENV_KWARGS.beam_width",
        "apples_per_agent",
        "target_apples",
        "sweep_axis",
        "variant",
        "mode",
        "OPTIMIZE_SHARED_REWARDS",
        "OPTIMIZE_BASELINE",
        "EQUALITY_COEF",
        "NUM_SEEDS",
        "SEED",
        "GAMMA",
        "ENT_COEF",
        "TOTAL_TIMESTEPS",
    ):
        val = _get_nested(flat, key)
        if val is not None:
            meta[key] = val

    if meta.get("ENV_NUM_AGENTS") is None:
        n = _parse_n_from_tags(tags) or _parse_n_from_name(name)
        if n is not None:
            meta["ENV_NUM_AGENTS"] = n

    rg, apa, axis = _parse_regrowth_apa_from_name(name)
    if meta.get("ENV_KWARGS.regrowth_scale") is None and rg is not None:
        meta["ENV_KWARGS.regrowth_scale"] = rg
    if meta.get("apples_per_agent") is None and apa is not None:
        meta["apples_per_agent"] = apa
    if meta.get("sweep_axis") is None and axis is not None:
        meta["sweep_axis"] = axis
    if "regrowth_sweep" in tags:
        meta.setdefault("sweep_axis", "regrowth")
    if "density_sweep" in tags or "n_apa_sweep" in tags:
        meta.setdefault("sweep_axis", "density")
    if "fixed98" in tags:
        meta.setdefault("target_apples", 98)

    if meta.get("variant") is None:
        variant = _parse_variant_from_name(name)
        if variant:
            meta["variant"] = variant.replace("_shared", "")
            if variant.endswith("_shared"):
                meta["shared"] = True

    if meta.get("EQUALITY_COEF") is None:
        for tag in tags:
            m = re.fullmatch(r"eq_(\d+(?:p\d+)?)", tag)
            if m:
                meta["EQUALITY_COEF"] = float(m.group(1).replace("p", "."))
                break
            m = re.fullmatch(r"eq_eq(\d+(?:p\d+)?)", tag)  # eq_eq0p5 from sbatch tag
            if m:
                meta["EQUALITY_COEF"] = float(m.group(1).replace("p", "."))
                break
        if meta.get("EQUALITY_COEF") is None:
            m = re.search(r"_eq(\d+p?\d*)(?:_|$)", name)
            if m:
                meta["EQUALITY_COEF"] = float(m.group(1).replace("p", "."))

    dens = meta.get("ENV_KWARGS.apple_density")
    n_agents = meta.get("ENV_NUM_AGENTS")
    if meta.get("apples_per_agent") is None and dens is not None:
        try:
            # Keep fractional apa (24.5), not int-rounded 24.
            meta["apples_per_agent"] = round(float(dens) * 18.0, 2)
        except (TypeError, ValueError):
            pass
    if meta.get("target_apples") is None and dens is not None and n_agents is not None:
        try:
            meta["target_apples"] = int(round(float(dens) * 18.0 * float(n_agents)))
        except (TypeError, ValueError):
            pass

    return meta


def _approx_in(value: Any, allowed: Any, *, tol: float = 0.06) -> bool:
    if value is None:
        return False
    opts = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value in opts
    for opt in opts:
        try:
            if abs(v - float(opt)) <= tol:
                return True
        except (TypeError, ValueError):
            if value == opt:
                return True
    return False


def match_key_tuple(meta: dict[str, Any], match_keys: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(meta.get(k) for k in match_keys)


def passes_filters(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
    tags = set(meta.get("tags") or [])
    for key, expected in filters.items():
        if key == "tags_any":
            needed = expected if isinstance(expected, (list, tuple, set)) else [expected]
            if not any(str(t) in tags for t in needed):
                return False
            continue
        if key == "tags_none":
            banned = expected if isinstance(expected, (list, tuple, set)) else [expected]
            if any(str(t) in tags for t in banned):
                return False
            continue
        if key == "TOTAL_TIMESTEPS_min":
            try:
                if float(meta.get("TOTAL_TIMESTEPS") or 0) < float(expected):
                    return False
            except (TypeError, ValueError):
                return False
            continue
        if key == "apples_per_agent_in":
            if not _approx_in(meta.get("apples_per_agent"), expected):
                return False
            continue
        if key == "target_apples_in":
            if not _approx_in(meta.get("target_apples"), expected, tol=0.5):
                return False
            continue
        if key == "variant_in":
            allowed = expected if isinstance(expected, (list, tuple, set)) else [expected]
            if meta.get("variant") not in allowed:
                return False
            continue
        if key == "variant_prefix":
            variant = str(meta.get("variant") or "")
            if not variant.startswith(str(expected)):
                return False
            continue
        if key in (
            "apples_per_agent",
            "ENV_KWARGS.regrowth_scale",
            "ENV_KWARGS.apple_density",
            "ENV_KWARGS.tag_timeout",
            "ENV_KWARGS.beam_length",
            "ENV_KWARGS.beam_width",
            "TOTAL_TIMESTEPS",
        ):
            if not _approx_in(meta.get(key), [expected]):
                return False
            continue
        if meta.get(key) != expected:
            return False
    return True


def format_composite_sweep_label(meta: dict[str, Any], match_keys: tuple[str, ...]) -> str:
    return " ".join(format_sweep_label(k, meta.get(k)) for k in match_keys)


def composite_sort_key(meta: dict[str, Any], match_keys: tuple[str, ...]) -> float | None:
    """Sort multi-key sweeps (e.g. N then apa)."""
    if match_keys == ("ENV_NUM_AGENTS", "apples_per_agent"):
        try:
            n = float(meta["ENV_NUM_AGENTS"])
            apa = float(meta["apples_per_agent"])
            return n * 1000.0 + apa
        except (KeyError, TypeError, ValueError):
            return None
    return None


def format_sweep_label(sweep_param: str, value: Any) -> str:
    if sweep_param == "ENV_NUM_AGENTS":
        return f"N={value}"
    if sweep_param == "ENV_KWARGS.regrowth_scale":
        return f"rg={value}"
    if sweep_param == "apples_per_agent":
        return f"apa={value}"
    if sweep_param == "variant":
        return str(value)
    if sweep_param == "ENV_KWARGS.tag_timeout":
        return f"tag={value}"
    if sweep_param == "ENV_KWARGS.apple_layout_mode":
        return str(value)
    if sweep_param == "ENV_KWARGS.beam_length":
        return f"beam_L={value}"
    return f"{sweep_param}={value}"
