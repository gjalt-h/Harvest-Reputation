"""Per-replica W&B keys for IPPO multi-seed vmap (no JAX).

``NUM_SEEDS==1`` keeps the plain metric name. ``NUM_SEEDS>1`` emits
``metric/seed_i`` only — not a cross-seed mean, not replica 0 alone.
"""
from __future__ import annotations

import numpy as onp

# ToC series logged per vmap replica. Zap/tag recovery = tagged_out_frac.
PER_SEED_TOC_METRICS = (
    "mean_reward",
    "peace",
    "system_peace",
    "sustainability",
    "equality",
    "tagged_out_frac",
)


def pack_per_seed_series(name: str, per_seed, num_seeds: int) -> dict[str, float]:
    """W&B keys for one metric across ``NUM_SEEDS`` replicas.

    ``NUM_SEEDS==1`` → ``{name: value}``.
    ``NUM_SEEDS>1`` → ``{name/seed_0: ..., name/seed_1: ...}`` only — no mean.
    """
    v = onp.asarray(per_seed, dtype=onp.float64).reshape(-1)
    if v.size != int(num_seeds):
        raise ValueError(
            f"pack_per_seed_series({name!r}): got {v.size} values, "
            f"num_seeds={num_seeds}"
        )
    if int(num_seeds) <= 1:
        return {name: float(v[0])}
    return {f"{name}/seed_{i}": float(xi) for i, xi in enumerate(v)}
