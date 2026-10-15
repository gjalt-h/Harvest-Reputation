"""
Pure functional core of the reputation system. No env/wrapper state here —
just testable math. All functions are jit-compatible (no Python control flow
on traced values).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ReputationConfig:
    r_star: float = 0.2
    ema_beta: float = 0.99
    decay: float = 0.995
    alpha_h: float = 0.05
    alpha_z: float = 0.15
    # Burst harvest penalty (additive to EMA harvest-rate component).
    # burst_count *= burst_decay each step; +1 if harvested this step.
    # penalty = max(0, burst_count - 1)^2 subtracted as alpha_burst * penalty.
    burst_decay: float = 0.9
    alpha_burst: float = 0.0
    reset_rep_on_tag_respawn: bool = True
    # Stochastic timeout for negative-rep agents (env-side enforcement).
    # When ENABLE_REPUTATION is on, train injects these into Harvest_open.
    enable_enforcement: bool = True
    # P(zap) = clip(-strictness * rep, 0, 1) for rep < 0; else 0.
    # strictness=0 → never; strictness=1 → certain at rep=-1.
    enforcement_strictness: float = 0.01


@struct.dataclass
class ReputationState:
    raw_rep: jnp.ndarray
    rep: jnp.ndarray
    rate_ema: jnp.ndarray
    burst_count: jnp.ndarray


def init_reputation_state(num_agents: int, cfg: ReputationConfig) -> ReputationState:
    return ReputationState(
        raw_rep=jnp.zeros((num_agents,), dtype=jnp.float32),
        rep=jnp.zeros((num_agents,), dtype=jnp.float32),
        rate_ema=jnp.full((num_agents,), cfg.r_star, dtype=jnp.float32),
        burst_count=jnp.zeros((num_agents,), dtype=jnp.float32),
    )


def clamp_enforcement_strictness(strictness: float) -> float:
    """Host-side clamp of strictness into ``[0, 1]``."""
    return float(min(1.0, max(0.0, float(strictness))))


def reputation_enforcement_prob(
    rep: jnp.ndarray,
    strictness: float,
) -> jnp.ndarray:
    """Per-agent probability of a reputation-enforcement timeout.

    Rule (strictness ``s ∈ [0, 1]``, displayed reputation ``rep = tanh(raw)``):

    - ``rep >= 0`` → ``P = 0``
    - ``rep < 0``  → ``P = clip(-s * rep, 0, 1)``

    So ``s = 0`` never punishes; ``s = 1`` and ``rep = -1`` gives ``P = 1``.
    """
    s = jnp.clip(jnp.asarray(strictness, dtype=jnp.float32), 0.0, 1.0)
    p = -s * rep.astype(jnp.float32)
    return jnp.clip(jnp.maximum(p, 0.0), 0.0, 1.0)


def sample_reputation_enforcement(
    key: jax.Array,
    rep: jnp.ndarray,
    strictness: float,
    active: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bernoulli reputation zaps for currently-active agents.

    Returns ``(zapped, probs)`` with shapes matching ``rep``. Already timed-out
    agents (``~active``) never re-trigger; their existing ``time_out`` countdown
    continues unchanged (same as agent-beam tags).
    """
    probs = reputation_enforcement_prob(rep, strictness)
    rolls = jax.random.uniform(key, shape=rep.shape, dtype=jnp.float32)
    zapped = (rolls < probs) & active.astype(jnp.bool_)
    return zapped, probs


def harvest_reputation_delta(
    rate_ema: jnp.ndarray, cfg: ReputationConfig
) -> jnp.ndarray:
    """Normalized harvest deviation; no tanh here (bounded at ``rep = tanh(raw_rep)``)."""
    d_i = (rate_ema - cfg.r_star) / cfg.r_star
    return -d_i


def burst_penalty_schedule(burst_count: jnp.ndarray) -> jnp.ndarray:
    """Quadratic escalation: first harvest in a burst is free, then (n-1)^2."""
    return jnp.square(jnp.maximum(burst_count.astype(jnp.float32) - 1.0, 0.0))


def zap_reputation_delta(
    zap_hit_matrix: jnp.ndarray, rep_prev: jnp.ndarray
) -> jnp.ndarray:
    contribution_matrix = (
        -zap_hit_matrix.astype(jnp.float32) * rep_prev[None, :]
    )
    return jnp.sum(contribution_matrix, axis=1)


def update_reputation(
    state: ReputationState,
    harvested_this_step: jnp.ndarray,
    zap_hit_matrix: jnp.ndarray,
    is_frozen: jnp.ndarray,
    full_reset: jnp.ndarray,
    cfg: ReputationConfig,
    just_respawned: jnp.ndarray | None = None,
) -> ReputationState:
    """Single-step reputation update. Vmap-safe, jit-safe."""
    rate_ema_updated = (
        cfg.ema_beta * state.rate_ema
        + (1.0 - cfg.ema_beta) * harvested_this_step
    )
    rate_ema_new = jnp.where(
        is_frozen, state.rate_ema, rate_ema_updated
    )

    harvested_flag = (harvested_this_step > 0.0).astype(jnp.float32)
    burst_count_updated = (
        cfg.burst_decay * state.burst_count + harvested_flag
    )
    burst_count_new = jnp.where(
        is_frozen, state.burst_count, burst_count_updated
    )
    burst_penalty = burst_penalty_schedule(burst_count_new)

    harvest_component = harvest_reputation_delta(rate_ema_new, cfg)
    zap_component = zap_reputation_delta(zap_hit_matrix, state.rep)

    raw_rep_updated = (
        cfg.decay * state.raw_rep
        + cfg.alpha_h * harvest_component
        + cfg.alpha_z * zap_component
        - cfg.alpha_burst * burst_penalty
    )
    raw_rep_new = jnp.where(is_frozen, state.raw_rep, raw_rep_updated)
    rep_new = jnp.tanh(raw_rep_new)

    raw_rep_new = jnp.where(full_reset, 0.0, raw_rep_new)
    rep_new = jnp.where(full_reset, 0.0, rep_new)
    rate_ema_new = jnp.where(full_reset, cfg.r_star, rate_ema_new)
    burst_count_new = jnp.where(full_reset, 0.0, burst_count_new)

    if just_respawned is not None and cfg.reset_rep_on_tag_respawn:
        respawn_mask = jnp.asarray(just_respawned, dtype=jnp.bool_)
        raw_rep_new = jnp.where(respawn_mask, 0.0, raw_rep_new)
        rep_new = jnp.where(respawn_mask, 0.0, rep_new)
        rate_ema_new = jnp.where(respawn_mask, cfg.r_star, rate_ema_new)
        burst_count_new = jnp.where(respawn_mask, 0.0, burst_count_new)

    return ReputationState(
        raw_rep=raw_rep_new,
        rep=rep_new,
        rate_ema=rate_ema_new,
        burst_count=burst_count_new,
    )
