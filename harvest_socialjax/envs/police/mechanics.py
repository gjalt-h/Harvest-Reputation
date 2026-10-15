"""Pure police mechanics (jit-compatible helpers)."""
from __future__ import annotations

from typing import Iterable, Sequence

import jax.numpy as jnp
import numpy as onp


def normalize_police_agent_ids(
    police_agent_ids: Sequence[int] | int | None,
    *,
    num_agents: int,
) -> tuple[int, ...]:
    if police_agent_ids is None:
        return ()
    if isinstance(police_agent_ids, int):
        ids = (int(police_agent_ids),)
    else:
        ids = tuple(int(i) for i in police_agent_ids)
    if len(ids) == 0:
        return ()
    seen: set[int] = set()
    for i in ids:
        if i < 0 or i >= num_agents:
            raise ValueError(
                f"police_agent_ids contains invalid index {i} "
                f"for num_agents={num_agents}"
            )
        if i in seen:
            raise ValueError(f"police_agent_ids contains duplicate index {i}")
        seen.add(i)
    return ids


def police_agent_mask(num_agents: int, police_agent_ids: Iterable[int]) -> jnp.ndarray:
    mask = jnp.zeros((num_agents,), dtype=jnp.bool_)
    for i in police_agent_ids:
        mask = mask.at[int(i)].set(True)
    return mask


def mask_police_harvest(
    harvested: jnp.ndarray, police_mask: jnp.ndarray
) -> jnp.ndarray:
    """Zero harvest for police agents (apples must stay on the cell)."""
    return jnp.where(police_mask, jnp.zeros_like(harvested), harvested)


def mask_zap_hits_on_police(
    zap_hit_matrix: jnp.ndarray, police_mask: jnp.ndarray
) -> jnp.ndarray:
    """Police cannot be timed out / tagged by anyone."""
    return zap_hit_matrix & ~police_mask[None, :]


def apply_police_focus_on_bad_zap_hits(
    zap_hit_matrix: jnp.ndarray,
    police_mask: jnp.ndarray,
    target_label: jnp.ndarray,
    *,
    enabled: bool,
) -> jnp.ndarray:
    """Optionally clear police hits on non-Bad targets (label >= 0)."""
    if not enabled:
        return zap_hit_matrix
    bad_target = target_label[None, :] < 0.0
    clear = police_mask[:, None] & ~bad_target
    return zap_hit_matrix & ~clear


def apply_peer_focus_on_bad_zap_hits(
    zap_hit_matrix: jnp.ndarray,
    police_mask: jnp.ndarray,
    target_label: jnp.ndarray,
    *,
    enabled: bool,
) -> jnp.ndarray:
    """Optionally clear peer/harvester hits on non-Bad targets (label >= 0).

    Police rows are left unchanged (use ``police_focus_on_bad`` for those).
    """
    if not enabled:
        return zap_hit_matrix
    bad_target = jnp.asarray(target_label, dtype=jnp.float32)[None, :] < 0.0
    peer_row = ~jnp.asarray(police_mask, dtype=jnp.bool_)[:, None]
    clear = peer_row & ~bad_target
    return zap_hit_matrix & ~clear


def police_binary_zap_reward(
    zap_hit_matrix: jnp.ndarray,
    target_label: jnp.ndarray,
    police_mask: jnp.ndarray,
) -> jnp.ndarray:
    """+1 per Bad hit, −1 per Good hit; 0 for empty/neutral (label == 0).

    Only police rows receive reward; harvester rows are zero.
    """
    hits = zap_hit_matrix.astype(jnp.float32)
    # Bad → +1, Good → −1, empty → 0
    per_target = jnp.where(
        target_label < 0.0,
        jnp.float32(1.0),
        jnp.where(target_label > 0.0, jnp.float32(-1.0), jnp.float32(0.0)),
    )
    per_agent = jnp.sum(hits * per_target[None, :], axis=1)
    return jnp.where(police_mask, per_agent, jnp.float32(0.0))


def harvester_metric_weights(
    num_agents: int,
    police_mask: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.float32]:
    if police_mask is None:
        return jnp.ones((num_agents,), dtype=jnp.float32), jnp.float32(num_agents)
    harvester = (~jnp.asarray(police_mask, dtype=jnp.bool_)).astype(jnp.float32)
    return harvester, jnp.sum(harvester)


def police_agent_indices(police_mask: jnp.ndarray | None) -> tuple[int, ...]:
    if police_mask is None:
        return ()
    pm = onp.asarray(police_mask, dtype=onp.bool_)
    if pm.size == 0 or not onp.any(pm):
        return ()
    return tuple(int(i) for i in onp.where(pm)[0])


def split_beam_victims(
    tag_matrix: jnp.ndarray,
    police_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Victims hit by harvester beams vs police beams (columns of tag matrix)."""
    harvester_hit = jnp.any(tag_matrix & ~police_mask[:, None], axis=0)
    police_hit = jnp.any(tag_matrix & police_mask[:, None], axis=0)
    return harvester_hit, police_hit
