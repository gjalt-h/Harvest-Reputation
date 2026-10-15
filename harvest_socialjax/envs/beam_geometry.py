"""Rectangular zap/tag beam geometry for Commons Harvest: Open."""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Perolat et al. (2017) / Lupu & Precup (2020): length 20, width 5.
# Width must be odd so the beam is centered on the facing axis.
BEAM_LENGTH = 20
BEAM_WIDTH = 5

_STEP4 = jnp.array(
    [[1, 0], [0, 1], [-1, 0], [0, -1]],
    dtype=jnp.int16,
)


def compute_beam_positions(
    agent_locs: jnp.ndarray,
    grid_size_row: int,
    grid_size_col: int,
    beam_length: int = BEAM_LENGTH,
    beam_width: int = BEAM_WIDTH,
) -> jnp.ndarray:
    """Rectangular beam cells ``(row, col)`` for each agent — same as zap.

    The beam is ``beam_length`` cells deep × ``beam_width`` cells across,
    centred on the agent. Depth 0 is the agent's own cell; depths
    ``1 .. beam_length-1`` extend forward. ``beam_hit_matrix`` excludes
    self-hits so zapping never tags the shooter.

    Out-of-bounds cells are left unclipped so they do not match any on-grid
    agent; callers that index the grid must clip / mask.

    Returns:
        ``(num_agents, beam_length * beam_width, 2)`` integer positions.
    """
    del grid_size_row, grid_size_col  # kept for call-site compatibility
    half_w = (beam_width - 1) // 2
    depths = jnp.arange(0, beam_length, dtype=jnp.int16)  # (L,) incl. agent
    laterals = jnp.arange(-half_w, half_w + 1, dtype=jnp.int16)  # (W,)

    def _beam_for_agent(loc: jnp.ndarray) -> jnp.ndarray:
        d = loc[2]
        forward = _STEP4[d]
        right = _STEP4[(d + 1) % 4]
        offsets = (
            depths[:, None, None] * forward[None, None, :]
            + laterals[None, :, None] * right[None, None, :]
        )
        return loc[:2].astype(jnp.int16) + offsets.reshape(-1, 2)

    return jax.vmap(_beam_for_agent)(agent_locs)


def clip_beam_positions(
    beam_positions: jnp.ndarray,
    grid_size_row: int,
    grid_size_col: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Clip beam cells to the grid and return ``(clipped, in_bounds)``."""
    in_bounds = jnp.logical_and(
        jnp.logical_and(beam_positions[..., 0] >= 0, beam_positions[..., 1] >= 0),
        jnp.logical_and(
            beam_positions[..., 0] < grid_size_row,
            beam_positions[..., 1] < grid_size_col,
        ),
    )
    clipped = jnp.stack(
        [
            jnp.clip(beam_positions[..., 0], 0, grid_size_row - 1),
            jnp.clip(beam_positions[..., 1], 0, grid_size_col - 1),
        ],
        axis=-1,
    ).astype(beam_positions.dtype)
    return clipped, in_bounds


def beam_hit_matrix(
    agent_locs: jnp.ndarray,
    beam_positions: jnp.ndarray,
    grid_size_row: int,
    grid_size_col: int,
    active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Zap occupancy: ``(N, N)`` with ``[i, j]`` if j is in i's beam.

    Hits **all** agents whose positions fall in the in-bounds rectangle.
    No wall / first-hit occlusion.

    If ``active`` is provided, inactive (timed-out) agents are not hit.
    Self-hits are always excluded.
    """
    clipped, in_bounds = clip_beam_positions(
        beam_positions, grid_size_row, grid_size_col
    )
    sentinel = jnp.array([-1, -1], dtype=clipped.dtype)
    cells = jnp.where(in_bounds[..., None], clipped, sentinel)
    positions = agent_locs[:, :2].astype(cells.dtype)

    def _mask_for_source(source_cells: jnp.ndarray) -> jnp.ndarray:
        return jnp.any(
            jnp.all(source_cells[:, None, :] == positions[None, :, :], axis=-1),
            axis=0,
        )

    raw = jax.vmap(_mask_for_source)(cells)
    num_agents = positions.shape[0]
    hit = raw & jnp.logical_not(jnp.eye(num_agents, dtype=jnp.bool_))
    if active is not None:
        hit = hit & active.astype(jnp.bool_)[None, :]
    return hit


def agents_hit_by_actions(
    active: jnp.ndarray,
    agent_locs: jnp.ndarray,
    beam_positions: jnp.ndarray,
    grid_size_row: int,
    grid_size_col: int,
) -> jnp.ndarray:
    """``(N,)`` bool — agent j is in at least one active agent's in-bounds beam."""
    hit = beam_hit_matrix(
        agent_locs,
        beam_positions,
        grid_size_row,
        grid_size_col,
        active=active,
    )
    shooters = active.astype(jnp.bool_).reshape(-1)
    return jnp.any(shooters[:, None] & hit, axis=0)
