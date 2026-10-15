"""Environment-side bandit entities (no policy slots).

Bandits imitate permanently Bad agents: random cardinal moves, fixed
``BANDIT_REP`` (−1) reputation in obs, zappable with the same peer/police Bad
reward path as real agents. They never occupy action/policy indices.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

# Match reputation_v2.REP_V2_BAD without a hard import cycle.
BANDIT_REP = jnp.float32(-1.0)


def _parse_int_tuple(val, *, default: tuple[int, ...] = ()) -> tuple[int, ...]:
    """Parse Hydra list / comma-string / tuple into ``tuple[int, ...]``."""
    if val is None or val == "" or val is False:
        return default
    if isinstance(val, (list, tuple)):
        return tuple(int(x) for x in val)
    # OmegaConf ListConfig is not a list/tuple; try iteration first.
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        try:
            return tuple(int(x) for x in val)
        except (TypeError, ValueError):
            pass
    text = str(val).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return default
    return tuple(int(x.strip()) for x in text.split(",") if str(x).strip())


@dataclass(frozen=True)
class BanditConfig:
    """Tunable bandit population / respawn / anneal (ENV_KWARGS)."""

    # Number of bandit slots (static JIT capacity). Additive to real agents.
    bandit_count: int = 0
    # Steps absent after zap before respawn; 0 = instant respawn.
    bandit_respawn_delay: int = 0
    # Global linear schedule for respawn delay (W&B episode_count units).
    bandit_respawn_delay_anneal: bool = False
    # <0 → use static bandit_respawn_delay as start.
    bandit_respawn_delay_start: int = -1
    bandit_respawn_delay_end: int = 0
    bandit_respawn_delay_anneal_episodes: int = 50000
    # Decay of *active* count over global episodes (W&B episode_count).
    bandit_count_anneal: bool = False
    # ``linear`` (start→end) or ``step`` (piecewise plateaus).
    bandit_anneal_mode: str = "linear"
    bandit_count_start: int = -1  # <0 → use bandit_count
    bandit_count_end: int = 0
    bandit_count_anneal_episodes: int = 20000
    # Step schedule (W&B episode_count units). Used when mode=step.
    # Example: counts=(2,1,0), episodes=(15000,15000,20000).
    bandit_step_counts: tuple[int, ...] = ()
    bandit_step_episodes: tuple[int, ...] = ()
    # Align with W&B episode_count when set to NUM_ENVS by the trainer.
    schedule_env_multiplier: int = 1


def bandit_config_from_kwargs(kwargs: dict) -> BanditConfig:
    """Build ``BanditConfig`` from Harvest / Hydra ENV_KWARGS."""
    n = int(kwargs.get("bandit_count", 0) or 0)
    start = kwargs.get("bandit_count_start", None)
    if start is None or int(start) < 0:
        start = n
    mode = str(kwargs.get("bandit_anneal_mode", "linear") or "linear").strip().lower()
    if mode not in ("linear", "step"):
        raise ValueError(
            f"bandit_anneal_mode must be 'linear' or 'step', got {mode!r}"
        )
    step_counts = _parse_int_tuple(kwargs.get("bandit_step_counts"), default=())
    step_eps = _parse_int_tuple(kwargs.get("bandit_step_episodes"), default=())
    if mode == "step" and bool(kwargs.get("bandit_count_anneal", False)):
        if not step_counts or not step_eps:
            raise ValueError(
                "bandit_anneal_mode=step requires bandit_step_counts and "
                "bandit_step_episodes (comma-separated or list)"
            )
        if len(step_counts) != len(step_eps):
            raise ValueError(
                f"bandit_step_counts ({len(step_counts)}) and "
                f"bandit_step_episodes ({len(step_eps)}) length mismatch"
            )
        if any(d < 0 for d in step_eps):
            raise ValueError("bandit_step_episodes must be non-negative")

    static_delay = max(int(kwargs.get("bandit_respawn_delay", 0) or 0), 0)
    delay_start = kwargs.get("bandit_respawn_delay_start", None)
    if delay_start is None or int(delay_start) < 0:
        delay_start = static_delay
    return BanditConfig(
        bandit_count=max(n, 0),
        bandit_respawn_delay=static_delay,
        bandit_respawn_delay_anneal=bool(
            kwargs.get("bandit_respawn_delay_anneal", False)
        ),
        bandit_respawn_delay_start=max(int(delay_start), 0),
        bandit_respawn_delay_end=max(
            int(kwargs.get("bandit_respawn_delay_end", 0) or 0), 0
        ),
        bandit_respawn_delay_anneal_episodes=max(
            int(kwargs.get("bandit_respawn_delay_anneal_episodes", 50000) or 1), 1
        ),
        bandit_count_anneal=bool(kwargs.get("bandit_count_anneal", False)),
        bandit_anneal_mode=mode,
        bandit_count_start=max(int(start), 0),
        bandit_count_end=max(int(kwargs.get("bandit_count_end", 0) or 0), 0),
        bandit_count_anneal_episodes=max(
            int(kwargs.get("bandit_count_anneal_episodes", 20000) or 1), 1
        ),
        bandit_step_counts=tuple(max(int(c), 0) for c in step_counts),
        bandit_step_episodes=tuple(max(int(d), 0) for d in step_eps),
        schedule_env_multiplier=max(
            int(kwargs.get("bandit_schedule_env_multiplier", 1) or 1), 1
        ),
    )


def resolve_bandit_active_count(
    episodes_completed: jnp.ndarray,
    cfg: BanditConfig,
) -> jnp.ndarray:
    """Integer active bandits in ``[0, bandit_count]`` (scalar int32).

    Anneal off: return ``bandit_count``.
    Anneal ``linear``: ``start → end`` over ``anneal_episodes`` global episodes
    (``completed * schedule_env_multiplier``), then hold at ``end``.
    Rounding: nearest integer, half away from zero (``floor(x + 0.5)``).
    Anneal ``step``: piecewise plateaus from ``bandit_step_counts`` lasting
    ``bandit_step_episodes`` each (same global-episode units), then hold last.
    """
    cap = max(int(cfg.bandit_count), 0)
    if cap == 0:
        return jnp.int32(0)
    if not bool(cfg.bandit_count_anneal):
        return jnp.int32(cap)

    completed = jnp.asarray(episodes_completed, dtype=jnp.int32)
    mult = jnp.int32(max(int(cfg.schedule_env_multiplier), 1))
    global_eps = completed * mult

    if str(cfg.bandit_anneal_mode) == "step":
        counts = jnp.asarray(cfg.bandit_step_counts, dtype=jnp.int32)
        durs = jnp.asarray(cfg.bandit_step_episodes, dtype=jnp.int32)
        bounds = jnp.cumsum(durs)
        # side='right': ep==bound_i enters next phase.
        idx = jnp.searchsorted(bounds, global_eps, side="right")
        idx = jnp.minimum(idx, jnp.int32(counts.shape[0] - 1))
        return jnp.clip(counts[idx], jnp.int32(0), jnp.int32(cap))

    horizon = max(int(cfg.bandit_count_anneal_episodes), 1)
    s0 = min(max(float(cfg.bandit_count_start), 0.0), float(cap))
    s1 = min(max(float(cfg.bandit_count_end), 0.0), float(cap))
    t = jnp.clip(
        global_eps.astype(jnp.float32) / jnp.float32(horizon), 0.0, 1.0
    )
    raw = jnp.float32(s0) + (jnp.float32(s1) - jnp.float32(s0)) * t
    rounded = jnp.floor(raw + jnp.float32(0.5)).astype(jnp.int32)
    return jnp.clip(rounded, jnp.int32(0), jnp.int32(cap))


def resolve_bandit_respawn_delay(
    episodes_completed: jnp.ndarray,
    cfg: BanditConfig,
) -> jnp.ndarray:
    """Respawn delay in env steps (scalar int32) after a bandit is zapped.

    Anneal off: return static ``bandit_respawn_delay``.
    Anneal on: linear ``start → end`` over ``bandit_respawn_delay_anneal_episodes``
    global episodes (``completed * schedule_env_multiplier``), then hold at
    ``end``. Rounding: nearest integer, half away from zero.
    """
    static = max(int(cfg.bandit_respawn_delay), 0)
    if not bool(cfg.bandit_respawn_delay_anneal):
        return jnp.int32(static)

    completed = jnp.asarray(episodes_completed, dtype=jnp.int32)
    mult = jnp.int32(max(int(cfg.schedule_env_multiplier), 1))
    global_eps = completed * mult
    horizon = max(int(cfg.bandit_respawn_delay_anneal_episodes), 1)
    s0 = float(max(int(cfg.bandit_respawn_delay_start), 0))
    s1 = float(max(int(cfg.bandit_respawn_delay_end), 0))
    t = jnp.clip(
        global_eps.astype(jnp.float32) / jnp.float32(horizon), 0.0, 1.0
    )
    raw = jnp.float32(s0) + (jnp.float32(s1) - jnp.float32(s0)) * t
    return jnp.maximum(
        jnp.floor(raw + jnp.float32(0.5)).astype(jnp.int32), jnp.int32(0)
    )


def empty_bandit_state(bandit_count: int) -> dict:
    """Zero / empty bandit arrays for ``State``."""
    b = max(int(bandit_count), 0)
    return {
        "bandit_locs": jnp.zeros((b, 3), dtype=jnp.int32),
        "bandit_active": jnp.zeros((b,), dtype=jnp.bool_),
        "bandit_respawn_cd": jnp.zeros((b,), dtype=jnp.int32),
        "bandit_visual_id": jnp.zeros((b,), dtype=jnp.int32),
    }


def cell_blocked(
    r: jnp.ndarray,
    c: jnp.ndarray,
    *,
    grid: jnp.ndarray,
    apples: jnp.ndarray,
    agent_locs: jnp.ndarray,
    agent_active: jnp.ndarray,
    bandit_locs: jnp.ndarray,
    bandit_active: jnp.ndarray,
    wall_id: int,
    grid_h: int,
    grid_w: int,
    exclude_bandit: int = -1,
) -> jnp.ndarray:
    """True if (r,c) is illegal for a bandit."""
    oob = (r < 0) | (c < 0) | (r >= grid_h) | (c >= grid_w)
    r_c = jnp.clip(r, 0, grid_h - 1)
    c_c = jnp.clip(c, 0, grid_w - 1)
    wall = grid[r_c, c_c] == jnp.int16(wall_id)
    apple = apples[r_c, c_c] > 0
    agent_hit = jnp.any(
        agent_active
        & (agent_locs[:, 0] == r_c)
        & (agent_locs[:, 1] == c_c)
    )
    b_idx = jnp.arange(bandit_locs.shape[0])
    b_hit = jnp.any(
        bandit_active
        & (b_idx != jnp.int32(exclude_bandit))
        & (bandit_locs[:, 0] == r_c)
        & (bandit_locs[:, 1] == c_c)
    )
    return oob | wall | apple | agent_hit | b_hit


def sample_legal_bandit_cell(
    key: jax.Array,
    *,
    grid: jnp.ndarray,
    apples: jnp.ndarray,
    agent_locs: jnp.ndarray,
    agent_active: jnp.ndarray,
    bandit_locs: jnp.ndarray,
    bandit_active: jnp.ndarray,
    wall_id: int,
    grid_h: int,
    grid_w: int,
    exclude_bandit: int = -1,
    max_tries: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a random empty cell; returns ``(rc, ok)`` with rc shape (2,)."""

    def body(carry):
        key_i, _rc, _ok, t = carry
        key_i, k1, k2 = jax.random.split(key_i, 3)
        r = jax.random.randint(k1, (), 0, grid_h, dtype=jnp.int32)
        c = jax.random.randint(k2, (), 0, grid_w, dtype=jnp.int32)
        bad = cell_blocked(
            r,
            c,
            grid=grid,
            apples=apples,
            agent_locs=agent_locs,
            agent_active=agent_active,
            bandit_locs=bandit_locs,
            bandit_active=bandit_active,
            wall_id=wall_id,
            grid_h=grid_h,
            grid_w=grid_w,
            exclude_bandit=exclude_bandit,
        )
        rc = jnp.array([r, c], dtype=jnp.int32)
        return key_i, rc, ~bad, t + 1

    def cond(carry):
        _k, _rc, ok, t = carry
        return (~ok) & (t < max_tries)

    key, rc, ok, _ = jax.lax.while_loop(
        cond, body, (key, jnp.zeros((2,), dtype=jnp.int32), False, jnp.int32(0))
    )
    return rc, ok


def init_bandits_on_reset(
    key: jax.Array,
    *,
    cfg: BanditConfig,
    episodes_completed: jnp.ndarray,
    grid: jnp.ndarray,
    apples: jnp.ndarray,
    agent_locs: jnp.ndarray,
    agent_active: jnp.ndarray,
    num_agents: int,
    wall_id: int,
    grid_h: int,
    grid_w: int,
) -> tuple[dict, jax.Array]:
    """Spawn up to ``resolve_bandit_active_count`` bandits on legal cells."""
    b_max = max(int(cfg.bandit_count), 0)
    if b_max == 0:
        return empty_bandit_state(0), key

    n_active = resolve_bandit_active_count(episodes_completed, cfg)
    locs = jnp.zeros((b_max, 3), dtype=jnp.int32)
    cd = jnp.zeros((b_max,), dtype=jnp.int32)
    visual = jnp.zeros((b_max,), dtype=jnp.int32)
    ok_flags = jnp.zeros((b_max,), dtype=jnp.bool_)

    def spawn_one(i, carry):
        key_i, locs_i, vis_i, ok_i = carry
        key_i, k_cell, k_dir, k_vis = jax.random.split(key_i, 4)
        should = jnp.int32(i) < n_active
        already = (jnp.arange(b_max) < i) & ok_i
        rc, ok = sample_legal_bandit_cell(
            k_cell,
            grid=grid,
            apples=apples,
            agent_locs=agent_locs,
            agent_active=agent_active,
            bandit_locs=locs_i,
            bandit_active=already,
            wall_id=wall_id,
            grid_h=grid_h,
            grid_w=grid_w,
            exclude_bandit=i,
        )
        d = jax.random.randint(k_dir, (), 0, 4, dtype=jnp.int32)
        vid = jax.random.randint(
            k_vis, (), 0, max(num_agents, 1), dtype=jnp.int32
        )
        loc = jnp.array([rc[0], rc[1], d], dtype=jnp.int32)
        success = should & ok
        locs_i = locs_i.at[i].set(jnp.where(success, loc, jnp.zeros((3,), jnp.int32)))
        vis_i = vis_i.at[i].set(jnp.where(success, vid, jnp.int32(0)))
        ok_i = ok_i.at[i].set(success)
        return key_i, locs_i, vis_i, ok_i

    key, locs, visual, ok_flags = jax.lax.fori_loop(
        0, b_max, spawn_one, (key, locs, visual, ok_flags)
    )
    return {
        "bandit_locs": locs,
        "bandit_active": ok_flags,
        "bandit_respawn_cd": cd,
        "bandit_visual_id": visual,
    }, key


_CARDINAL = jnp.array(
    [[1, 0], [0, 1], [-1, 0], [0, -1]],
    dtype=jnp.int32,
)


def move_bandits(
    key: jax.Array,
    *,
    bandit_locs: jnp.ndarray,
    bandit_active: jnp.ndarray,
    grid: jnp.ndarray,
    apples: jnp.ndarray,
    agent_locs: jnp.ndarray,
    agent_active: jnp.ndarray,
    wall_id: int,
    grid_h: int,
    grid_w: int,
) -> tuple[jnp.ndarray, jax.Array]:
    """One random cardinal step per active bandit if legal; else stay."""
    b = bandit_locs.shape[0]
    if b == 0:
        return bandit_locs, key

    def body(i, carry):
        key_i, locs_i = carry
        key_i, k_dir = jax.random.split(key_i)
        d = jax.random.randint(k_dir, (), 0, 4, dtype=jnp.int32)
        cur = locs_i[i]
        nr = cur[0] + _CARDINAL[d, 0]
        nc = cur[1] + _CARDINAL[d, 1]
        blocked = cell_blocked(
            nr,
            nc,
            grid=grid,
            apples=apples,
            agent_locs=agent_locs,
            agent_active=agent_active,
            bandit_locs=locs_i,
            bandit_active=bandit_active,
            wall_id=wall_id,
            grid_h=grid_h,
            grid_w=grid_w,
            exclude_bandit=i,
        )
        moved = jnp.array([nr, nc, d], dtype=jnp.int32)
        stayed = cur.at[2].set(d)
        new_loc = jnp.where(
            bandit_active[i] & (~blocked),
            moved,
            jnp.where(bandit_active[i], stayed, cur),
        )
        return key_i, locs_i.at[i].set(new_loc)

    key, locs = jax.lax.fori_loop(0, b, body, (key, bandit_locs))
    return locs, key


def beam_hit_bandits(
    beam_positions: jnp.ndarray,
    bandit_locs: jnp.ndarray,
    bandit_active: jnp.ndarray,
    grid_h: int,
    grid_w: int,
) -> jnp.ndarray:
    """``(N, B)`` True if active bandit j is in agent i's in-bounds beam."""
    from harvest_socialjax.envs.beam_geometry import clip_beam_positions

    b = bandit_locs.shape[0]
    n = beam_positions.shape[0]
    if b == 0:
        return jnp.zeros((n, 0), dtype=jnp.bool_)

    clipped, in_bounds = clip_beam_positions(beam_positions, grid_h, grid_w)
    sentinel = jnp.array([-1, -1], dtype=clipped.dtype)
    cells = jnp.where(in_bounds[..., None], clipped, sentinel)
    positions = bandit_locs[:, :2].astype(cells.dtype)

    def _mask_for_source(source_cells: jnp.ndarray) -> jnp.ndarray:
        return jnp.any(
            jnp.all(source_cells[:, None, :] == positions[None, :, :], axis=-1),
            axis=0,
        )

    raw = jax.vmap(_mask_for_source)(cells)
    return raw & bandit_active[None, :]


def bandit_occupancy_grid(
    bandit_locs: jnp.ndarray,
    bandit_active: jnp.ndarray,
    grid_h: int,
    grid_w: int,
) -> jnp.ndarray:
    """HxW bool: cells occupied by an active bandit."""
    occ = jnp.zeros((grid_h, grid_w), dtype=jnp.bool_)
    b = bandit_locs.shape[0]
    if b == 0:
        return occ

    def body(i, occ_i):
        r, c = bandit_locs[i, 0], bandit_locs[i, 1]
        return occ_i.at[r, c].set(
            jnp.where(bandit_active[i], True, occ_i[r, c])
        )

    return jax.lax.fori_loop(0, b, body, occ)


def paint_bandits_on_grid(
    grid: jnp.ndarray,
    *,
    bandit_locs: jnp.ndarray,
    bandit_active: jnp.ndarray,
    bandit_visual_id: jnp.ndarray,
    agent_item_base: int,
) -> jnp.ndarray:
    """Paint active bandits as agent-grid IDs (``base + visual_id``)."""
    b = bandit_locs.shape[0]
    if b == 0:
        return grid

    def body(i, g):
        r, c = bandit_locs[i, 0], bandit_locs[i, 1]
        val = jnp.int16(agent_item_base + bandit_visual_id[i])
        return g.at[r, c].set(jnp.where(bandit_active[i], val, g[r, c]))

    return jax.lax.fori_loop(0, b, body, grid)


def clear_bandits_from_grid(
    grid: jnp.ndarray,
    *,
    bandit_locs: jnp.ndarray,
    bandit_active: jnp.ndarray,
    apples: jnp.ndarray,
    agent_item_base: int,
    bandit_visual_id: jnp.ndarray,
) -> jnp.ndarray:
    """Clear bandit markers; restore apple under cell if present."""
    b = bandit_locs.shape[0]
    if b == 0:
        return grid

    def body(i, g):
        r, c = bandit_locs[i, 0], bandit_locs[i, 1]
        expected = jnp.int16(agent_item_base + bandit_visual_id[i])
        is_me = (g[r, c] == expected) & bandit_active[i]
        under = jnp.where(
            apples[r, c] == 1,
            jnp.int16(3),  # Items.apple
            jnp.int16(0),
        )
        return g.at[r, c].set(jnp.where(is_me, under, g[r, c]).astype(jnp.int16))

    return jax.lax.fori_loop(0, b, body, grid)


def apply_bandit_zap_respawn(
    key: jax.Array,
    *,
    bandit_locs: jnp.ndarray,
    bandit_active: jnp.ndarray,
    bandit_respawn_cd: jnp.ndarray,
    bandit_visual_id: jnp.ndarray,
    zapped: jnp.ndarray,
    respawn_delay: int | jnp.ndarray,
    grid: jnp.ndarray,
    apples: jnp.ndarray,
    agent_locs: jnp.ndarray,
    agent_active: jnp.ndarray,
    num_agents: int,
    wall_id: int,
    grid_h: int,
    grid_w: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]:
    """Handle zap hits + countdown ticks for bandits.

    ``zapped`` is ``(B,)`` bool — bandit was hit this step.
    ``respawn_delay`` may be a Python int or a scalar JAX int (annealed).
    """
    b = bandit_locs.shape[0]
    if b == 0:
        return (
            bandit_locs,
            bandit_active,
            bandit_respawn_cd,
            bandit_visual_id,
            key,
        )

    delay = jnp.maximum(
        jnp.asarray(respawn_delay, dtype=jnp.int32), jnp.int32(0)
    )

    cd = jnp.maximum(bandit_respawn_cd - 1, 0)
    ready = (~bandit_active) & (bandit_respawn_cd > 0) & (cd == 0)

    active = bandit_active & (~zapped)
    # delay==0 → instant respawn this step; else countdown then spawn when ready.
    cd = jnp.where(zapped, delay, cd)
    need_spawn = jnp.where(delay == 0, zapped | ready, ready)

    def spawn_body(i, carry):
        key_i, locs_i, act_i, cd_i, vis_i = carry
        key_i, k_cell, k_dir, k_vis = jax.random.split(key_i, 4)
        do = need_spawn[i]
        rc, ok = sample_legal_bandit_cell(
            k_cell,
            grid=grid,
            apples=apples,
            agent_locs=agent_locs,
            agent_active=agent_active,
            bandit_locs=locs_i,
            bandit_active=act_i,
            wall_id=wall_id,
            grid_h=grid_h,
            grid_w=grid_w,
            exclude_bandit=i,
        )
        d = jax.random.randint(k_dir, (), 0, 4, dtype=jnp.int32)
        vid = jax.random.randint(
            k_vis, (), 0, max(num_agents, 1), dtype=jnp.int32
        )
        loc = jnp.array([rc[0], rc[1], d], dtype=jnp.int32)
        success = do & ok
        locs_i = locs_i.at[i].set(jnp.where(success, loc, locs_i[i]))
        vis_i = vis_i.at[i].set(jnp.where(success, vid, vis_i[i]))
        act_i = act_i.at[i].set(jnp.where(success, True, act_i[i]))
        cd_i = cd_i.at[i].set(
            jnp.where(
                do & (~ok),
                jnp.int32(1),
                jnp.where(success, jnp.int32(0), cd_i[i]),
            )
        )
        act_i = act_i.at[i].set(jnp.where(do & (~ok), False, act_i[i]))
        return key_i, locs_i, act_i, cd_i, vis_i

    key, locs, active, cd, visual = jax.lax.fori_loop(
        0,
        b,
        spawn_body,
        (key, bandit_locs, active, cd, bandit_visual_id),
    )
    return locs, active, cd, visual, key


def extended_zap_labels_and_hits(
    zap_hit_matrix: jnp.ndarray,
    agent_labels: jnp.ndarray,
    bandit_hit_matrix: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Concat agent + bandit targets for ``peer_punish_zap_reward`` / police.

    Bandit columns use ``BANDIT_REP``. Returns ``(hits_ext, labels_ext)``.
    """
    n = zap_hit_matrix.shape[0]
    b = bandit_hit_matrix.shape[1] if bandit_hit_matrix.ndim == 2 else 0
    if b == 0:
        return zap_hit_matrix, agent_labels
    labels_b = jnp.full((b,), BANDIT_REP, dtype=jnp.float32)
    labels_ext = jnp.concatenate([agent_labels.astype(jnp.float32), labels_b], axis=0)
    hits_ext = jnp.concatenate(
        [zap_hit_matrix.astype(jnp.bool_), bandit_hit_matrix.astype(jnp.bool_)],
        axis=1,
    )
    return hits_ext, labels_ext
