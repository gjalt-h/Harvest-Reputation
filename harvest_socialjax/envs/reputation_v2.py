"""Reputation v2: rolling-window Bad labels + optional delayed/periodic enforcement.

Labels painted on spatial ch15/ch16 via ``tag_filter_rep``:

- empty cell: ``0``
- Binary mode: Good ``+1``, Bad ``−1``
- Graded mode (``graded_reputation``): Bad ``−1``; Good ``+1,+2,+3,…``
  (never 0 on an occupied agent cell so empty ≠ Bad)

Binary label rules (pre-step labels for zap targets):

1. All agents start Good (+1).
2. Burst: window sum > max_harvests → Bad.
3. Local-last apple (optional): L2≤2 neighbor L=0 → Bad.
4. Zap (if ``zap_affects_label``): hit only Bad → zapper Good. Hitting Good
   marks Bad by default (any Good in beam). With
   ``zap_to_bad_only_pure_good=true``, hitting Good marks Bad only when the
   beam contains **no** Bad targets (mixed Good+Bad leaves label unchanged
   by zap). Set ``zap_affects_label=false`` for harvest-only v3 (labels
   ignore zaps; recovery only via respawn / episode reset).
5. On respawn (optional): label resets to Good.

Graded label rules (same triggers, integer stack):

1. Start at Good (+1).
2. While Good: each Bad agent hit → ``+1``; each Bad act (burst / last-apple /
   Good-hit zap per binary rule above) → ``−1``. At ``+1``, a Bad act → ``−1``.
3. While Bad: stay ``−1`` on further Bad acts; hit only Bad (no Good, no burst /
   last-apple) → rehabilitate to ``+1``.
4. Respawn / episode reset → ``+1`` (optional).
5. With ``zap_affects_label=false``: ignore zap hits (no +1 / rehab / Good-hit
   Bad acts); only burst / last-apple change labels.

Enforcement (optional, separate from labeling):

- Immediate: freeze on the burst/last-apple step (legacy).
- Per-agent ``timeout_delay``: freeze exactly N steps after becoming Bad.
- Periodic: every ``enforcement_period`` steps, freeze all Bad agents that have
  been Bad for at least ``enforcement_delay`` steps.
- Episode schedules: ``delay_ramp_*`` raises delay over the first ~10k W&B
  episodes; ``enforcement_stop_after_episodes`` turns freezes off after a
  global episode threshold.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from flax import struct

from harvest_socialjax.envs.harvest_open import TAG_TIMEOUT_STEPS

REP_V2_EMPTY = 0.0
REP_V2_GOOD = 1.0
REP_V2_BAD = -1.0


def max_harvests_for_r_star(r_star: float, window_len: int) -> int:
    """Largest harvest count in a window that is still sustainable at ``r_star``."""
    return int(jnp.floor(float(r_star) * int(window_len)))


@struct.dataclass
class BurstWindowConfig:
    window_len: int = 10
    max_harvests_in_window: int = 2
    timeout_duration: int = TAG_TIMEOUT_STEPS
    enforce_mode: str = "freeze"
    # Local-last apple → Bad label (and immediate freeze only if immediate_timeout).
    last_apple_timeout: bool = False
    # Reset label to Good when tag timeout expires / agent respawns.
    reset_label_on_respawn: bool = True
    # Legacy: freeze on the same step as burst / local-last Bad labeling.
    immediate_timeout: bool = False
    # Per-agent: freeze exactly this many steps after becoming Bad (0 = off).
    timeout_delay: int = 0
    # Periodic Bad enforcement (timeout all eligible Bad agents).
    enforcement_enabled: bool = True
    # Steps labeled Bad before eligible for periodic enforcement.
    enforcement_delay: int = 0
    # Env steps between periodic Bad sweeps (0 = off).
    enforcement_period: int = 0
    # --- Episode-indexed schedules (see ``resolve_enforcement_schedule``) ---
    # Ramp timeout_delay every ``delay_ramp_every_episodes`` *global* episodes
    # (global ≈ per-env completions × schedule_env_multiplier). Phase 0 can be
    # immediate freeze; then delay=1,2,… up to delay_ramp_max_delay.
    delay_ramp_enabled: bool = False
    delay_ramp_every_episodes: int = 1000
    delay_ramp_max_delay: int = 9
    delay_ramp_start_immediate: bool = True
    # Hold-then-ramp curriculum (when delay_hold_episodes > 0):
    #   [0, hold): fixed delay = delay_hold_start_delay
    #   [hold, delay_ramp_end_episodes): delay rises by +1 every
    #     delay_ramp_every_episodes (capped at delay_ramp_max_delay)
    #   [delay_ramp_end_episodes, ∞): if delay_curriculum_end_immediate →
    #     immediate freeze; if delay_curriculum_end_disable_enforcement →
    #     no system freezes (peer/police zaps only). Prefer disable over
    #     immediate when both are set.
    # Legacy ramp (hold=0) ignores these fields.
    delay_hold_episodes: int = 0
    delay_hold_start_delay: int = 0
    delay_ramp_end_episodes: int = 0  # 0 = never enter end phase
    delay_curriculum_end_immediate: bool = False
    delay_curriculum_end_disable_enforcement: bool = False
    # After this many *global* episodes, disable all freezes (0 = never).
    # Also used alone; curriculum end_disable ORs with this threshold.
    enforcement_stop_after_episodes: int = 0
    # Set to NUM_ENVS by the trainer so schedules track W&B episode_count.
    schedule_env_multiplier: int = 1
    # Graded Good stack (−1 Bad; +1,+2,+3,… Good). False = binary ±1.
    graded_reputation: bool = False
    # Cap on Good levels (0 = uncapped). Only used when graded_reputation.
    graded_max_reputation: int = 0
    # Peer punish (no police): +bad_reward per Bad in beam; good_penalty per Good.
    peer_punish_reward_enabled: bool = False
    peer_punish_bad_reward: float = 1.0
    # 0 = off; typically −bad_reward for matched Good-zap penalty.
    peer_punish_good_penalty: float = 0.0
    # Linear / piecewise anneal of peer magnitudes over global episodes
    # (W&B episode_count via schedule_env_multiplier).
    # Legacy (intro=0, anneal_start=0): peak → *_end over [0, anneal_episodes], hold.
    # Piecewise: 0 until intro → peak until anneal_start → linear to *_end by
    # anneal_end_episodes (or anneal_episodes if end=0) → hold.
    peer_punish_anneal: bool = False
    peer_punish_anneal_episodes: int = 30000
    peer_punish_intro_episodes: int = 0
    peer_punish_anneal_start_episodes: int = 0
    peer_punish_anneal_end_episodes: int = 0  # 0 → use anneal_episodes as end
    peer_punish_bad_reward_end: float = 0.0
    peer_punish_good_penalty_end: float = 0.0
    # If True, Bad agents (label < 0) cannot fire zap (action remapped to stay).
    only_good_can_zap: bool = False
    # If True, peer/harvester zaps only freeze Bad (label < 0) targets.
    # Good / empty agents hit by a peer beam stay active (no timeout).
    # Freeze uses post-focus ``zap_hit_matrix``; peer-punish rewards use
    # pre-focus ``zap_beam_matrix`` so Good hits still apply good_penalty.
    peer_focus_on_bad: bool = False
    # If True, Harvest zeros ch15/ch16 in obs; labels still update + drive
    # peer-punish / focus / metrics. Injected into ENV_KWARGS by the trainer.
    mask_reputation_obs: bool = False
    # If False (reputation v3 / harvest-only): zap hits do not change labels;
    # only burst + last-apple do. Peer-punish rewards still use labels.
    zap_affects_label: bool = True
    # Explicit zap→Bad rule switch (only applies when zap_affects_label=True):
    #   False (default / legacy): any Good hit in beam → zapper Bad.
    #   True: zapper Bad only if beam hits ≥1 Good and 0 Bad (mixed stays).
    # Toggle via REPUTATION_V2.zap_to_bad_only_pure_good=true|false.
    zap_to_bad_only_pure_good: bool = False
    # --- Probabilistic Bad enforcement (reuse classic P = clip(-s*rep,0,1)) ---
    # For binary Bad (rep=-1): P = s each timestep while Bad and active.
    prob_enforcement_enabled: bool = False
    # Constant strictness when decay schedule is off.
    prob_enforcement_strictness: float = 0.01
    # Linear decay of s over global episodes (W&B episode_count units).
    prob_enforcement_decay_enabled: bool = False
    prob_enforcement_strictness_start: float = 0.5
    prob_enforcement_strictness_end: float = 0.005
    prob_enforcement_decay_episodes: int = 20000
    # Optional phase 2: after phase 1, linear end → final over this many episodes
    # (0 = hold at ``prob_enforcement_strictness_end``).
    prob_enforcement_decay2_episodes: int = 0
    prob_enforcement_strictness_final: float = 0.0


@struct.dataclass
class BurstWindowState:
    """Burst ring + labels + Bad age + pending delayed freeze."""

    harvest_window: jnp.ndarray  # (num_agents, window_len)
    window_ptr: jnp.ndarray  # (num_agents,) int32
    label: jnp.ndarray  # (num_agents,) float32  +1/+k Good or -1 Bad
    bad_age: jnp.ndarray  # (num_agents,) int32  consecutive Bad steps
    good_age: jnp.ndarray  # (num_agents,) int32  consecutive Good steps
    # Countdown to freeze after becoming Bad; 0 = nothing scheduled.
    pending_timeout: jnp.ndarray  # (num_agents,) int32


def init_burst_window_state(
    num_agents: int, cfg: BurstWindowConfig
) -> BurstWindowState:
    w = int(cfg.window_len)
    return BurstWindowState(
        harvest_window=jnp.zeros((num_agents, w), dtype=jnp.float32),
        window_ptr=jnp.zeros((num_agents,), dtype=jnp.int32),
        label=jnp.full((num_agents,), REP_V2_GOOD, dtype=jnp.float32),
        bad_age=jnp.zeros((num_agents,), dtype=jnp.int32),
        good_age=jnp.zeros((num_agents,), dtype=jnp.int32),
        pending_timeout=jnp.zeros((num_agents,), dtype=jnp.int32),
    )


def update_burst_window(
    state: BurstWindowState,
    harvested_this_step: jnp.ndarray,
    is_frozen: jnp.ndarray,
    cfg: BurstWindowConfig,
) -> Tuple[BurstWindowState, jnp.ndarray]:
    """Update ring buffer; return ``(new_state, burst_overharvest_bool)``.

    ``burst_overharvest`` is for **labeling** (and optional immediate timeout).
    """
    if str(cfg.enforce_mode) != "freeze":
        raise ValueError(
            f"BurstWindowConfig.enforce_mode={cfg.enforce_mode!r}; "
            "only 'freeze' is supported (Harvest_open time_out semantics)."
        )

    harvested = (harvested_this_step > 0.0).astype(jnp.float32)
    frozen = jnp.asarray(is_frozen, dtype=jnp.bool_)
    n_agents, window_len = state.harvest_window.shape
    ptr = state.window_ptr.astype(jnp.int32) % window_len

    agent_ix = jnp.arange(n_agents, dtype=jnp.int32)
    updated_window = state.harvest_window.at[agent_ix, ptr].set(harvested)
    updated_window = jnp.where(
        frozen[:, None], state.harvest_window, updated_window
    )
    ptr_next = jnp.where(frozen, state.window_ptr, (ptr + 1) % window_len)

    window_sum = jnp.sum(updated_window, axis=-1)
    over = window_sum > jnp.float32(cfg.max_harvests_in_window)
    burst_over = over & (~frozen)

    cleared = jnp.where(
        burst_over[:, None],
        jnp.zeros_like(updated_window),
        updated_window,
    )
    ptr_cleared = jnp.where(burst_over, jnp.zeros_like(ptr_next), ptr_next)

    new_state = BurstWindowState(
        harvest_window=cleared,
        window_ptr=ptr_cleared.astype(jnp.int32),
        label=state.label,
        bad_age=state.bad_age,
        good_age=state.good_age,
        pending_timeout=state.pending_timeout,
    )
    return new_state, burst_over


def zap_marks_zapper_bad(
    hit_good: jnp.ndarray,
    hit_bad: jnp.ndarray,
    *,
    zap_to_bad_only_pure_good: bool = False,
) -> jnp.ndarray:
    """Explicit toggle: does a zap that hit Good mark the zapper Bad?

    - ``zap_to_bad_only_pure_good=False`` (legacy): any Good hit → Bad.
    - ``zap_to_bad_only_pure_good=True``: Bad only when ≥1 Good and 0 Bad
      in the beam (mixed Good+Bad does not mark via zap).
    """
    hit_good = jnp.asarray(hit_good, dtype=jnp.bool_)
    hit_bad = jnp.asarray(hit_bad, dtype=jnp.bool_)
    if bool(zap_to_bad_only_pure_good):
        return hit_good & (~hit_bad)
    return hit_good


def update_binary_labels(
    pre_label: jnp.ndarray,
    *,
    burst_trigger: jnp.ndarray,
    last_apple_trigger: jnp.ndarray,
    zap_hit_matrix: jnp.ndarray,
    is_frozen: jnp.ndarray,
    full_reset: jnp.ndarray,
    just_respawned: jnp.ndarray | None = None,
    reset_label_on_respawn: bool = True,
    zap_affects_label: bool = True,
    zap_to_bad_only_pure_good: bool = False,
) -> jnp.ndarray:
    """Binary Good(+1) / Bad(−1) update for reputation v2 / harvest-only v3."""
    pre = jnp.asarray(pre_label, dtype=jnp.float32)
    frozen = jnp.asarray(is_frozen, dtype=jnp.bool_)
    burst = jnp.asarray(burst_trigger, dtype=jnp.bool_)
    last_apple = jnp.asarray(last_apple_trigger, dtype=jnp.bool_)
    hits = jnp.asarray(zap_hit_matrix, dtype=jnp.bool_)

    target_good = pre > 0.0
    target_bad = pre < 0.0
    hit_good = jnp.any(hits & target_good[None, :], axis=1)
    hit_bad = jnp.any(hits & target_bad[None, :], axis=1)
    use_zap = bool(zap_affects_label)
    if use_zap:
        zap_to_bad = zap_marks_zapper_bad(
            hit_good,
            hit_bad,
            zap_to_bad_only_pure_good=zap_to_bad_only_pure_good,
        )
        zap_to_good = hit_bad & (~hit_good)
    else:
        zap_to_bad = jnp.zeros_like(hit_good)
        zap_to_good = jnp.zeros_like(hit_bad)

    force_bad = burst | last_apple | zap_to_bad
    force_good = zap_to_good & (~force_bad)

    updated = jnp.where(
        force_bad,
        jnp.float32(REP_V2_BAD),
        jnp.where(
            force_good,
            jnp.float32(REP_V2_GOOD),
            pre,
        ),
    )
    updated = jnp.where(frozen, pre, updated)

    if just_respawned is not None and reset_label_on_respawn:
        respawn = jnp.asarray(just_respawned, dtype=jnp.bool_)
        updated = jnp.where(respawn, jnp.float32(REP_V2_GOOD), updated)

    reset = jnp.asarray(full_reset, dtype=jnp.bool_)
    updated = jnp.where(reset, jnp.float32(REP_V2_GOOD), updated)
    return updated


def update_graded_labels(
    pre_label: jnp.ndarray,
    *,
    burst_trigger: jnp.ndarray,
    last_apple_trigger: jnp.ndarray,
    zap_hit_matrix: jnp.ndarray,
    is_frozen: jnp.ndarray,
    full_reset: jnp.ndarray,
    just_respawned: jnp.ndarray | None = None,
    reset_label_on_respawn: bool = True,
    max_reputation: int = 0,
    zap_affects_label: bool = True,
    zap_to_bad_only_pure_good: bool = False,
) -> jnp.ndarray:
    """Graded Good(+1,+2,…) / Bad(−1) update for reputation v2.

    While Good: ``+1`` per Bad agent hit; ``−1`` for a Bad act (burst, last-apple,
    or a Good-hit zap under the same pure-Good rule as binary). Crossing
    ``<= 0`` maps to Bad (−1). While Bad: stay −1 unless rehabilitating by
    hitting only Bad (→ +1). Agent labels never 0.

    With ``zap_affects_label=false``: zap hits are ignored (harvest-only v3).
    """
    pre = jnp.asarray(pre_label, dtype=jnp.float32)
    frozen = jnp.asarray(is_frozen, dtype=jnp.bool_)
    burst = jnp.asarray(burst_trigger, dtype=jnp.bool_)
    last_apple = jnp.asarray(last_apple_trigger, dtype=jnp.bool_)
    hits = jnp.asarray(zap_hit_matrix, dtype=jnp.bool_)

    target_good = pre > 0.0
    target_bad = pre < 0.0
    n_bad_hits = jnp.sum(hits & target_bad[None, :], axis=1).astype(jnp.float32)
    n_good_hits = jnp.sum(hits & target_good[None, :], axis=1).astype(jnp.float32)
    if not bool(zap_affects_label):
        n_bad_hits = jnp.zeros_like(n_bad_hits)
        n_good_hits = jnp.zeros_like(n_good_hits)
    hit_good = n_good_hits > 0.0
    hit_bad = n_bad_hits > 0.0
    zap_bad_act = zap_marks_zapper_bad(
        hit_good,
        hit_bad,
        zap_to_bad_only_pure_good=zap_to_bad_only_pure_good,
    )
    did_bad_act = burst | last_apple | zap_bad_act

    is_good = pre > 0.0
    is_bad = pre < 0.0

    # Good stack: reward Bad hits, penalize Bad acts once per step.
    good_delta = n_bad_hits - did_bad_act.astype(jnp.float32)
    after_good = pre + good_delta
    after_good = jnp.where(after_good <= 0.0, jnp.float32(REP_V2_BAD), after_good)
    max_rep = int(max_reputation)
    if max_rep > 0:
        after_good = jnp.minimum(after_good, jnp.float32(max_rep))
        # Cap only applies to Good levels; Bad stays −1.
        after_good = jnp.where(
            after_good < 0.0, jnp.float32(REP_V2_BAD), after_good
        )

    # Bad: rehab to +1 only when hitting Bad without any Bad act.
    rehab = is_bad & hit_bad & (~hit_good) & (~burst) & (~last_apple)
    after_bad = jnp.where(rehab, jnp.float32(REP_V2_GOOD), jnp.float32(REP_V2_BAD))

    updated = jnp.where(is_good, after_good, jnp.where(is_bad, after_bad, pre))
    updated = jnp.where(frozen, pre, updated)

    if just_respawned is not None and reset_label_on_respawn:
        respawn = jnp.asarray(just_respawned, dtype=jnp.bool_)
        updated = jnp.where(respawn, jnp.float32(REP_V2_GOOD), updated)

    reset = jnp.asarray(full_reset, dtype=jnp.bool_)
    updated = jnp.where(reset, jnp.float32(REP_V2_GOOD), updated)
    return updated


def update_reputation_labels(
    pre_label: jnp.ndarray,
    *,
    burst_trigger: jnp.ndarray,
    last_apple_trigger: jnp.ndarray,
    zap_hit_matrix: jnp.ndarray,
    is_frozen: jnp.ndarray,
    full_reset: jnp.ndarray,
    just_respawned: jnp.ndarray | None = None,
    reset_label_on_respawn: bool = True,
    graded_reputation: bool = False,
    graded_max_reputation: int = 0,
    zap_affects_label: bool = True,
    zap_to_bad_only_pure_good: bool = False,
) -> jnp.ndarray:
    """Dispatch binary vs graded label update.

    Bandits never appear here: labels and ``zap_hit_matrix`` are length /
    shape ``N`` (real agents only). Env-side bandits use hardcoded
    ``BANDIT_REP`` in obs / reward paths only.
    """
    if bool(graded_reputation):
        return update_graded_labels(
            pre_label,
            burst_trigger=burst_trigger,
            last_apple_trigger=last_apple_trigger,
            zap_hit_matrix=zap_hit_matrix,
            is_frozen=is_frozen,
            full_reset=full_reset,
            just_respawned=just_respawned,
            reset_label_on_respawn=reset_label_on_respawn,
            max_reputation=graded_max_reputation,
            zap_affects_label=zap_affects_label,
            zap_to_bad_only_pure_good=zap_to_bad_only_pure_good,
        )
    return update_binary_labels(
        pre_label,
        burst_trigger=burst_trigger,
        last_apple_trigger=last_apple_trigger,
        zap_hit_matrix=zap_hit_matrix,
        is_frozen=is_frozen,
        full_reset=full_reset,
        just_respawned=just_respawned,
        reset_label_on_respawn=reset_label_on_respawn,
        zap_affects_label=zap_affects_label,
        zap_to_bad_only_pure_good=zap_to_bad_only_pure_good,
    )


def update_bad_age(
    bad_age: jnp.ndarray,
    label: jnp.ndarray,
    *,
    is_frozen: jnp.ndarray,
    full_reset: jnp.ndarray,
) -> jnp.ndarray:
    """Consecutive active steps spent Bad; 0 while Good / on episode reset."""
    bad = label < 0.0
    frozen = jnp.asarray(is_frozen, dtype=jnp.bool_)
    # Count only while active and Bad; freeze pauses the grace clock.
    age_next = jnp.where(
        bad & (~frozen),
        bad_age.astype(jnp.int32) + 1,
        jnp.where(bad & frozen, bad_age.astype(jnp.int32), jnp.int32(0)),
    )
    reset = jnp.asarray(full_reset, dtype=jnp.bool_)
    return jnp.where(reset, jnp.int32(0), age_next)


def update_good_age(
    good_age: jnp.ndarray,
    label: jnp.ndarray,
    *,
    is_frozen: jnp.ndarray,
    full_reset: jnp.ndarray,
) -> jnp.ndarray:
    """Consecutive active steps spent Good; 0 while Bad/empty / on episode reset."""
    good = label > 0.0
    frozen = jnp.asarray(is_frozen, dtype=jnp.bool_)
    age_next = jnp.where(
        good & (~frozen),
        good_age.astype(jnp.int32) + 1,
        jnp.where(good & frozen, good_age.astype(jnp.int32), jnp.int32(0)),
    )
    reset = jnp.asarray(full_reset, dtype=jnp.bool_)
    return jnp.where(reset, jnp.int32(0), age_next)


def resolve_enforcement_schedule(
    episodes_completed: jnp.ndarray,
    cfg: BurstWindowConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Map per-env completed episodes → (immediate, timeout_delay, enf_enabled).

    Global episode index ≈ ``episodes_completed * schedule_env_multiplier``
    (matches W&B ``episode_count`` when multiplier = NUM_ENVS and envs stay
    synchronized).

    Delay ramp (when ``delay_ramp_enabled`` and ``delay_hold_episodes == 0``):
      global ∈ [0, every)     → delay 0 (+ immediate if start_immediate)
      global ∈ [every, 2e)    → delay 1
      …
      capped at ``delay_ramp_max_delay``.

    Hold-then-ramp curriculum (``delay_hold_episodes > 0``):
      global ∈ [0, hold)                    → delay = hold_start_delay
      global ∈ [hold, delay_ramp_end)       → delay rises from hold_start
      global ≥ delay_ramp_end (if > 0)      →
        ``delay_curriculum_end_disable_enforcement``: enf off (peer zaps only)
        else ``delay_curriculum_end_immediate``: immediate freeze
        else stay at capped ramp delay

    ``enforcement_stop_after_episodes > 0``: after that many global episodes,
    ``enf_enabled`` is False (no immediate / delayed / periodic freezes).
    """
    completed = jnp.asarray(episodes_completed, dtype=jnp.int32)
    mult = jnp.int32(max(int(cfg.schedule_env_multiplier), 1))
    global_eps = completed * mult

    stop_after = int(cfg.enforcement_stop_after_episodes)
    stopped = jnp.bool_(stop_after > 0) & (global_eps >= jnp.int32(stop_after))

    base_enabled = jnp.bool_(bool(cfg.enforcement_enabled)) & (~stopped)
    base_immediate = jnp.bool_(bool(cfg.immediate_timeout))
    base_delay = jnp.int32(max(int(cfg.timeout_delay), 0))

    if not bool(cfg.delay_ramp_enabled):
        return base_immediate & base_enabled, base_delay, base_enabled

    every = max(int(cfg.delay_ramp_every_episodes), 1)
    max_delay = max(int(cfg.delay_ramp_max_delay), 0)
    hold_eps = max(int(cfg.delay_hold_episodes), 0)
    hold_start = max(int(cfg.delay_hold_start_delay), 0)
    end_eps = max(int(cfg.delay_ramp_end_episodes), 0)
    end_immediate = bool(cfg.delay_curriculum_end_immediate)
    end_disable = bool(cfg.delay_curriculum_end_disable_enforcement)

    if hold_eps > 0:
        # Hold → ramp-up → optional immediate or disable-enforcement end.
        max_delay = max(max_delay, hold_start)
        in_hold = global_eps < jnp.int32(hold_eps)
        in_end = jnp.bool_(end_eps > 0) & (global_eps >= jnp.int32(end_eps))
        ramp_extra = (global_eps - jnp.int32(hold_eps)) // jnp.int32(every)
        ramp_delay = jnp.minimum(
            jnp.int32(hold_start) + jnp.maximum(ramp_extra, jnp.int32(0)),
            jnp.int32(max_delay),
        )
        delay = jnp.where(in_hold, jnp.int32(hold_start), ramp_delay)
        immediate = jnp.bool_(False)
        enabled = base_enabled
        if end_disable and end_eps > 0:
            enabled = base_enabled & (~in_end)
            delay = jnp.where(in_end, jnp.int32(0), delay)
        elif end_immediate and end_eps > 0:
            delay = jnp.where(in_end, jnp.int32(0), delay)
            immediate = in_end
        return immediate & enabled, delay, enabled

    # Legacy ramp from phase 0.
    phase = jnp.minimum(global_eps // jnp.int32(every), jnp.int32(max_delay))
    start_imm = jnp.bool_(bool(cfg.delay_ramp_start_immediate))
    immediate = base_enabled & (phase == 0) & start_imm
    delay = jnp.where(phase == 0, jnp.int32(0), phase.astype(jnp.int32))
    return immediate, delay, base_enabled


def resolve_prob_enforcement_strictness(
    episodes_completed: jnp.ndarray,
    cfg: BurstWindowConfig,
) -> jnp.ndarray:
    """Strictness ``s`` for probabilistic Bad enforcement (scalar float32).

    When ``prob_enforcement_decay_enabled``:
      phase 1: linear ``start → end`` over ``decay_episodes`` global episodes;
      phase 2 (optional): if ``decay2_episodes > 0``, linear ``end → final``
        over the next ``decay2_episodes``, then hold at ``final``;
      else hold at ``end`` after phase 1.
    Otherwise use constant ``prob_enforcement_strictness``.

    Independent of delayed/periodic ``enforcement_enabled``, but ``s → 0`` after
    ``enforcement_stop_after_episodes`` or a delay-curriculum end-disable phase.
    """
    if not bool(cfg.prob_enforcement_enabled):
        return jnp.float32(0.0)
    completed = jnp.asarray(episodes_completed, dtype=jnp.int32)
    mult = jnp.int32(max(int(cfg.schedule_env_multiplier), 1))
    global_eps = completed * mult
    if not bool(cfg.prob_enforcement_decay_enabled):
        s = jnp.float32(
            min(1.0, max(0.0, float(cfg.prob_enforcement_strictness)))
        )
    else:
        horizon1 = max(int(cfg.prob_enforcement_decay_episodes), 1)
        s0 = jnp.float32(float(cfg.prob_enforcement_strictness_start))
        s1 = jnp.float32(float(cfg.prob_enforcement_strictness_end))
        t1 = jnp.clip(
            global_eps.astype(jnp.float32) / jnp.float32(horizon1), 0.0, 1.0
        )
        s_phase1 = jnp.clip(s0 + (s1 - s0) * t1, jnp.float32(0.0), jnp.float32(1.0))
        horizon2 = max(int(cfg.prob_enforcement_decay2_episodes), 0)
        if horizon2 > 0:
            s2 = jnp.float32(
                min(1.0, max(0.0, float(cfg.prob_enforcement_strictness_final)))
            )
            after1 = jnp.maximum(
                global_eps - jnp.int32(horizon1), jnp.int32(0)
            ).astype(jnp.float32)
            t2 = jnp.clip(after1 / jnp.float32(horizon2), 0.0, 1.0)
            s_phase2 = jnp.clip(
                s1 + (s2 - s1) * t2, jnp.float32(0.0), jnp.float32(1.0)
            )
            s = jnp.where(global_eps < jnp.int32(horizon1), s_phase1, s_phase2)
        else:
            s = s_phase1

    stop_after = int(cfg.enforcement_stop_after_episodes)
    if stop_after > 0:
        s = jnp.where(
            global_eps >= jnp.int32(stop_after), jnp.float32(0.0), s
        )
    end_eps = max(int(cfg.delay_ramp_end_episodes), 0)
    if (
        bool(cfg.delay_ramp_enabled)
        and bool(cfg.delay_curriculum_end_disable_enforcement)
        and end_eps > 0
    ):
        s = jnp.where(global_eps >= jnp.int32(end_eps), jnp.float32(0.0), s)
    return s


def resolve_peer_punish_magnitudes(
    episodes_completed: jnp.ndarray,
    cfg: BurstWindowConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Peer Bad bonus / Good penalty (scalar float32) for this step.

    Anneal off: return static ``peer_punish_bad_reward`` /
    ``peer_punish_good_penalty``.

    Anneal on (legacy, intro=0 and anneal_start=0): linear peak→*_end over
    ``[0, peer_punish_anneal_episodes]``, then hold.

    Anneal on (piecewise): magnitudes are 0 until ``intro``, peak until
    ``anneal_start``, linear peak→*_end until ``anneal_end`` (from
    ``peer_punish_anneal_end_episodes`` or ``peer_punish_anneal_episodes``),
    then hold. Episode index is ``completed * schedule_env_multiplier``.
    """
    bad0 = jnp.float32(float(cfg.peer_punish_bad_reward))
    good0 = jnp.float32(float(cfg.peer_punish_good_penalty))
    if not bool(cfg.peer_punish_anneal):
        return bad0, good0

    completed = jnp.asarray(episodes_completed, dtype=jnp.int32)
    mult = jnp.int32(max(int(cfg.schedule_env_multiplier), 1))
    global_eps = completed * mult
    bad1 = jnp.float32(float(cfg.peer_punish_bad_reward_end))
    good1 = jnp.float32(float(cfg.peer_punish_good_penalty_end))

    intro = max(int(cfg.peer_punish_intro_episodes), 0)
    anneal_start = max(int(cfg.peer_punish_anneal_start_episodes), 0)
    anneal_end = int(cfg.peer_punish_anneal_end_episodes)
    if anneal_end <= 0:
        anneal_end = max(int(cfg.peer_punish_anneal_episodes), 1)

    if intro > 0 or anneal_start > 0:
        # Piecewise: 0 → [intro] peak → [anneal_start] linear → [anneal_end] hold.
        anneal_start = max(anneal_start, intro)
        anneal_end = max(anneal_end, anneal_start + 1)
        ep = global_eps.astype(jnp.float32)
        intro_f = jnp.float32(intro)
        start_f = jnp.float32(anneal_start)
        end_f = jnp.float32(anneal_end)
        span = jnp.maximum(end_f - start_f, jnp.float32(1.0))
        t = jnp.clip((ep - start_f) / span, 0.0, 1.0)
        bad_decay = bad0 + (bad1 - bad0) * t
        good_decay = good0 + (good1 - good0) * t
        # Before intro: end values (typically 0). Intro..start: peak. Else decay/hold.
        before_intro = ep < intro_f
        before_decay = ep < start_f
        bad = jnp.where(
            before_intro,
            bad1,
            jnp.where(before_decay, bad0, bad_decay),
        )
        good = jnp.where(
            before_intro,
            good1,
            jnp.where(before_decay, good0, good_decay),
        )
        return bad, good

    horizon = max(int(cfg.peer_punish_anneal_episodes), 1)
    t = jnp.clip(
        global_eps.astype(jnp.float32) / jnp.float32(horizon), 0.0, 1.0
    )
    bad = bad0 + (bad1 - bad0) * t
    good = good0 + (good1 - good0) * t
    return bad, good


def update_pending_timeout(
    pending: jnp.ndarray,
    *,
    pre_label: jnp.ndarray,
    label_new: jnp.ndarray,
    is_frozen: jnp.ndarray,
    full_reset: jnp.ndarray,
    timeout_delay: int | jnp.ndarray,
    enforcement_enabled: bool | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-agent countdown: freeze exactly ``timeout_delay`` steps after Bad.

    On Good→Bad (while active): schedule ``pending = timeout_delay``.
    Each later step: decrement; when the counter hits 0, fire freeze.
    Cleared if the agent becomes Good, freezes for another reason, or resets.

    ``timeout_delay <= 0`` or enforcement off → never schedules / never fires.
    ``timeout_delay`` / ``enforcement_enabled`` may be traced JAX scalars.
    """
    delay = jnp.maximum(jnp.asarray(timeout_delay, dtype=jnp.int32), jnp.int32(0))
    enabled = jnp.asarray(enforcement_enabled, dtype=jnp.bool_) & (delay > 0)
    pending_i = jnp.asarray(pending, dtype=jnp.int32)
    frozen = jnp.asarray(is_frozen, dtype=jnp.bool_)
    pre = jnp.asarray(pre_label, dtype=jnp.float32)
    lab = jnp.asarray(label_new, dtype=jnp.float32)
    newly_bad = (lab < 0.0) & (pre >= 0.0) & (~frozen)
    became_good = lab >= 0.0

    had_pending = pending_i > 0

    # Fresh schedule on newly Bad (keep an already-running countdown).
    after_schedule = jnp.where(
        enabled & newly_bad & (~had_pending),
        delay,
        pending_i,
    )
    # Tick down existing countdowns (not the same step they were scheduled).
    after_tick = jnp.where(
        had_pending & (~frozen) & (~became_good),
        jnp.maximum(pending_i - 1, 0),
        after_schedule,
    )
    # Cancel if Good or already frozen.
    pending_next = jnp.where(became_good | frozen, jnp.int32(0), after_tick)

    # Fire when a previously-pending countdown reaches 0 this step.
    fire = (
        enabled
        & had_pending
        & (pending_i > 0)
        & (pending_next == 0)
        & (~frozen)
        & (~became_good)
    )

    reset = jnp.asarray(full_reset, dtype=jnp.bool_)
    pending_next = jnp.where(reset, jnp.int32(0), pending_next)
    fire = fire & (~reset)
    return pending_next, fire


def periodic_bad_enforcement_trigger(
    label: jnp.ndarray,
    bad_age: jnp.ndarray,
    *,
    env_step: jnp.ndarray,
    is_frozen: jnp.ndarray,
    enforcement_enabled: bool,
    enforcement_delay: int,
    enforcement_period: int,
) -> jnp.ndarray:
    """True for Bad agents eligible on a periodic enforcement tick.

    JIT-safe: period<=0 or disabled → never triggers (no Python early-return).
    """
    period = max(int(enforcement_period), 0)
    delay = max(int(enforcement_delay), 0)
    enabled = jnp.bool_(bool(enforcement_enabled) and period > 0)
    # Avoid mod-by-zero when period is 0: use period_safe=1 but gate with enabled.
    period_safe = jnp.int32(period if period > 0 else 1)
    step = jnp.asarray(env_step, dtype=jnp.int32)
    tick = enabled & (step > 0) & ((step % period_safe) == 0)
    eligible = (
        (label < 0.0)
        & (bad_age >= jnp.int32(delay))
        & (~jnp.asarray(is_frozen, dtype=jnp.bool_))
    )
    return tick & eligible


def peer_punish_zap_reward(
    zap_hit_matrix: jnp.ndarray,
    target_label: jnp.ndarray,
    *,
    enabled: bool,
    bad_reward: float = 1.0,
    good_penalty: float = 0.0,
    exclude_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Harvester peer-punish signal from zap targets' reputation (no police).

    Per target in the beam (pre-step label):

    - Bad (label < 0): ``+bad_reward`` each
    - Good (label > 0): ``good_penalty`` each (0 = off; typically ``-bad_reward``)
    - Empty / neutral (label == 0): 0

    Mixed beams sum both terms. ``exclude_mask`` agents (e.g. police) receive 0.

    ``zap_hit_matrix`` is ``(N, T)`` (T may be N or N+B when bandit columns are
    appended). Returned reward is always length ``N`` (one per shooter).
    """
    hits = jnp.asarray(zap_hit_matrix, dtype=jnp.float32)
    n_agents = int(hits.shape[0])
    zeros = jnp.zeros((n_agents,), dtype=jnp.float32)
    if not bool(enabled):
        return zeros

    lab = jnp.asarray(target_label, dtype=jnp.float32)
    bad_t = lab < 0.0
    good_t = lab > 0.0
    n_bad = jnp.sum(hits * bad_t[None, :], axis=1)
    n_good = jnp.sum(hits * good_t[None, :], axis=1)

    br = jnp.float32(bad_reward)
    gp = jnp.float32(good_penalty)
    rew = n_bad * br + n_good * gp

    if exclude_mask is not None:
        rew = jnp.where(
            jnp.asarray(exclude_mask, dtype=jnp.bool_), zeros, rew
        )
    return rew


def mask_bad_agent_zap_actions(
    actions: jnp.ndarray,
    label: jnp.ndarray,
    *,
    enabled: bool,
    zap_action: int = 7,
    noop_action: int = 6,
    allow_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Replace Bad agents' zap actions with stay when ``enabled``.

    ``zap_action`` / ``noop_action`` default to Harvest ``Actions.zap_forward`` /
    ``Actions.stay``. Agents with ``allow_mask`` True (e.g. police) are unchanged.
    """
    acts = jnp.asarray(actions)
    if not bool(enabled):
        return acts
    lab = jnp.asarray(label, dtype=jnp.float32)
    bad = lab < 0.0
    if allow_mask is not None:
        bad = bad & ~jnp.asarray(allow_mask, dtype=jnp.bool_)
    is_zap = acts == jnp.asarray(zap_action, dtype=acts.dtype)
    return jnp.where(bad & is_zap, jnp.asarray(noop_action, dtype=acts.dtype), acts)


def apply_existing_timeout_trigger(
    time_out: jnp.ndarray,
    timeout_from_enforcement: jnp.ndarray,
    trigger: jnp.ndarray,
    timeout_duration: int = TAG_TIMEOUT_STEPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Feed a v2 trigger into the EXISTING freeze fields on ``Harvest_open.State``."""
    trigger = jnp.asarray(trigger, dtype=jnp.bool_)
    dur = jnp.int16(int(timeout_duration))
    time_out_i16 = time_out.astype(jnp.int16)
    time_out_next = jnp.where(
        trigger, jnp.maximum(dur, time_out_i16), time_out_i16
    ).astype(jnp.int16)
    enf_next = jnp.where(
        trigger,
        True,
        timeout_from_enforcement.astype(jnp.bool_),
    )
    return time_out_next, enf_next
