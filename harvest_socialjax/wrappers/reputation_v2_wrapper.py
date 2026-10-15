"""Stateful reputation v2 wrapper: Good/Bad labels + delayed/periodic enforcement.

Stack: ``Harvest → ReputationV2Wrapper → LogWrapper → …``.

Binary mode: labels ∈ {+1, −1}. Graded mode (``cfg.graded_reputation``):
Bad −1, Good +1,+2,+3,…
"""
from __future__ import annotations

from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct

from harvest_socialjax.envs.bandits import extended_zap_labels_and_hits
from harvest_socialjax.envs.harvest_open import Actions
from harvest_socialjax.envs.reputation import sample_reputation_enforcement
from harvest_socialjax.envs.reputation_v2 import (
    BurstWindowConfig,
    BurstWindowState,
    apply_existing_timeout_trigger,
    init_burst_window_state,
    mask_bad_agent_zap_actions,
    peer_punish_zap_reward,
    periodic_bad_enforcement_trigger,
    resolve_enforcement_schedule,
    resolve_peer_punish_magnitudes,
    resolve_prob_enforcement_strictness,
    update_bad_age,
    update_burst_window,
    update_good_age,
    update_pending_timeout,
    update_reputation_labels,
)
from harvest_socialjax.envs.police.mechanics import mask_zap_hits_on_police
from harvest_socialjax.socialjax_path import ensure_socialjax_on_path

ensure_socialjax_on_path()
from socialjax.wrappers.baselines import JaxMARLWrapper  # noqa: E402


@struct.dataclass
class ReputationV2EnvState:
    env_state: Any
    burst_state: BurstWindowState
    env_step: jnp.ndarray  # scalar int32 — clocks periodic enforcement
    # Inner episodes completed in this parallel env (for delay/stop schedules).
    episodes_completed: jnp.ndarray  # scalar int32


def wandb_ep_to_per_env_completed(
    wandb_episode_count: int, schedule_env_multiplier: int
) -> int:
    """Map W&B ``episode_count`` → per-env ``episodes_completed``.

    Schedules use ``global_eps = completed * schedule_env_multiplier`` with
    ``schedule_env_multiplier`` typically ``NUM_ENVS``.
    """
    mult = max(int(schedule_env_multiplier), 1)
    return max(int(wandb_episode_count), 0) // mult


def seed_schedule_episode_clock(env_state: Any, per_env_completed: int) -> Any:
    """Seed Harvest + ReputationV2 schedule clocks after a resume ``batch_reset``.

    ``env_state`` may be a single env or a batched ``LogEnvState`` tree
    (leading env axis). Sets both wrapper and Harvest ``episodes_completed``.
    """
    eps = jnp.asarray(per_env_completed, dtype=jnp.int32)

    def _seed_one(state: Any) -> Any:
        # LogEnvState → ReputationV2EnvState → Harvest
        if hasattr(state, "episode_returns") and hasattr(state, "env_state"):
            return state.replace(env_state=_seed_one(state.env_state))
        if hasattr(state, "burst_state") and hasattr(state, "episodes_completed"):
            harvest = state.env_state
            if hasattr(harvest, "episodes_completed"):
                harvest = harvest.replace(
                    episodes_completed=jnp.full_like(
                        harvest.episodes_completed, eps
                    )
                )
            return state.replace(
                env_state=harvest,
                episodes_completed=jnp.full_like(state.episodes_completed, eps),
            )
        if hasattr(state, "episodes_completed"):
            return state.replace(
                episodes_completed=jnp.full_like(state.episodes_completed, eps)
            )
        return state

    return _seed_one(env_state)


class ReputationV2Wrapper(JaxMARLWrapper):
    """Reputation on ch15/ch16 + optional delayed/periodic Bad timeouts."""

    def __init__(
        self,
        env,
        cfg: BurstWindowConfig | None = None,
    ):
        super().__init__(env)
        self.cfg = cfg if cfg is not None else BurstWindowConfig()
        self.num_agents = int(env.num_agents)
        # Optional police (no rep dynamics; never system-enforced).
        pm = getattr(env, "police_mask", None)
        if pm is None:
            pm = getattr(env, "_police_mask", None)
        if pm is None:
            self.police_mask = jnp.zeros((self.num_agents,), dtype=jnp.bool_)
        else:
            self.police_mask = jnp.asarray(pm, dtype=jnp.bool_)

    def _refresh_obs(self, env_state):
        return self._env.get_obs(env_state)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        _, env_state = self._env.reset(key)
        burst_state = init_burst_window_state(self.num_agents, self.cfg)
        # Police stay empty (0); harvesters start Good.
        label0 = jnp.where(
            self.police_mask, jnp.float32(0.0), burst_state.label
        )
        burst_state = BurstWindowState(
            harvest_window=burst_state.harvest_window,
            window_ptr=burst_state.window_ptr,
            label=label0,
            bad_age=burst_state.bad_age,
            good_age=burst_state.good_age,
            pending_timeout=burst_state.pending_timeout,
        )
        env_state = env_state.replace(tag_filter_rep=burst_state.label)
        obs = self._refresh_obs(env_state)
        state = ReputationV2EnvState(
            env_state=env_state,
            burst_state=burst_state,
            env_step=jnp.int32(0),
            episodes_completed=jnp.int32(0),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: ReputationV2EnvState, actions):
        pre_label = state.burst_state.label
        # Keep Harvest_open's bandit schedule clock aligned with the wrapper
        # counter (W&B episode_count / peer anneal). Auto-reset used to zero
        # the env field every outer done; we still seed it explicitly here.
        env_state_in = state.env_state.replace(
            tag_filter_rep=pre_label,
            episodes_completed=jnp.asarray(
                state.episodes_completed, dtype=jnp.int32
            ),
        )
        # Bad agents cannot fire zap when only_good_can_zap (police exempt).
        actions_in = mask_bad_agent_zap_actions(
            actions,
            pre_label,
            enabled=bool(self.cfg.only_good_can_zap),
            zap_action=int(Actions.zap_forward),
            noop_action=int(Actions.stay),
            allow_mask=self.police_mask,
        )
        key, step_key, enf_key = jax.random.split(key, 3)
        obs, env_state_inner, reward, done, info = self._env.step(
            step_key, env_state_in, actions_in
        )

        harvested_this_step = info["environmental_rewards"]
        is_frozen = info["tagged"]
        zap_hit_matrix = info["zap_hit_matrix"]
        # Pre-focus beam (Good still present). Used for peer-punish rewards +
        # Good/Bad hit counts when peer_focus_on_bad clears Good from freeze.
        zap_beam_matrix = info.get("zap_beam_matrix", zap_hit_matrix)
        # Police never participate in reputation label dynamics.
        # Labels / freeze follow post-focus hits; rewards follow the beam.
        zap_for_labels = mask_zap_hits_on_police(zap_hit_matrix, self.police_mask)
        zap_for_labels = zap_for_labels & ~self.police_mask[:, None]
        zap_for_rewards = mask_zap_hits_on_police(
            jnp.asarray(zap_beam_matrix, dtype=jnp.bool_), self.police_mask
        )
        zap_for_rewards = zap_for_rewards & ~self.police_mask[:, None]
        # Bandits are env-side only: never enter label updates (NxN matrix only).
        just_respawned = info.get(
            "just_respawned",
            jnp.zeros((self.num_agents,), dtype=jnp.bool_),
        )
        full_reset = done["__all__"]

        burst_state_new, burst_over = update_burst_window(
            state.burst_state,
            harvested_this_step,
            is_frozen,
            self.cfg,
        )
        # Police cannot become Bad via burst (they never harvest; belt-and-suspenders).
        burst_over = burst_over & ~self.police_mask

        locally_last = info.get(
            "locally_last_apple",
            jnp.zeros((self.num_agents,), dtype=jnp.bool_),
        )
        last_apple_bad = (
            jnp.asarray(locally_last, dtype=jnp.bool_)
            & jnp.bool_(self.cfg.last_apple_timeout)
            & (~jnp.asarray(is_frozen, dtype=jnp.bool_))
            & (~self.police_mask)
        )

        # Clear window on local-last Bad (burst already clears on overharvest).
        clear_mask = last_apple_bad & (~burst_over)
        burst_state_new = BurstWindowState(
            harvest_window=jnp.where(
                clear_mask[:, None],
                jnp.zeros_like(burst_state_new.harvest_window),
                burst_state_new.harvest_window,
            ),
            window_ptr=jnp.where(
                clear_mask,
                jnp.zeros_like(burst_state_new.window_ptr),
                burst_state_new.window_ptr,
            ),
            label=burst_state_new.label,
            bad_age=burst_state_new.bad_age,
            good_age=burst_state_new.good_age,
            pending_timeout=burst_state_new.pending_timeout,
        )

        label_new = update_reputation_labels(
            pre_label,
            burst_trigger=burst_over,
            last_apple_trigger=last_apple_bad,
            zap_hit_matrix=zap_for_labels,
            is_frozen=is_frozen,
            full_reset=full_reset,
            just_respawned=just_respawned,
            reset_label_on_respawn=bool(self.cfg.reset_label_on_respawn),
            graded_reputation=bool(self.cfg.graded_reputation),
            graded_max_reputation=int(self.cfg.graded_max_reputation),
            zap_affects_label=bool(self.cfg.zap_affects_label),
            zap_to_bad_only_pure_good=bool(self.cfg.zap_to_bad_only_pure_good),
        )
        # Police: fixed empty label (0) for the whole episode.
        label_new = jnp.where(self.police_mask, jnp.float32(0.0), label_new)

        bad_age_new = update_bad_age(
            burst_state_new.bad_age,
            label_new,
            is_frozen=is_frozen,
            full_reset=full_reset,
        )
        good_age_new = update_good_age(
            burst_state_new.good_age,
            label_new,
            is_frozen=is_frozen,
            full_reset=full_reset,
        )

        # Episode-indexed delay ramp / stop-after (W&B episode_count units).
        immediate_on, timeout_delay_t, enf_enabled_t = resolve_enforcement_schedule(
            state.episodes_completed, self.cfg
        )

        pending_next, delayed_fire = update_pending_timeout(
            burst_state_new.pending_timeout,
            pre_label=pre_label,
            label_new=label_new,
            is_frozen=is_frozen,
            full_reset=full_reset,
            timeout_delay=timeout_delay_t,
            enforcement_enabled=enf_enabled_t,
        )
        # If enforcement is currently off, drop any leftover countdowns.
        pending_next = jnp.where(enf_enabled_t, pending_next, jnp.int32(0))
        delayed_fire = delayed_fire & enf_enabled_t

        env_step_next = jnp.where(
            full_reset,
            jnp.int32(0),
            jnp.asarray(state.env_step, dtype=jnp.int32) + 1,
        )

        window_reset = jnp.where(
            full_reset,
            jnp.zeros_like(burst_state_new.harvest_window),
            burst_state_new.harvest_window,
        )
        ptr_reset = jnp.where(
            full_reset,
            jnp.zeros_like(burst_state_new.window_ptr),
            burst_state_new.window_ptr,
        )
        burst_state_new = BurstWindowState(
            harvest_window=window_reset,
            window_ptr=ptr_reset,
            label=label_new,
            bad_age=bad_age_new,
            good_age=good_age_new,
            pending_timeout=pending_next,
        )

        # --- Timeouts (optional) --------------------------------------------
        immediate = (
            immediate_on
            & (burst_over | last_apple_bad)
            & (~jnp.asarray(is_frozen, dtype=jnp.bool_))
        )
        periodic = periodic_bad_enforcement_trigger(
            label_new,
            bad_age_new,
            env_step=env_step_next,
            is_frozen=is_frozen,
            enforcement_enabled=bool(self.cfg.enforcement_enabled),
            enforcement_delay=int(self.cfg.enforcement_delay),
            enforcement_period=int(self.cfg.enforcement_period),
        )
        periodic = periodic & enf_enabled_t
        # Peer-beam freezes already applied in the env step; only still-active
        # agents can take a system freeze this step (peer takes precedence).
        still_active = env_state_inner.time_out <= 0
        timeout_trigger = (
            (immediate | delayed_fire | periodic)
            & still_active
            & (~self.police_mask)
        )

        # Probabilistic Bad enforcement (classic P=clip(-s*rep,0,1); binary → P=s).
        strictness_t = resolve_prob_enforcement_strictness(
            state.episodes_completed, self.cfg
        )
        if bool(self.cfg.prob_enforcement_enabled):
            # Use post-step label so newly Bad agents are eligible this step.
            prob_zapped, prob_ps = sample_reputation_enforcement(
                enf_key,
                label_new,
                strictness_t,
                still_active & (~self.police_mask),
            )
            # Do not double-fire if delayed/immediate/periodic already selected.
            prob_zapped = prob_zapped & still_active & (~timeout_trigger)
        else:
            prob_zapped = jnp.zeros((self.num_agents,), dtype=jnp.bool_)
            prob_ps = jnp.zeros((self.num_agents,), dtype=jnp.float32)

        timeout_trigger = timeout_trigger | prob_zapped

        time_out_next, enf_next = apply_existing_timeout_trigger(
            env_state_inner.time_out,
            env_state_inner.timeout_from_enforcement,
            timeout_trigger,
            timeout_duration=int(self.cfg.timeout_duration),
        )
        # Never leave police with a freeze or enforcement cause flag.
        time_out_next = jnp.where(self.police_mask, jnp.int16(0), time_out_next)
        enf_next = jnp.where(self.police_mask, False, enf_next)
        env_state_inner = env_state_inner.replace(
            time_out=time_out_next,
            timeout_from_enforcement=enf_next,
            tag_filter_rep=label_new,
        )

        inner_done = info.get(
            "inner_episode_done",
            jnp.zeros((self.num_agents,), dtype=jnp.bool_),
        )
        inner_done_flag = jnp.asarray(inner_done, dtype=jnp.bool_).reshape(-1)[0]
        episodes_completed_next = jnp.asarray(
            state.episodes_completed, dtype=jnp.int32
        ) + jnp.where(inner_done_flag, jnp.int32(1), jnp.int32(0))

        # Env state may have been auto-reset; keep schedule clock in sync.
        env_state_inner = env_state_inner.replace(
            episodes_completed=episodes_completed_next,
        )

        state_new = ReputationV2EnvState(
            env_state=env_state_inner,
            burst_state=burst_state_new,
            env_step=env_step_next,
            episodes_completed=episodes_completed_next,
        )

        info["burst_trigger"] = burst_over
        info["last_apple_trigger"] = last_apple_bad
        info["immediate_timeout_trigger"] = immediate
        info["delayed_timeout_trigger"] = delayed_fire
        info["periodic_enforcement_trigger"] = periodic
        info["prob_enforcement_zapped"] = prob_zapped
        info["prob_enforcement_prob"] = prob_ps
        info["schedule_prob_strictness"] = strictness_t
        # Fold v2 system freezes into classic enforcement hit / zap fields.
        info["enforcement_timeout_hit"] = (
            info["enforcement_timeout_hit"] | timeout_trigger
        )
        info["reputation_zapped"] = jnp.asarray(
            info["reputation_zapped"], dtype=jnp.bool_
        ) | timeout_trigger
        # Survival clock at system freeze (Bad age when enforcement fires).
        info["enforcement_freeze_bad_age"] = jnp.where(
            timeout_trigger, bad_age_new.astype(jnp.float32), jnp.float32(0.0)
        )
        info["pending_timeout"] = pending_next
        info["burst_window_sum"] = jnp.sum(
            burst_state_new.harvest_window, axis=-1
        )
        info["bad_age"] = bad_age_new
        info["good_age"] = good_age_new
        info["time_out"] = time_out_next
        info["rep"] = label_new
        info["raw_rep"] = label_new
        info["rep_pre"] = pre_label
        info["rep_good"] = label_new > 0.0
        info["rep_bad"] = label_new < 0.0
        info["schedule_timeout_delay"] = timeout_delay_t
        info["schedule_immediate"] = immediate_on
        info["schedule_enforcement_enabled"] = enf_enabled_t
        info["episodes_completed"] = episodes_completed_next

        # Peer punish: ± per *beam* target by rep (optional Good penalty); no police.
        # Uses pre-focus beam so peer_focus_on_bad (no timeout on Good) still
        # charges good_penalty. Folded into returned ``reward`` for PPO only.
        # Logged mean_reward/R uses ``info["environmental_rewards"]`` (apples).
        # Bandit hits extend the target axis with fixed Bad labels (same path).
        bandit_zap = info.get(
            "bandit_zap_matrix",
            jnp.zeros((self.num_agents, 0), dtype=jnp.bool_),
        )
        peer_hits, peer_labels = extended_zap_labels_and_hits(
            zap_for_rewards,
            pre_label,
            jnp.asarray(bandit_zap, dtype=jnp.bool_),
        )
        bad_t, good_t = resolve_peer_punish_magnitudes(
            state.episodes_completed, self.cfg
        )
        peer_rew = peer_punish_zap_reward(
            peer_hits,
            peer_labels,
            enabled=bool(self.cfg.peer_punish_reward_enabled),
            bad_reward=bad_t,
            good_penalty=good_t,
            exclude_mask=self.police_mask,
        )
        # Separate Good vs Bad beam-hit counts (harvester rows; exclude police
        # columns). Freeze hits stay on post-focus ``zap_hit_matrix``.
        pre_lab = jnp.asarray(pre_label, dtype=jnp.float32)
        beam_f = jnp.asarray(zap_for_rewards, dtype=jnp.float32)
        is_good_t = (pre_lab > 0.0).astype(jnp.float32)
        is_bad_t = (pre_lab < 0.0).astype(jnp.float32)
        info["peer_beam_hits_on_good"] = jnp.sum(
            beam_f * is_good_t[None, :], axis=-1
        )
        info["peer_beam_hits_on_bad"] = jnp.sum(
            beam_f * is_bad_t[None, :], axis=-1
        )
        freeze_f = jnp.asarray(zap_for_labels, dtype=jnp.float32)
        info["peer_freeze_hits_on_bad"] = jnp.sum(
            freeze_f * is_bad_t[None, :], axis=-1
        )
        info["peer_freeze_hits_on_good"] = jnp.sum(
            freeze_f * is_good_t[None, :], axis=-1
        )
        info["peer_punish_reward"] = peer_rew
        info["peer_punish_bad_reward"] = bad_t
        info["peer_punish_good_penalty"] = good_t
        if isinstance(reward, dict):
            reward_out = {}
            for i, agent in enumerate(self._env.agents):
                reward_out[agent] = (
                    jnp.asarray(reward[agent], dtype=jnp.float32) + peer_rew[i]
                )
            if "__all__" in reward:
                reward_out["__all__"] = jnp.sum(
                    jnp.stack([reward_out[a] for a in self._env.agents]), axis=0
                )
            reward = reward_out
        else:
            reward = jnp.asarray(reward, dtype=jnp.float32) + peer_rew

        return obs, state_new, reward, done, info

    def render(self, state: ReputationV2EnvState):
        return self._env.render(state.env_state)
