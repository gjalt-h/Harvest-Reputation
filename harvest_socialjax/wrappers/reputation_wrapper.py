"""Stateful reputation wrapper — nests ``rep_state`` like LogWrapper nests episode stats."""
from __future__ import annotations

from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct

from harvest_socialjax.envs.reputation import (
    ReputationConfig,
    ReputationState,
    burst_penalty_schedule,
    init_reputation_state,
    update_reputation,
)
from harvest_socialjax.socialjax_path import ensure_socialjax_on_path

ensure_socialjax_on_path()
from socialjax.wrappers.baselines import JaxMARLWrapper  # noqa: E402


@struct.dataclass
class ReputationEnvState:
    """Wraps the inner env state with reputation bookkeeping."""

    env_state: Any
    rep_state: ReputationState


class ReputationWrapper(JaxMARLWrapper):
    """Adds reputation state, spatial rep channels (ch15/ch16), and ``info["rep"]``.

    Observations are plain spatial tensors ``(H, W, C)`` per agent. Reputation
    reaches the policy only via ch15 ``self_rep`` and ch16 ``other_rep`` inside
    the CNN crop (``tag_filter_rep`` set pre-step on the inner env state).

    Stack: ``Harvest → ReputationWrapper → LogWrapper → CTRolloutManager``.
    """

    def __init__(
        self,
        env,
        cfg: ReputationConfig | None = None,
    ):
        super().__init__(env)
        self.cfg = cfg if cfg is not None else ReputationConfig()
        self.num_agents = int(env.num_agents)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        obs, env_state = self._env.reset(key)
        rep_state = init_reputation_state(self.num_agents, self.cfg)
        state = ReputationEnvState(
            env_state=env_state,
            rep_state=rep_state,
        )
        obs = self._spatial_obs_dict(obs)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: ReputationEnvState, actions):
        pre_step_rep = state.rep_state.rep
        env_state_in = state.env_state.replace(tag_filter_rep=pre_step_rep)
        obs, env_state_inner, reward, done, info = self._env.step(
            key, env_state_in, actions
        )

        harvested_this_step = info["environmental_rewards"]
        zap_hit_matrix = info["zap_hit_matrix"]
        is_frozen = info["tagged"]
        just_respawned = info.get(
            "just_respawned",
            jnp.zeros((self.num_agents,), dtype=jnp.bool_),
        )
        full_reset = done["__all__"]

        rep_state_new = update_reputation(
            state.rep_state,
            harvested_this_step,
            zap_hit_matrix,
            is_frozen,
            full_reset,
            self.cfg,
            just_respawned=just_respawned,
        )

        state_new = ReputationEnvState(
            env_state=env_state_inner,
            rep_state=rep_state_new,
        )
        obs = self._spatial_obs_dict(obs)

        info["rep"] = rep_state_new.rep
        info["raw_rep"] = rep_state_new.raw_rep
        info["rep_pre"] = pre_step_rep
        info["rate_ema"] = rep_state_new.rate_ema
        info["burst_count"] = rep_state_new.burst_count
        info["burst_penalty"] = burst_penalty_schedule(rep_state_new.burst_count)

        return obs, state_new, reward, done, info

    def render(self, state: ReputationEnvState):
        return self._env.render(state.env_state)

    def _spatial_obs_dict(self, obs):
        if isinstance(obs, dict):
            out = {}
            for agent in self.agents:
                key = agent if agent in obs else str(agent)
                entry = obs[key]
                if isinstance(entry, dict) and "spatial" in entry:
                    out[str(agent)] = entry["spatial"]
                else:
                    out[str(agent)] = entry
            return out

        return {str(agent): obs[i] for i, agent in enumerate(self.agents)}
