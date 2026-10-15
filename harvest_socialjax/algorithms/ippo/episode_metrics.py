"""Episode metrics and CNN torso for IPPO harvest training (R/S/P/E, zap, reputation)."""
from __future__ import annotations

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

from harvest_socialjax.envs.police.mechanics import harvester_metric_weights


class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        x = act(nn.Conv(features=32, kernel_size=(5, 5))(x))
        x = act(nn.Conv(features=32, kernel_size=(3, 3))(x))
        x = act(nn.Conv(features=32, kernel_size=(3, 3))(x))
        x = act(nn.Dense(64)(x.reshape((x.shape[0], -1))))
        return x

@struct.dataclass
class StepMetrics:
    """Per-step env stats collected alongside the rollout (batched over envs).

    Kept for tests / offline analysis. The jitted train loop accumulates
    :class:`RolloutLogStats` scalars on-device instead of returning full tensors.
    """
    env_rewards: jnp.ndarray  # (NUM_ENVS, N)
    total_rewards: jnp.ndarray  # (NUM_ENVS, N)
    tagged: jnp.ndarray  # (NUM_ENVS, N)
    zap_successful: jnp.ndarray  # (NUM_ENVS, N)
    zap_attempted: jnp.ndarray  # (NUM_ENVS, N)
    # Inner-episode boundary (Harvest T = num_inner_steps). Prefer this over
    # LogWrapper's returned_episode, which only fires on done["__all__"] (outer).
    inner_episode_done: jnp.ndarray  # (NUM_ENVS, N) bool
    returned_episode: jnp.ndarray  # (NUM_ENVS, N) bool — outer done (LogWrapper)
    returned_episode_returns: jnp.ndarray  # (NUM_ENVS, N)
    # Q diagnostics (averaged over envs/agents this step).
    mean_q: jnp.ndarray  # (A,)
    greedy_zap_rate: jnp.ndarray  # scalar
    mean_q_zap: jnp.ndarray  # scalar
    # Reputation (zeros when ENABLE_REPUTATION is false / DQN path).
    rep: jnp.ndarray  # (NUM_ENVS, N)
    raw_rep: jnp.ndarray  # (NUM_ENVS, N)
    tags_received: jnp.ndarray  # (NUM_ENVS, N)


def compute_peace(tagged: jnp.ndarray, num_agents: int) -> jnp.ndarray:
    """Lupu §4.2 / Perolat: P = N − (1/T) Σ I{timed-out}.

    Pass agent-beam occupancy for peace, or enforcement occupancy for
    system_peace — same formula, different cause mask.
    """
    t = tagged.shape[0]
    return jnp.float32(num_agents) - jnp.sum(tagged.astype(jnp.float32)) / jnp.float32(t)


def update_apple_count_step(apple_count_e: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """On-device apple-count accumulator for one env step (all parallel envs).

    ``info["AppleCount_info"]`` is ``(E, N)`` with the same scalar repeated on
    the agent axis; pass the per-env slice ``[..., 0]`` here. Call every scanned
    step and keep lifetime sum/count in the jitted train carry (same pattern as
    ``ep_lifetime_sum`` / ``ep_lifetime_count``) so log intervals report a mean
    over all env-steps since the previous log â€” not a single noisy snapshot.

    Returns:
        ``(sum, count)`` scalars: sum of per-env apple counts, and ``E``.
    """
    apple_count_e = apple_count_e.astype(jnp.float32)
    return jnp.sum(apple_count_e), jnp.asarray(apple_count_e.shape[0], dtype=jnp.float32)

@struct.dataclass
class PerEpisodeAcc:
    """In-flight per-episode accumulators, one row per parallel env.

    Reward / tagged fields are ``(E, N)``. Apple-depletion fields are ``(E,)``
    (stock is a shared env resource). Carried in the jitted train state and reset
    per-env at each inner-episode boundary (horizon ``T``, or earlier if
    ``terminate_on_barren``) so that R / S / P / E are computed over *full*
    inner episodes rather than a single ``NUM_STEPS`` rollout window.
    """

    # Caller must pass apple ``environmental_rewards`` as total_rew so that
    # R / mean_reward exclude peer-punish and police zap learning bonuses.
    rew_sum: jnp.ndarray       # Σ apple (env) reward over episode → R / mean_reward
    env_ret_sum: jnp.ndarray    # Σ environmental (apple) reward — always ≥ 0
    env_pos_count: jnp.ndarray  # # steps with env (apple) reward > 0
    env_pos_time: jnp.ndarray   # Î£ within-episode time index where env reward > 0
    tagged_sum: jnp.ndarray     # # agent-beam timed-out steps (peace)
    enforcement_tagged_sum: jnp.ndarray  # # enforcement timed-out steps (system_peace)
    police_tagged_sum: jnp.ndarray  # # police-beam timed-out steps (police_peace)
    zap_ok_sum: jnp.ndarray     # # successful zaps over the episode
    zap_try_sum: jnp.ndarray    # # zap attempts over the episode
    police_zap_ok_sum: jnp.ndarray   # # successful police zaps
    police_zap_try_sum: jnp.ndarray  # # police zap attempts
    police_reward_sum: jnp.ndarray   # # Σ police zap reward over the episode
    # Per-agent reputation stats over the in-flight episode (post-step rep).
    rep_min: jnp.ndarray        # (E, N) running min
    rep_max: jnp.ndarray        # (E, N) running max
    rep_sum: jnp.ndarray        # (E, N) Î£ rep
    rep_n: jnp.ndarray          # (E, N) # steps with rep logged
    tags_received_sum: jnp.ndarray  # (E, N) times tagged as zap target
    zap_target_rep_sum: jnp.ndarray  # (E, N) Î£ rep_pre over zap hits (shooter)
    zap_target_hit_sum: jnp.ndarray  # (E, N) count of zap targets hit as shooter
    # --- Bad-targeting diagnostics (attempt vs base rate; see update_per_episode_acc) ---
    # Beam-occupancy targets at zap-attempt time (info["zap_beam_matrix"]).
    target_bad_attempt_num: jnp.ndarray  # (E, N) Σ I{beam[i,j] & rep_pre[j]<0}
    target_bad_attempt_den: jnp.ndarray  # (E, N) Σ I{beam[i,j]}
    target_good_attempt_num: jnp.ndarray  # (E, N) Σ I{beam[i,j] & rep_pre[j]>0}
    target_bad_hit_num: jnp.ndarray      # (E, N) Σ I{hit[i,j] & rep_pre[j]<0}
    target_bad_hit_den: jnp.ndarray      # (E, N) Σ I{hit[i,j]}
    # Shooter-weighted eligible Bad fraction among active non-self non-police.
    base_rate_bad_sum: jnp.ndarray       # (E,) Σ eligible_bad_frac over harvester attempts
    base_rate_bad_n: jnp.ndarray         # (E,) # such attempts
    # Timeout-cause diagnostics (hit events this episode).
    peer_zap_timeout_sum: jnp.ndarray    # (E,) # Bad agents frozen by peer beam
    enf_timeout_sum: jnp.ndarray         # (E,) # agents frozen by system enforcement
    enf_time_to_freeze_sum: jnp.ndarray  # (E,) Σ bad_age at enf freeze
    enf_time_to_freeze_n: jnp.ndarray    # (E,) # enf freezes with age logged
    bad_survival_sum: jnp.ndarray        # (E,) Σ bad_age at any freeze (peer|enf)
    bad_survival_n: jnp.ndarray          # (E,) # such freezes
    good_survival_sum: jnp.ndarray       # (E,) Σ good_age at any freeze of Good (peer|enf)
    good_survival_n: jnp.ndarray         # (E,) # such freezes
    # Env-side bandits (excluded from real-agent zap-accuracy / Bad-targeting).
    bandit_zap_count_sum: jnp.ndarray    # (E,) Σ bandit zap hits this episode
    # Reputation enforcement (negative-rep stochastic timeout).
    enforcement_hit_sum: jnp.ndarray  # (E, N) times this agent was enforcement-zapped
    enforcement_rep_sum: jnp.ndarray  # (E, N) Î£ rep_pre at enforcement hit time
    # Harvest-rate EMA (reputation system's rate_ema), post-step each timestep.
    harvest_rate_min: jnp.ndarray
    harvest_rate_max: jnp.ndarray
    harvest_rate_sum: jnp.ndarray
    harvest_rate_n: jnp.ndarray
    # Burst harvest penalty state / schedule value, post-step each timestep.
    burst_count_min: jnp.ndarray
    burst_count_max: jnp.ndarray
    burst_count_sum: jnp.ndarray
    burst_count_n: jnp.ndarray
    burst_penalty_min: jnp.ndarray
    burst_penalty_max: jnp.ndarray
    burst_penalty_sum: jnp.ndarray
    burst_penalty_n: jnp.ndarray
    # Apple depletion (per env): late-window sum/count, first barren time.
    apple_late_sum: jnp.ndarray  # (E,) Î£ live apples over last 25% of steps
    apple_late_n: jnp.ndarray    # (E,) # late-window steps so far
    barren_t: jnp.ndarray        # (E,) first 0-index t with stock==0; else T

def init_per_episode_acc(
    num_envs: int, num_agents: int, num_inner_steps: int = 1000
) -> PerEpisodeAcc:
    z = jnp.zeros((num_envs, num_agents), dtype=jnp.float32)
    ze = jnp.zeros((num_envs,), dtype=jnp.float32)
    # Sentinel: never barren within the episode â†’ report T (horizon).
    barren0 = jnp.full((num_envs,), jnp.float32(num_inner_steps), dtype=jnp.float32)
    return PerEpisodeAcc(
        rew_sum=z,
        env_ret_sum=z,
        env_pos_count=z,
        env_pos_time=z,
        tagged_sum=z,
        enforcement_tagged_sum=z,
        police_tagged_sum=z,
        zap_ok_sum=z,
        zap_try_sum=z,
        police_zap_ok_sum=z,
        police_zap_try_sum=z,
        police_reward_sum=z,
        rep_min=jnp.full((num_envs, num_agents), jnp.inf, dtype=jnp.float32),
        rep_max=jnp.full((num_envs, num_agents), -jnp.inf, dtype=jnp.float32),
        rep_sum=z,
        rep_n=z,
        tags_received_sum=z,
        zap_target_rep_sum=z,
        zap_target_hit_sum=z,
        target_bad_attempt_num=z,
        target_bad_attempt_den=z,
        target_good_attempt_num=z,
        target_bad_hit_num=z,
        target_bad_hit_den=z,
        base_rate_bad_sum=ze,
        base_rate_bad_n=ze,
        peer_zap_timeout_sum=ze,
        enf_timeout_sum=ze,
        enf_time_to_freeze_sum=ze,
        enf_time_to_freeze_n=ze,
        bad_survival_sum=ze,
        bad_survival_n=ze,
        good_survival_sum=ze,
        good_survival_n=ze,
        bandit_zap_count_sum=ze,
        enforcement_hit_sum=z,
        enforcement_rep_sum=z,
        harvest_rate_min=jnp.full((num_envs, num_agents), jnp.inf, dtype=jnp.float32),
        harvest_rate_max=jnp.full((num_envs, num_agents), -jnp.inf, dtype=jnp.float32),
        harvest_rate_sum=z,
        harvest_rate_n=z,
        burst_count_min=jnp.full((num_envs, num_agents), jnp.inf, dtype=jnp.float32),
        burst_count_max=jnp.full((num_envs, num_agents), -jnp.inf, dtype=jnp.float32),
        burst_count_sum=z,
        burst_count_n=z,
        burst_penalty_min=jnp.full((num_envs, num_agents), jnp.inf, dtype=jnp.float32),
        burst_penalty_max=jnp.full((num_envs, num_agents), -jnp.inf, dtype=jnp.float32),
        burst_penalty_sum=z,
        burst_penalty_n=z,
        apple_late_sum=ze,
        apple_late_n=ze,
        barren_t=barren0,
    )

@struct.dataclass
class EpisodeAggregates:
    """Bundled scalar aggregates threaded through the jitted train carry.

    Used both as the per-update *window* sums (episodes that completed in one
    rollout) and as the *lifetime* running totals (sum since training start).
    Host code reads lifetime fields at each log point and divides deltas by the
    ``count`` delta to get interval means (except ``apple_sum`` / ``apple_count``,
    which are per-step means: sum over env-steps / number of env-steps).

    Per-episode apple depletion (emitted only on completed episodes):
      * ``apple_end_sum`` â€” final live stock at t=Tâˆ’1
      * ``apple_late_mean_sum`` â€” mean stock over the last 25% of steps
      * ``barren_t_sum`` â€” first barren timestep (0..Tâˆ’1), or T if never barren
      * ``barren_event_sum`` â€” 1 if stock hit 0 at least once, else 0
      * ``tagged_out_frac_sum`` â€” fraction of agent-timesteps tagged-out
      * ``reward_variance_across_agents_sum`` â€” var of per-agent apple returns
    """

    R: jnp.ndarray
    S: jnp.ndarray
    P: jnp.ndarray
    P_sys: jnp.ndarray  # system_peace: enforcement timeouts only
    P_police: jnp.ndarray  # police_peace: police-beam timeouts only
    E: jnp.ndarray
    E_total: jnp.ndarray
    count: jnp.ndarray
    apple_sum: jnp.ndarray
    apple_count: jnp.ndarray
    zap_success_rate_sum: jnp.ndarray
    zap_attempt_rate_sum: jnp.ndarray
    zap_hit_rate_sum: jnp.ndarray
    # Reputation enforcement (completed-episode sums).
    enforcement_hit_rate_sum: jnp.ndarray   # hits / (ep_len * N) per episode
    enforcement_hit_count_sum: jnp.ndarray  # total enforcement hits
    enforcement_rep_total_sum: jnp.ndarray  # sum of rep_pre over enforcement hits
    # Zap-tagged victims: hit-weighted sum of target rep_pre (harvester shooters only).
    zap_tagged_hit_count_sum: jnp.ndarray
    zap_tagged_rep_total_sum: jnp.ndarray
    # Same for police shooters only (logged under enforcement/*).
    police_zap_tagged_hit_count_sum: jnp.ndarray
    police_zap_tagged_rep_total_sum: jnp.ndarray
    apple_end_sum: jnp.ndarray
    apple_late_mean_sum: jnp.ndarray
    barren_t_sum: jnp.ndarray
    barren_event_sum: jnp.ndarray
    tagged_out_frac_sum: jnp.ndarray  # agent-beam timeout occupancy
    enforcement_tagged_out_frac_sum: jnp.ndarray  # enforcement timeout occupancy
    police_tagged_out_frac_sum: jnp.ndarray  # police timeout occupancy
    police_zap_attempt_rate_sum: jnp.ndarray
    police_zap_hit_rate_sum: jnp.ndarray
    police_reward_sum: jnp.ndarray  # Σ police zap reward per completed episode
    reward_variance_across_agents_sum: jnp.ndarray
    # Bad-targeting diagnostics (episode means; see update_per_episode_acc block).
    base_rate_bad_ep_sum: jnp.ndarray  # eligible Bad frac at zap-attempt time
    # Timeout causes / Bad survival under probabilistic enforcement.
    peer_zap_timeouts_ep_sum: jnp.ndarray  # # Bad peer-beam freezes / episode
    enforcement_timeouts_ep_sum: jnp.ndarray  # # system-enf freezes / episode
    enforcement_time_to_freeze_ep_sum: jnp.ndarray  # mean Bad age at enf freeze
    bad_survival_ep_sum: jnp.ndarray  # mean Bad age at any freeze (peer|enf)
    good_survival_ep_sum: jnp.ndarray  # mean Good age at any freeze of Good (peer|enf)
    bandit_zap_count_ep_sum: jnp.ndarray  # # bandit zap hits / episode
    bandit_zap_rate_ep_sum: jnp.ndarray  # bandit hits / zap attempts / episode
    # Per-agent sums over completed episodes (not checkpointed; interval / count on host).
    agent_rep_min_sum: jnp.ndarray       # (N,)
    agent_rep_max_sum: jnp.ndarray       # (N,)
    agent_rep_mean_sum: jnp.ndarray      # (N,)
    agent_zap_ok_ep_sum: jnp.ndarray     # (N,) timesteps with >=1 hit
    agent_zap_try_ep_sum: jnp.ndarray    # (N,)
    agent_zap_targets_hit_ep_sum: jnp.ndarray  # (N,) total targets tagged (matrix row sum)
    agent_tags_received_ep_sum: jnp.ndarray  # (N,)
    agent_harvest_rate_mean_sum: jnp.ndarray   # (N,)
    agent_harvest_rate_max_sum: jnp.ndarray    # (N,)
    agent_burst_count_mean_sum: jnp.ndarray    # (N,)
    agent_burst_count_max_sum: jnp.ndarray     # (N,)
    agent_burst_penalty_mean_sum: jnp.ndarray  # (N,)
    agent_burst_penalty_max_sum: jnp.ndarray   # (N,)
    agent_zap_target_rep_mean_sum: jnp.ndarray  # (N,)
    agent_target_bad_rate_attempted_ep_sum: jnp.ndarray  # (N,) Bad frac of beam targets
    agent_target_bad_rate_hit_ep_sum: jnp.ndarray  # (N,) Bad frac of tagged targets
    agent_peer_beam_hits_good_ep_sum: jnp.ndarray  # (N,) # Good in beam (pre-focus)
    agent_peer_beam_hits_bad_ep_sum: jnp.ndarray  # (N,) # Bad in beam (pre-focus)
    agent_enforcement_hits_ep_sum: jnp.ndarray  # (N,)
    agent_enforcement_rep_mean_sum: jnp.ndarray  # (N,)
    agent_return_ep_sum: jnp.ndarray          # (N,) apple env return per episode

def init_episode_aggregates(num_agents: int = 1) -> EpisodeAggregates:
    z = jnp.array(0.0, dtype=jnp.float32)
    zn = jnp.zeros((num_agents,), dtype=jnp.float32)
    return EpisodeAggregates(
        R=z,
        S=z,
        P=z,
        P_sys=z,
        P_police=z,
        E=z,
        E_total=z,
        count=z,
        apple_sum=z,
        apple_count=z,
        zap_success_rate_sum=z,
        zap_attempt_rate_sum=z,
        zap_hit_rate_sum=z,
        enforcement_hit_rate_sum=z,
        enforcement_hit_count_sum=z,
        enforcement_rep_total_sum=z,
        zap_tagged_hit_count_sum=z,
        zap_tagged_rep_total_sum=z,
        police_zap_tagged_hit_count_sum=z,
        police_zap_tagged_rep_total_sum=z,
        apple_end_sum=z,
        apple_late_mean_sum=z,
        barren_t_sum=z,
        barren_event_sum=z,
        tagged_out_frac_sum=z,
        enforcement_tagged_out_frac_sum=z,
        police_tagged_out_frac_sum=z,
        police_zap_attempt_rate_sum=z,
        police_zap_hit_rate_sum=z,
        police_reward_sum=z,
        reward_variance_across_agents_sum=z,
        base_rate_bad_ep_sum=z,
        peer_zap_timeouts_ep_sum=z,
        enforcement_timeouts_ep_sum=z,
        enforcement_time_to_freeze_ep_sum=z,
        bad_survival_ep_sum=z,
        good_survival_ep_sum=z,
        bandit_zap_count_ep_sum=z,
        bandit_zap_rate_ep_sum=z,
        agent_rep_min_sum=zn,
        agent_rep_max_sum=zn,
        agent_rep_mean_sum=zn,
        agent_zap_ok_ep_sum=zn,
        agent_zap_try_ep_sum=zn,
        agent_zap_targets_hit_ep_sum=zn,
        agent_tags_received_ep_sum=zn,
        agent_harvest_rate_mean_sum=zn,
        agent_harvest_rate_max_sum=zn,
        agent_burst_count_mean_sum=zn,
        agent_burst_count_max_sum=zn,
        agent_burst_penalty_mean_sum=zn,
        agent_burst_penalty_max_sum=zn,
        agent_zap_target_rep_mean_sum=zn,
        agent_target_bad_rate_attempted_ep_sum=zn,
        agent_target_bad_rate_hit_ep_sum=zn,
        agent_peer_beam_hits_good_ep_sum=zn,
        agent_peer_beam_hits_bad_ep_sum=zn,
        agent_enforcement_hits_ep_sum=zn,
        agent_enforcement_rep_mean_sum=zn,
        agent_return_ep_sum=zn,
    )

def add_episode_aggregates(
    a: EpisodeAggregates, b: EpisodeAggregates
) -> EpisodeAggregates:
    return EpisodeAggregates(
        R=a.R + b.R,
        S=a.S + b.S,
        P=a.P + b.P,
        P_sys=a.P_sys + b.P_sys,
        P_police=a.P_police + b.P_police,
        E=a.E + b.E,
        E_total=a.E_total + b.E_total,
        count=a.count + b.count,
        apple_sum=a.apple_sum + b.apple_sum,
        apple_count=a.apple_count + b.apple_count,
        zap_success_rate_sum=a.zap_success_rate_sum + b.zap_success_rate_sum,
        zap_attempt_rate_sum=a.zap_attempt_rate_sum + b.zap_attempt_rate_sum,
        zap_hit_rate_sum=a.zap_hit_rate_sum + b.zap_hit_rate_sum,
        enforcement_hit_rate_sum=(
            a.enforcement_hit_rate_sum + b.enforcement_hit_rate_sum
        ),
        enforcement_hit_count_sum=(
            a.enforcement_hit_count_sum + b.enforcement_hit_count_sum
        ),
        enforcement_rep_total_sum=(
            a.enforcement_rep_total_sum + b.enforcement_rep_total_sum
        ),
        zap_tagged_hit_count_sum=(
            a.zap_tagged_hit_count_sum + b.zap_tagged_hit_count_sum
        ),
        zap_tagged_rep_total_sum=(
            a.zap_tagged_rep_total_sum + b.zap_tagged_rep_total_sum
        ),
        police_zap_tagged_hit_count_sum=(
            a.police_zap_tagged_hit_count_sum + b.police_zap_tagged_hit_count_sum
        ),
        police_zap_tagged_rep_total_sum=(
            a.police_zap_tagged_rep_total_sum + b.police_zap_tagged_rep_total_sum
        ),
        apple_end_sum=a.apple_end_sum + b.apple_end_sum,
        apple_late_mean_sum=a.apple_late_mean_sum + b.apple_late_mean_sum,
        barren_t_sum=a.barren_t_sum + b.barren_t_sum,
        barren_event_sum=a.barren_event_sum + b.barren_event_sum,
        tagged_out_frac_sum=a.tagged_out_frac_sum + b.tagged_out_frac_sum,
        enforcement_tagged_out_frac_sum=(
            a.enforcement_tagged_out_frac_sum + b.enforcement_tagged_out_frac_sum
        ),
        police_tagged_out_frac_sum=(
            a.police_tagged_out_frac_sum + b.police_tagged_out_frac_sum
        ),
        police_zap_attempt_rate_sum=(
            a.police_zap_attempt_rate_sum + b.police_zap_attempt_rate_sum
        ),
        police_zap_hit_rate_sum=(
            a.police_zap_hit_rate_sum + b.police_zap_hit_rate_sum
        ),
        police_reward_sum=a.police_reward_sum + b.police_reward_sum,
        reward_variance_across_agents_sum=(
            a.reward_variance_across_agents_sum + b.reward_variance_across_agents_sum
        ),
        base_rate_bad_ep_sum=a.base_rate_bad_ep_sum + b.base_rate_bad_ep_sum,
        peer_zap_timeouts_ep_sum=(
            a.peer_zap_timeouts_ep_sum + b.peer_zap_timeouts_ep_sum
        ),
        enforcement_timeouts_ep_sum=(
            a.enforcement_timeouts_ep_sum + b.enforcement_timeouts_ep_sum
        ),
        enforcement_time_to_freeze_ep_sum=(
            a.enforcement_time_to_freeze_ep_sum + b.enforcement_time_to_freeze_ep_sum
        ),
        bad_survival_ep_sum=a.bad_survival_ep_sum + b.bad_survival_ep_sum,
        good_survival_ep_sum=a.good_survival_ep_sum + b.good_survival_ep_sum,
        bandit_zap_count_ep_sum=a.bandit_zap_count_ep_sum + b.bandit_zap_count_ep_sum,
        bandit_zap_rate_ep_sum=a.bandit_zap_rate_ep_sum + b.bandit_zap_rate_ep_sum,
        agent_rep_min_sum=a.agent_rep_min_sum + b.agent_rep_min_sum,
        agent_rep_max_sum=a.agent_rep_max_sum + b.agent_rep_max_sum,
        agent_rep_mean_sum=a.agent_rep_mean_sum + b.agent_rep_mean_sum,
        agent_zap_ok_ep_sum=a.agent_zap_ok_ep_sum + b.agent_zap_ok_ep_sum,
        agent_zap_try_ep_sum=a.agent_zap_try_ep_sum + b.agent_zap_try_ep_sum,
        agent_zap_targets_hit_ep_sum=(
            a.agent_zap_targets_hit_ep_sum + b.agent_zap_targets_hit_ep_sum
        ),
        agent_tags_received_ep_sum=(
            a.agent_tags_received_ep_sum + b.agent_tags_received_ep_sum
        ),
        agent_harvest_rate_mean_sum=(
            a.agent_harvest_rate_mean_sum + b.agent_harvest_rate_mean_sum
        ),
        agent_harvest_rate_max_sum=(
            a.agent_harvest_rate_max_sum + b.agent_harvest_rate_max_sum
        ),
        agent_burst_count_mean_sum=(
            a.agent_burst_count_mean_sum + b.agent_burst_count_mean_sum
        ),
        agent_burst_count_max_sum=(
            a.agent_burst_count_max_sum + b.agent_burst_count_max_sum
        ),
        agent_burst_penalty_mean_sum=(
            a.agent_burst_penalty_mean_sum + b.agent_burst_penalty_mean_sum
        ),
        agent_burst_penalty_max_sum=(
            a.agent_burst_penalty_max_sum + b.agent_burst_penalty_max_sum
        ),
        agent_zap_target_rep_mean_sum=(
            a.agent_zap_target_rep_mean_sum + b.agent_zap_target_rep_mean_sum
        ),
        agent_target_bad_rate_attempted_ep_sum=(
            a.agent_target_bad_rate_attempted_ep_sum
            + b.agent_target_bad_rate_attempted_ep_sum
        ),
        agent_target_bad_rate_hit_ep_sum=(
            a.agent_target_bad_rate_hit_ep_sum + b.agent_target_bad_rate_hit_ep_sum
        ),
        agent_peer_beam_hits_good_ep_sum=(
            a.agent_peer_beam_hits_good_ep_sum + b.agent_peer_beam_hits_good_ep_sum
        ),
        agent_peer_beam_hits_bad_ep_sum=(
            a.agent_peer_beam_hits_bad_ep_sum + b.agent_peer_beam_hits_bad_ep_sum
        ),
        agent_enforcement_hits_ep_sum=(
            a.agent_enforcement_hits_ep_sum + b.agent_enforcement_hits_ep_sum
        ),
        agent_enforcement_rep_mean_sum=(
            a.agent_enforcement_rep_mean_sum + b.agent_enforcement_rep_mean_sum
        ),
        agent_return_ep_sum=a.agent_return_ep_sum + b.agent_return_ep_sum,
    )

def update_per_episode_acc(
    acc: PerEpisodeAcc,
    total_rew: jnp.ndarray,
    env_rew: jnp.ndarray,
    tagged: jnp.ndarray,
    zap_ok: jnp.ndarray,
    zap_try: jnp.ndarray,
    inner_t: jnp.ndarray,
    inner_done: jnp.ndarray,
    num_inner_steps: int,
    num_agents: int,
    apple_e: jnp.ndarray,
    rep: jnp.ndarray | None = None,
    tags_received: jnp.ndarray | None = None,
    rep_pre: jnp.ndarray | None = None,
    rate_ema: jnp.ndarray | None = None,
    burst_count: jnp.ndarray | None = None,
    burst_penalty: jnp.ndarray | None = None,
    zap_hit_matrix: jnp.ndarray | None = None,
    reputation_zapped: jnp.ndarray | None = None,
    enforcement_tagged: jnp.ndarray | None = None,
    police_mask: jnp.ndarray | None = None,
    police_tagged: jnp.ndarray | None = None,
    police_zap_try: jnp.ndarray | None = None,
    police_zap_ok: jnp.ndarray | None = None,
    police_reward: jnp.ndarray | None = None,
    zap_beam_matrix: jnp.ndarray | None = None,
    active: jnp.ndarray | None = None,
    zap_attempted_agents: jnp.ndarray | None = None,
    agent_beam_timeout_hit: jnp.ndarray | None = None,
    enforcement_timeout_hit: jnp.ndarray | None = None,
    enforcement_freeze_bad_age: jnp.ndarray | None = None,
    bad_age: jnp.ndarray | None = None,
    good_age: jnp.ndarray | None = None,
    bandit_zap_count: jnp.ndarray | None = None,
) -> tuple:
    """Accumulate one env step and emit completed-episode metric sums.

    Paper metrics (Lupu & Precup §4.2), computed over the full ``T``-step
    episode for every env that finishes on this step:

      * ``R``  = mean over harvesters of per-agent *apple* episode return
        (``total_rew`` must be ``environmental_rewards`` — never peer-punish
        or police zap bonuses; those are PPO-only learning signals)
      * ``S``  = mean over agents of E[t | env_reward_{i,t} > 0]
      * ``P``  = N − (1/T) Σ I{agent-beam timed out}  (peace)
      * ``P_sys`` = N − (1/T) Σ I{enforcement timed out}  (system_peace)
      * ``P_police`` = N_h − (1/T) Σ I{police-beam timed out}  (police_peace)
      * ``E``  = 1 − Gini over per-agent *apple* returns (≥ 0; the go-to)
      * ``E_total`` = 1 − Raffinetti-Siletti-Vernizzi Gini over the same
        ``total_rew`` stream (matches ``E`` when that stream is apple-only)
      * ``zap_*`` = per-episode action rates, computed within each completed
        episode and then averaged across episodes on the host side

    Bad-targeting diagnostics (metrics-only; no reward effect)
    ----------------------------------------------------------
    Uses pre-step ``rep_pre`` and beam occupancy at zap-attempt time. Bad =
    ``rep_pre < 0`` (binary v2: −1). Eligible targets for ``base_rate_bad_ep``
    match who the beam can tag: active (not timed-out/frozen), not self, not
    police-immune. ``zap_beam_matrix[i,j]`` = shooter i's zap beam covers j
    (attempt-time, before police-focus filtering). ``zap_hit_matrix`` is the
    post-focus tag matrix (successful hits).

      * ``base_rate_bad_ep`` — mean over harvester zap attempts of
        (# eligible Bad) / (# eligible) for that shooter
      * ``target_bad_rate_attempted_ep`` — per shooter, Bad frac among beam
        occupants over the episode
      * ``target_bad_rate_hit_ep`` — same, conditioned on successful tags

    Timeout-cause / Bad survival (hit events; harvesters only)
    ---------------------------------------------------------
      * ``peer_zap_timeouts_ep`` — # Bad agents frozen by peer beam this episode
      * ``enforcement_timeouts_ep`` — # agents frozen by system enforcement
      * ``enforcement_time_to_freeze_ep`` — mean ``bad_age`` at enf freeze
      * ``bad_survival_ep`` — mean ``bad_age`` at any freeze (peer or enf)
      * ``good_survival_ep`` — mean ``good_age`` at any freeze of Good (peer or enf)

    ``tagged`` must be agent-beam timeout occupancy (``info["agent_tagged"]``).
    ``enforcement_tagged`` is enforcement timeout occupancy
    (``info["enforcement_tagged"]``). Hit-event counts still use
    ``reputation_zapped``. ``police_tagged`` is police-beam timeout occupancy.

    Returns
    ``(new_acc, R_sum, S_sum, P_sum, E_sum, E_total_sum, count,
      ..., enforcement_rep_total_sum, zap_tagged_hit_count_sum,
      zap_tagged_rep_total_sum,
      police_zap_tagged_hit_count_sum, police_zap_tagged_rep_total_sum,
      P_sys_sum,
      enforcement_tagged_out_frac_sum,
      P_police_sum, police_tagged_out_frac_sum,
      police_zap_attempt_rate_sum, police_zap_hit_rate_sum,
      police_reward_ep_sum,
      agent_target_bad_rate_attempted_ep_sum,
      agent_target_bad_rate_hit_ep_sum,
      base_rate_bad_ep_sum,
      peer_zap_timeouts_ep_sum,
      enforcement_timeouts_ep_sum,
      enforcement_time_to_freeze_ep_sum,
      bad_survival_ep_sum,
      good_survival_ep_sum)``.
    Host code divides ``*_sum`` deltas by ``count`` deltas between log points.
    """
    harvester_w, n_h = harvester_metric_weights(num_agents, police_mask)
    n_police = jnp.float32(num_agents) - n_h
    police_w = jnp.float32(1.0) - harvester_w
    pair_w = harvester_w[:, None] * harvester_w[None, :]

    rew_sum = acc.rew_sum + total_rew
    env_ret_sum = acc.env_ret_sum + jnp.maximum(env_rew, 0.0)
    pos = env_rew > 0
    env_pos_count = acc.env_pos_count + pos.astype(jnp.float32)
    # 0-indexed within-episode time (inner_t is post-increment: 1..T).
    t0 = jnp.maximum(inner_t.astype(jnp.float32) - 1.0, 0.0)
    env_pos_time = acc.env_pos_time + jnp.where(pos, t0, 0.0)
    tagged_sum = acc.tagged_sum + tagged.astype(jnp.float32)
    if enforcement_tagged is None:
        enforcement_tagged = jnp.zeros_like(tagged)
    enforcement_tagged_sum = (
        acc.enforcement_tagged_sum + enforcement_tagged.astype(jnp.float32)
    )
    if police_tagged is None:
        police_tagged = jnp.zeros_like(tagged)
    police_tagged_sum = acc.police_tagged_sum + police_tagged.astype(jnp.float32)
    zap_ok_sum = acc.zap_ok_sum + zap_ok.astype(jnp.float32)
    zap_try_sum = acc.zap_try_sum + zap_try.astype(jnp.float32)
    if police_zap_ok is None:
        police_zap_ok = jnp.zeros_like(zap_ok)
    if police_zap_try is None:
        police_zap_try = jnp.zeros_like(zap_try)
    if police_reward is None:
        police_reward = jnp.zeros_like(total_rew)
    police_zap_ok_sum = acc.police_zap_ok_sum + police_zap_ok.astype(jnp.float32)
    police_zap_try_sum = acc.police_zap_try_sum + police_zap_try.astype(jnp.float32)
    police_reward_sum = acc.police_reward_sum + police_reward.astype(jnp.float32)

    if rep is None:
        rep = jnp.zeros_like(acc.rep_sum)
    if tags_received is None:
        tags_received = jnp.zeros_like(acc.tags_received_sum)
    rep = rep.astype(jnp.float32)
    tags_received = tags_received.astype(jnp.float32)
    rep_min = jnp.minimum(acc.rep_min, rep)
    rep_max = jnp.maximum(acc.rep_max, rep)
    rep_sum = acc.rep_sum + rep
    rep_n = acc.rep_n + 1.0
    tags_received_sum = acc.tags_received_sum + tags_received

    if rate_ema is None:
        rate_ema = jnp.zeros_like(acc.harvest_rate_sum)
    rate_ema = rate_ema.astype(jnp.float32)
    harvest_rate_min = jnp.minimum(acc.harvest_rate_min, rate_ema)
    harvest_rate_max = jnp.maximum(acc.harvest_rate_max, rate_ema)
    harvest_rate_sum = acc.harvest_rate_sum + rate_ema
    harvest_rate_n = acc.harvest_rate_n + 1.0

    if burst_count is None:
        burst_count = jnp.zeros_like(acc.burst_count_sum)
    burst_count = burst_count.astype(jnp.float32)
    burst_count_min = jnp.minimum(acc.burst_count_min, burst_count)
    burst_count_max = jnp.maximum(acc.burst_count_max, burst_count)
    burst_count_sum = acc.burst_count_sum + burst_count
    burst_count_n = acc.burst_count_n + 1.0

    if burst_penalty is None:
        burst_penalty = jnp.zeros_like(acc.burst_penalty_sum)
    burst_penalty = burst_penalty.astype(jnp.float32)
    burst_penalty_min = jnp.minimum(acc.burst_penalty_min, burst_penalty)
    burst_penalty_max = jnp.maximum(acc.burst_penalty_max, burst_penalty)
    burst_penalty_sum = acc.burst_penalty_sum + burst_penalty
    burst_penalty_n = acc.burst_penalty_n + 1.0

    zap_target_rep_sum = acc.zap_target_rep_sum
    zap_target_hit_sum = acc.zap_target_hit_sum
    target_bad_attempt_num = acc.target_bad_attempt_num
    target_bad_attempt_den = acc.target_bad_attempt_den
    target_good_attempt_num = acc.target_good_attempt_num
    target_bad_hit_num = acc.target_bad_hit_num
    target_bad_hit_den = acc.target_bad_hit_den
    base_rate_bad_sum = acc.base_rate_bad_sum
    base_rate_bad_n = acc.base_rate_bad_n
    if zap_hit_matrix is not None and rep_pre is not None:
        zm = zap_hit_matrix.astype(jnp.float32)
        rep_pre = rep_pre.astype(jnp.float32)
        zap_target_rep_step = jnp.sum(zm * rep_pre[:, None, :], axis=-1)
        zap_target_hit_step = jnp.sum(zm, axis=-1)
        zap_target_rep_sum = acc.zap_target_rep_sum + zap_target_rep_step
        zap_target_hit_sum = acc.zap_target_hit_sum + zap_target_hit_step
        # --- Bad-targeting diagnostics (hit-conditioned) ---
        is_bad = (rep_pre < 0.0).astype(jnp.float32)
        target_bad_hit_num = acc.target_bad_hit_num + jnp.sum(
            zm * is_bad[:, None, :], axis=-1
        )
        target_bad_hit_den = acc.target_bad_hit_den + zap_target_hit_step

    # Attempt-time beam occupancy (falls back to hit matrix if beam missing).
    beam_m = zap_beam_matrix if zap_beam_matrix is not None else zap_hit_matrix
    if beam_m is not None and rep_pre is not None:
        bm = beam_m.astype(jnp.float32)
        is_bad = (rep_pre.astype(jnp.float32) < 0.0).astype(jnp.float32)
        is_good = (rep_pre.astype(jnp.float32) > 0.0).astype(jnp.float32)
        target_bad_attempt_num = acc.target_bad_attempt_num + jnp.sum(
            bm * is_bad[:, None, :], axis=-1
        )
        target_bad_attempt_den = acc.target_bad_attempt_den + jnp.sum(bm, axis=-1)
        target_good_attempt_num = acc.target_good_attempt_num + jnp.sum(
            bm * is_good[:, None, :], axis=-1
        )

        # base_rate_bad: per harvester zap attempt, eligible Bad fraction.
        if active is None:
            active_m = jnp.ones_like(total_rew, dtype=jnp.bool_)
        else:
            active_m = active.astype(jnp.bool_)
        if zap_attempted_agents is None:
            # Any shooter with ≥1 beam occupant, else fall back to row-any of beam.
            tried = jnp.any(bm > 0.0, axis=-1)
        else:
            tried = zap_attempted_agents.astype(jnp.bool_)
        harv_try = tried & (harvester_w[None, :] > 0.0) & active_m
        # Eligible j for shooter i: active, not self, not police.
        eye = jnp.eye(num_agents, dtype=jnp.bool_)
        eligible = (
            active_m[:, None, :]
            & (~eye[None, :, :])
            & (harvester_w[None, None, :] > 0.0)  # exclude police victims
        )
        # Also exclude police shooters already via harv_try; victims use harvester_w.
        # Police are never harvesters, so harvester_w on victim axis = non-police.
        n_elig = jnp.sum(eligible.astype(jnp.float32), axis=-1)
        n_bad_elig = jnp.sum(
            eligible.astype(jnp.float32) * is_bad[:, None, :], axis=-1
        )
        frac = n_bad_elig / jnp.maximum(n_elig, 1.0)
        base_rate_bad_sum = acc.base_rate_bad_sum + jnp.sum(
            jnp.where(harv_try, frac, 0.0), axis=-1
        )
        base_rate_bad_n = acc.base_rate_bad_n + jnp.sum(
            harv_try.astype(jnp.float32), axis=-1
        )

    if reputation_zapped is None:
        reputation_zapped = jnp.zeros_like(acc.enforcement_hit_sum)
    enforcement_hit = reputation_zapped.astype(jnp.float32)
    enforcement_hit_sum = acc.enforcement_hit_sum + enforcement_hit
    if rep_pre is None:
        rep_pre_enf = jnp.zeros_like(enforcement_hit)
    else:
        rep_pre_enf = rep_pre.astype(jnp.float32)
    enforcement_rep_sum = acc.enforcement_rep_sum + enforcement_hit * rep_pre_enf

    # --- Timeout-cause / Bad survival (hit events; harvesters only) ----------
    if agent_beam_timeout_hit is None:
        peer_hit = jnp.zeros_like(enforcement_hit, dtype=jnp.bool_)
    else:
        peer_hit = agent_beam_timeout_hit.astype(jnp.bool_)
    if enforcement_timeout_hit is None:
        enf_hit = reputation_zapped.astype(jnp.bool_)
    else:
        enf_hit = enforcement_timeout_hit.astype(jnp.bool_)
    if enforcement_freeze_bad_age is None:
        enf_freeze_age = jnp.zeros_like(enforcement_hit)
    else:
        enf_freeze_age = enforcement_freeze_bad_age.astype(jnp.float32)
    if bad_age is None:
        age_now = enf_freeze_age
    else:
        age_now = bad_age.astype(jnp.float32)
    if rep is not None:
        is_bad = rep.astype(jnp.float32) < 0.0
    elif rep_pre is not None:
        is_bad = rep_pre.astype(jnp.float32) < 0.0
    else:
        is_bad = jnp.zeros_like(enforcement_hit, dtype=jnp.bool_)
    harv_m = harvester_w[None, :] > 0.0
    peer_bad = peer_hit & is_bad & harv_m
    # Peer takes precedence when both fire on the same step.
    enf_only = enf_hit & (~peer_hit) & harv_m
    peer_zap_timeout_sum = acc.peer_zap_timeout_sum + jnp.sum(
        peer_bad.astype(jnp.float32), axis=-1
    )
    enf_timeout_sum = acc.enf_timeout_sum + jnp.sum(
        enf_only.astype(jnp.float32), axis=-1
    )
    enf_age = jnp.where(enf_only, enf_freeze_age, 0.0)
    enf_age = jnp.where(enf_only & (enf_age <= 0.0), age_now, enf_age)
    enf_time_to_freeze_sum = acc.enf_time_to_freeze_sum + jnp.sum(enf_age, axis=-1)
    enf_time_to_freeze_n = acc.enf_time_to_freeze_n + jnp.sum(
        enf_only.astype(jnp.float32), axis=-1
    )
    any_freeze_bad = peer_bad | (enf_only & is_bad)
    surv_age = jnp.where(enf_only, enf_age, jnp.where(peer_bad, age_now, 0.0))
    bad_survival_sum = acc.bad_survival_sum + jnp.sum(
        jnp.where(any_freeze_bad, surv_age, 0.0), axis=-1
    )
    bad_survival_n = acc.bad_survival_n + jnp.sum(
        any_freeze_bad.astype(jnp.float32), axis=-1
    )

    if good_age is None:
        good_age_now = jnp.zeros_like(age_now)
    else:
        good_age_now = good_age.astype(jnp.float32)
    if rep is not None:
        is_good = rep.astype(jnp.float32) > 0.0
    elif rep_pre is not None:
        is_good = rep_pre.astype(jnp.float32) > 0.0
    else:
        is_good = jnp.zeros_like(enforcement_hit, dtype=jnp.bool_)
    peer_good = peer_hit & is_good & harv_m
    enf_good = enf_only & is_good & harv_m
    any_freeze_good = peer_good | enf_good
    good_surv_age = jnp.where(
        enf_good, enf_age, jnp.where(peer_good, good_age_now, 0.0)
    )
    good_surv_age = jnp.where(
        any_freeze_good & (good_surv_age <= 0.0), good_age_now, good_surv_age
    )
    good_survival_sum = acc.good_survival_sum + jnp.sum(
        jnp.where(any_freeze_good, good_surv_age, 0.0), axis=-1
    )
    good_survival_n = acc.good_survival_n + jnp.sum(
        any_freeze_good.astype(jnp.float32), axis=-1
    )

    # Bandit zaps: tracked separately; never mixed into NxN Bad-targeting mats.
    if bandit_zap_count is None:
        bandit_hits = jnp.zeros((total_rew.shape[0],), dtype=jnp.float32)
    else:
        bandit_hits = jnp.asarray(bandit_zap_count, dtype=jnp.float32).reshape(-1)
    bandit_zap_count_sum = acc.bandit_zap_count_sum + bandit_hits

    done_e = inner_done[..., 0]
    d = done_e.astype(jnp.float32)
    t0_e = t0[..., 0]
    T = jnp.float32(num_inner_steps)

    # --- apple depletion tracking (shared env resource) ---
    apple_e = apple_e.astype(jnp.float32)
    late_len = max(int(num_inner_steps) // 4, 1)
    late_start = jnp.float32(num_inner_steps - late_len)
    in_late = t0_e >= late_start
    apple_late_sum = acc.apple_late_sum + jnp.where(in_late, apple_e, 0.0)
    apple_late_n = acc.apple_late_n + in_late.astype(jnp.float32)
    # First time stock hits 0; leave sentinel T if never barren.
    unseen = acc.barren_t >= T
    barren_t = jnp.where((apple_e <= 0.0) & unseen, t0_e, acc.barren_t)

    # R: mean over harvesters of per-agent episode return.
    R_e = jnp.sum(rew_sum * harvester_w, axis=-1) / jnp.maximum(n_h, 1.0)

    # S: mean over harvesters (with any apples) of mean positive-reward timestep.
    valid = (env_pos_count > 0) & (harvester_w[None, :] > 0.0)
    s_i = jnp.where(valid, env_pos_time / jnp.maximum(env_pos_count, 1.0), 0.0)
    n_valid = jnp.sum(valid.astype(jnp.float32), axis=-1)
    S_e = jnp.where(n_valid > 0, jnp.sum(s_i, axis=-1) / jnp.maximum(n_valid, 1.0), 0.0)

    # Peace (P): agent-beam / peer timeouts only (info["agent_tagged"]).
    # system_peace (P_sys): reputation-enforcement + police timeouts
    #   (caller ORs police into ``enforcement_tagged``).
    # police_peace (P_police): police-beam timeouts only (breakdown).
    # Expected: P ≈ N_h * (1 - tagged_out_frac); P_sys ≈ N_h * (1 - enf_tagged_out_frac).
    ep_len = jnp.maximum(inner_t[..., 0].astype(jnp.float32), 1.0)
    P_e = n_h - jnp.sum(tagged_sum * harvester_w, axis=-1) / ep_len
    P_sys_e = n_h - jnp.sum(enforcement_tagged_sum * harvester_w, axis=-1) / ep_len
    P_police_e = n_h - jnp.sum(police_tagged_sum * harvester_w, axis=-1) / ep_len

    def _gini_metrics(_):
        v_env = env_ret_sum * harvester_w[None, :]
        v_tot = rew_sum * harvester_w[None, :]
        total_env = jnp.sum(v_env, axis=-1)
        pairwise = (
            jnp.abs(v_env[:, :, None] - v_env[:, None, :]) * pair_w[None, :, :]
        )
        gini_num = jnp.sum(pairwise, axis=(-1, -2))
        e_apple = jnp.where(
            total_env > 0, 1.0 - gini_num / (2.0 * n_h * total_env), 0.0
        )
        abs_sum = jnp.sum(jnp.abs(v_tot), axis=-1)
        pairwise_tot = (
            jnp.abs(v_tot[:, :, None] - v_tot[:, None, :]) * pair_w[None, :, :]
        )
        gini_num_tot = jnp.sum(pairwise_tot, axis=(-1, -2))
        g_rsv = jnp.where(
            abs_sum > 0,
            gini_num_tot / (2.0 * jnp.maximum(n_h - 1.0, 1.0) * abs_sum),
            0.0,
        )
        e_total = jnp.where(abs_sum > 0, 1.0 - g_rsv, 0.0)
        return e_apple, e_total

    E_e, E_total_e = jax.lax.cond(
        jnp.any(done_e),
        _gini_metrics,
        lambda _: (
            jnp.zeros((env_ret_sum.shape[0],), dtype=jnp.float32),
            jnp.zeros((rew_sum.shape[0],), dtype=jnp.float32),
        ),
        None,
    )

    R_sum = jnp.sum(d * R_e)
    S_sum = jnp.sum(d * S_e)
    P_sum = jnp.sum(d * P_e)
    P_sys_sum = jnp.sum(d * P_sys_e)
    P_police_sum = jnp.sum(d * P_police_e)
    E_sum = jnp.sum(d * E_e)
    E_total_sum = jnp.sum(d * E_total_e)
    count = jnp.sum(d)

    denom_rate = jnp.maximum(ep_len * n_h, 1.0)
    zap_ok_h = zap_ok_sum * harvester_w
    zap_try_h = zap_try_sum * harvester_w
    zap_success_rate_e = jnp.sum(zap_ok_h, axis=-1) / denom_rate
    zap_attempt_rate_e = jnp.sum(zap_try_h, axis=-1) / denom_rate
    zap_hit_rate_e = jnp.sum(zap_ok_h, axis=-1) / jnp.maximum(
        jnp.sum(zap_try_h, axis=-1), 1.0
    )
    zap_success_rate_sum = jnp.sum(d * zap_success_rate_e)
    zap_attempt_rate_sum = jnp.sum(d * zap_attempt_rate_e)
    zap_hit_rate_sum = jnp.sum(d * zap_hit_rate_e)

    enforcement_hits_total_e = jnp.sum(enforcement_hit_sum * harvester_w, axis=-1)
    enforcement_rep_total_e = jnp.sum(enforcement_rep_sum * harvester_w, axis=-1)
    enforcement_hit_rate_e = enforcement_hits_total_e / denom_rate
    enforcement_hit_rate_sum = jnp.sum(d * enforcement_hit_rate_e)
    enforcement_hit_count_sum = jnp.sum(d * enforcement_hits_total_e)
    enforcement_rep_total_sum = jnp.sum(d * enforcement_rep_total_e)

    # Hit-weighted mean reputation of zap victims (harvester shooters only).
    zap_tagged_hits_total_e = jnp.sum(zap_target_hit_sum * harvester_w, axis=-1)
    zap_tagged_rep_total_e = jnp.sum(zap_target_rep_sum * harvester_w, axis=-1)
    zap_tagged_hit_count_sum = jnp.sum(d * zap_tagged_hits_total_e)
    zap_tagged_rep_total_sum = jnp.sum(d * zap_tagged_rep_total_e)
    # Police shooters → separate enforcement/* series.
    police_zap_tagged_hits_total_e = jnp.sum(zap_target_hit_sum * police_w, axis=-1)
    police_zap_tagged_rep_total_e = jnp.sum(zap_target_rep_sum * police_w, axis=-1)
    police_zap_tagged_hit_count_sum = jnp.sum(d * police_zap_tagged_hits_total_e)
    police_zap_tagged_rep_total_sum = jnp.sum(d * police_zap_tagged_rep_total_e)

    late_mean_e = apple_late_sum / jnp.maximum(apple_late_n, 1.0)
    apple_end_sum = jnp.sum(d * apple_e)
    apple_late_mean_sum = jnp.sum(d * late_mean_e)
    barren_t_sum = jnp.sum(d * barren_t)
    barren_event_sum = jnp.sum(d * (barren_t < T).astype(jnp.float32))

    # tagged_out_frac: agent-beam timeout occupancy (pairs with peace).
    # enforcement_tagged_out_frac: enforcement occupancy (pairs with system_peace).
    tagged_agent_steps = jnp.sum(tagged_sum * harvester_w, axis=-1)
    tagged_out_frac_e = tagged_agent_steps / jnp.maximum(ep_len * n_h, 1.0)
    tagged_out_frac_sum = jnp.sum(d * tagged_out_frac_e)
    enf_tagged_agent_steps = jnp.sum(enforcement_tagged_sum * harvester_w, axis=-1)
    enforcement_tagged_out_frac_e = enf_tagged_agent_steps / jnp.maximum(
        ep_len * n_h, 1.0
    )
    enforcement_tagged_out_frac_sum = jnp.sum(d * enforcement_tagged_out_frac_e)
    police_tagged_agent_steps = jnp.sum(police_tagged_sum * harvester_w, axis=-1)
    police_tagged_out_frac_e = police_tagged_agent_steps / jnp.maximum(
        ep_len * n_h, 1.0
    )
    police_tagged_out_frac_sum = jnp.sum(d * police_tagged_out_frac_e)

    # Police zap rates: denominator uses police count only (0 if no police).
    denom_police = jnp.maximum(ep_len * jnp.maximum(n_police, 1.0), 1.0)
    police_zap_try_p = police_zap_try_sum * police_w
    police_zap_ok_p = police_zap_ok_sum * police_w
    police_zap_attempt_rate_e = jnp.where(
        n_police > 0.0,
        jnp.sum(police_zap_try_p, axis=-1) / denom_police,
        jnp.float32(0.0),
    )
    police_zap_hit_rate_e = jnp.where(
        n_police > 0.0,
        jnp.sum(police_zap_ok_p, axis=-1)
        / jnp.maximum(jnp.sum(police_zap_try_p, axis=-1), 1.0),
        jnp.float32(0.0),
    )
    police_zap_attempt_rate_sum = jnp.sum(d * police_zap_attempt_rate_e)
    police_zap_hit_rate_sum = jnp.sum(d * police_zap_hit_rate_e)
    police_reward_ep_e = jnp.sum(police_reward_sum * police_w, axis=-1)
    police_reward_ep_sum = jnp.sum(d * police_reward_ep_e)

    # Population variance of per-agent apple returns (same quantity Equality uses).
    env_h = env_ret_sum * harvester_w[None, :]
    env_mean = jnp.sum(env_h, axis=-1, keepdims=True) / jnp.maximum(n_h, 1.0)
    rew_var_e = jnp.sum(
        jnp.square(env_h - env_mean) * harvester_w[None, :], axis=-1
    ) / jnp.maximum(n_h, 1.0)
    reward_variance_across_agents_sum = jnp.sum(d * rew_var_e)

    rep_mean_e = rep_sum / jnp.maximum(rep_n, 1.0)
    has_rep = rep_n > 0.0
    rep_min_e = jnp.where(has_rep, rep_min, 0.0)
    rep_max_e = jnp.where(has_rep, rep_max, 0.0)
    harvest_rate_mean_e = harvest_rate_sum / jnp.maximum(harvest_rate_n, 1.0)
    has_hr = harvest_rate_n > 0.0
    harvest_rate_max_e = jnp.where(has_hr, harvest_rate_max, 0.0)
    burst_count_mean_e = burst_count_sum / jnp.maximum(burst_count_n, 1.0)
    has_bc = burst_count_n > 0.0
    burst_count_max_e = jnp.where(has_bc, burst_count_max, 0.0)
    burst_penalty_mean_e = burst_penalty_sum / jnp.maximum(burst_penalty_n, 1.0)
    has_bp = burst_penalty_n > 0.0
    burst_penalty_max_e = jnp.where(has_bp, burst_penalty_max, 0.0)
    zap_target_rep_mean_e = zap_target_rep_sum / jnp.maximum(zap_target_hit_sum, 1.0)
    target_bad_rate_attempted_e = target_bad_attempt_num / jnp.maximum(
        target_bad_attempt_den, 1.0
    )
    target_bad_rate_hit_e = target_bad_hit_num / jnp.maximum(target_bad_hit_den, 1.0)
    base_rate_bad_e = base_rate_bad_sum / jnp.maximum(base_rate_bad_n, 1.0)
    enforcement_rep_mean_e = enforcement_rep_sum / jnp.maximum(enforcement_hit_sum, 1.0)

    agent_rep_min_sum = jnp.sum(jnp.where(done_e[:, None], rep_min_e, 0.0), axis=0)
    agent_rep_max_sum = jnp.sum(jnp.where(done_e[:, None], rep_max_e, 0.0), axis=0)
    agent_rep_mean_sum = jnp.sum(jnp.where(done_e[:, None], rep_mean_e, 0.0), axis=0)
    agent_zap_ok_ep_sum = jnp.sum(jnp.where(done_e[:, None], zap_ok_sum, 0.0), axis=0)
    agent_zap_try_ep_sum = jnp.sum(jnp.where(done_e[:, None], zap_try_sum, 0.0), axis=0)
    agent_zap_targets_hit_ep_sum = jnp.sum(
        jnp.where(done_e[:, None], zap_target_hit_sum, 0.0), axis=0
    )
    agent_tags_received_ep_sum = jnp.sum(
        jnp.where(done_e[:, None], tags_received_sum, 0.0), axis=0
    )
    agent_harvest_rate_mean_sum = jnp.sum(
        jnp.where(done_e[:, None], harvest_rate_mean_e, 0.0), axis=0
    )
    agent_harvest_rate_max_sum = jnp.sum(
        jnp.where(done_e[:, None], harvest_rate_max_e, 0.0), axis=0
    )
    agent_burst_count_mean_sum = jnp.sum(
        jnp.where(done_e[:, None], burst_count_mean_e, 0.0), axis=0
    )
    agent_burst_count_max_sum = jnp.sum(
        jnp.where(done_e[:, None], burst_count_max_e, 0.0), axis=0
    )
    agent_burst_penalty_mean_sum = jnp.sum(
        jnp.where(done_e[:, None], burst_penalty_mean_e, 0.0), axis=0
    )
    agent_burst_penalty_max_sum = jnp.sum(
        jnp.where(done_e[:, None], burst_penalty_max_e, 0.0), axis=0
    )
    agent_zap_target_rep_mean_sum = jnp.sum(
        jnp.where(done_e[:, None], zap_target_rep_mean_e, 0.0), axis=0
    )
    agent_target_bad_rate_attempted_ep_sum = jnp.sum(
        jnp.where(done_e[:, None], target_bad_rate_attempted_e, 0.0), axis=0
    )
    agent_target_bad_rate_hit_ep_sum = jnp.sum(
        jnp.where(done_e[:, None], target_bad_rate_hit_e, 0.0), axis=0
    )
    agent_peer_beam_hits_good_ep_sum = jnp.sum(
        jnp.where(done_e[:, None], target_good_attempt_num, 0.0), axis=0
    )
    agent_peer_beam_hits_bad_ep_sum = jnp.sum(
        jnp.where(done_e[:, None], target_bad_attempt_num, 0.0), axis=0
    )
    base_rate_bad_ep_sum = jnp.sum(d * base_rate_bad_e)
    peer_zap_timeouts_ep_sum = jnp.sum(d * peer_zap_timeout_sum)
    enforcement_timeouts_ep_sum = jnp.sum(d * enf_timeout_sum)
    enf_ttf_e = enf_time_to_freeze_sum / jnp.maximum(enf_time_to_freeze_n, 1.0)
    enforcement_time_to_freeze_ep_sum = jnp.sum(d * enf_ttf_e)
    bad_surv_e = bad_survival_sum / jnp.maximum(bad_survival_n, 1.0)
    bad_survival_ep_sum = jnp.sum(d * bad_surv_e)
    good_surv_e = good_survival_sum / jnp.maximum(good_survival_n, 1.0)
    good_survival_ep_sum = jnp.sum(d * good_surv_e)
    bandit_zap_count_ep_sum = jnp.sum(d * bandit_zap_count_sum)
    zap_try_ep = jnp.sum(zap_try_sum * harvester_w[None, :], axis=-1)
    bandit_zap_rate_e = bandit_zap_count_sum / jnp.maximum(zap_try_ep, 1.0)
    bandit_zap_rate_ep_sum = jnp.sum(d * bandit_zap_rate_e)
    agent_enforcement_hits_ep_sum = jnp.sum(
        jnp.where(done_e[:, None], enforcement_hit_sum, 0.0), axis=0
    )
    agent_enforcement_rep_mean_sum = jnp.sum(
        jnp.where(done_e[:, None], enforcement_rep_mean_e, 0.0), axis=0
    )
    agent_return_ep_sum = jnp.sum(
        jnp.where(done_e[:, None], rew_sum, 0.0), axis=0
    )

    keep = jnp.logical_not(done_e)[:, None]
    keep_e = jnp.logical_not(done_e)
    inf = jnp.float32(jnp.inf)
    acc = acc.replace(
        rew_sum=jnp.where(keep, rew_sum, 0.0),
        env_ret_sum=jnp.where(keep, env_ret_sum, 0.0),
        env_pos_count=jnp.where(keep, env_pos_count, 0.0),
        env_pos_time=jnp.where(keep, env_pos_time, 0.0),
        tagged_sum=jnp.where(keep, tagged_sum, 0.0),
        enforcement_tagged_sum=jnp.where(keep, enforcement_tagged_sum, 0.0),
        police_tagged_sum=jnp.where(keep, police_tagged_sum, 0.0),
        zap_ok_sum=jnp.where(keep, zap_ok_sum, 0.0),
        zap_try_sum=jnp.where(keep, zap_try_sum, 0.0),
        police_zap_ok_sum=jnp.where(keep, police_zap_ok_sum, 0.0),
        police_zap_try_sum=jnp.where(keep, police_zap_try_sum, 0.0),
        police_reward_sum=jnp.where(keep, police_reward_sum, 0.0),
        rep_min=jnp.where(keep, rep_min, inf),
        rep_max=jnp.where(keep, rep_max, -inf),
        rep_sum=jnp.where(keep, rep_sum, 0.0),
        rep_n=jnp.where(keep, rep_n, 0.0),
        tags_received_sum=jnp.where(keep, tags_received_sum, 0.0),
        zap_target_rep_sum=jnp.where(keep, zap_target_rep_sum, 0.0),
        zap_target_hit_sum=jnp.where(keep, zap_target_hit_sum, 0.0),
        target_bad_attempt_num=jnp.where(keep, target_bad_attempt_num, 0.0),
        target_bad_attempt_den=jnp.where(keep, target_bad_attempt_den, 0.0),
        target_good_attempt_num=jnp.where(keep, target_good_attempt_num, 0.0),
        target_bad_hit_num=jnp.where(keep, target_bad_hit_num, 0.0),
        target_bad_hit_den=jnp.where(keep, target_bad_hit_den, 0.0),
        base_rate_bad_sum=jnp.where(keep_e, base_rate_bad_sum, 0.0),
        base_rate_bad_n=jnp.where(keep_e, base_rate_bad_n, 0.0),
        peer_zap_timeout_sum=jnp.where(keep_e, peer_zap_timeout_sum, 0.0),
        enf_timeout_sum=jnp.where(keep_e, enf_timeout_sum, 0.0),
        enf_time_to_freeze_sum=jnp.where(keep_e, enf_time_to_freeze_sum, 0.0),
        enf_time_to_freeze_n=jnp.where(keep_e, enf_time_to_freeze_n, 0.0),
        bad_survival_sum=jnp.where(keep_e, bad_survival_sum, 0.0),
        bad_survival_n=jnp.where(keep_e, bad_survival_n, 0.0),
        good_survival_sum=jnp.where(keep_e, good_survival_sum, 0.0),
        good_survival_n=jnp.where(keep_e, good_survival_n, 0.0),
        bandit_zap_count_sum=jnp.where(keep_e, bandit_zap_count_sum, 0.0),
        enforcement_hit_sum=jnp.where(keep, enforcement_hit_sum, 0.0),
        enforcement_rep_sum=jnp.where(keep, enforcement_rep_sum, 0.0),
        harvest_rate_min=jnp.where(keep, harvest_rate_min, inf),
        harvest_rate_max=jnp.where(keep, harvest_rate_max, -inf),
        harvest_rate_sum=jnp.where(keep, harvest_rate_sum, 0.0),
        harvest_rate_n=jnp.where(keep, harvest_rate_n, 0.0),
        burst_count_min=jnp.where(keep, burst_count_min, inf),
        burst_count_max=jnp.where(keep, burst_count_max, -inf),
        burst_count_sum=jnp.where(keep, burst_count_sum, 0.0),
        burst_count_n=jnp.where(keep, burst_count_n, 0.0),
        burst_penalty_min=jnp.where(keep, burst_penalty_min, inf),
        burst_penalty_max=jnp.where(keep, burst_penalty_max, -inf),
        burst_penalty_sum=jnp.where(keep, burst_penalty_sum, 0.0),
        burst_penalty_n=jnp.where(keep, burst_penalty_n, 0.0),
        apple_late_sum=jnp.where(keep_e, apple_late_sum, 0.0),
        apple_late_n=jnp.where(keep_e, apple_late_n, 0.0),
        barren_t=jnp.where(keep_e, barren_t, T),
    )
    return (
        acc,
        R_sum,
        S_sum,
        P_sum,
        E_sum,
        E_total_sum,
        count,
        zap_success_rate_sum,
        zap_attempt_rate_sum,
        zap_hit_rate_sum,
        apple_end_sum,
        apple_late_mean_sum,
        barren_t_sum,
        barren_event_sum,
        tagged_out_frac_sum,
        reward_variance_across_agents_sum,
        agent_rep_min_sum,
        agent_rep_max_sum,
        agent_rep_mean_sum,
        agent_zap_ok_ep_sum,
        agent_zap_try_ep_sum,
        agent_zap_targets_hit_ep_sum,
        agent_tags_received_ep_sum,
        agent_harvest_rate_mean_sum,
        agent_harvest_rate_max_sum,
        agent_burst_count_mean_sum,
        agent_burst_count_max_sum,
        agent_burst_penalty_mean_sum,
        agent_burst_penalty_max_sum,
        agent_zap_target_rep_mean_sum,
        agent_enforcement_hits_ep_sum,
        agent_enforcement_rep_mean_sum,
        agent_return_ep_sum,
        enforcement_hit_rate_sum,
        enforcement_hit_count_sum,
        enforcement_rep_total_sum,
        zap_tagged_hit_count_sum,
        zap_tagged_rep_total_sum,
        police_zap_tagged_hit_count_sum,
        police_zap_tagged_rep_total_sum,
        P_sys_sum,
        enforcement_tagged_out_frac_sum,
        P_police_sum,
        police_tagged_out_frac_sum,
        police_zap_attempt_rate_sum,
        police_zap_hit_rate_sum,
        police_reward_ep_sum,
        agent_target_bad_rate_attempted_ep_sum,
        agent_target_bad_rate_hit_ep_sum,
        base_rate_bad_ep_sum,
        peer_zap_timeouts_ep_sum,
        enforcement_timeouts_ep_sum,
        enforcement_time_to_freeze_ep_sum,
        bad_survival_ep_sum,
        good_survival_ep_sum,
        bandit_zap_count_ep_sum,
        bandit_zap_rate_ep_sum,
        agent_peer_beam_hits_good_ep_sum,
        agent_peer_beam_hits_bad_ep_sum,
    )


def _stack_agent_dim(tensor_dict, agents):
    return jnp.stack([tensor_dict[str(agent)] for agent in agents], axis=1)


def _info_field(info: dict, key: str, fallback_shape, dtype=jnp.float32):
    """Pull a batched info field; fall back to zeros if missing."""
    if key in info:
        return jnp.asarray(info[key])
    return jnp.zeros(fallback_shape, dtype=dtype)


def _as_seed_vec(x, num_seeds: int) -> onp.ndarray:
    """Device/host value â†’ float64 vector of length ``num_seeds``."""
    arr = onp.asarray(jax.device_get(x), dtype=onp.float64).reshape(-1)
    if arr.size == 1 and num_seeds > 1:
        return onp.full((num_seeds,), float(arr[0]), dtype=onp.float64)
    if arr.size == num_seeds:
        return arr
    if arr.size == 1:
        return arr
    raise ValueError(f"expected {num_seeds} seed values, got shape {arr.shape}")


def _interval_mean_vec(
    cur, prev: onp.ndarray, n_since: onp.ndarray
) -> onp.ndarray:
    """Per-seed interval mean; NaN where no episodes completed."""
    cur_v = onp.asarray(cur, dtype=onp.float64).reshape(-1)
    out = onp.full_like(cur_v, onp.nan, dtype=onp.float64)
    mask = n_since > 0
    out[mask] = (cur_v[mask] - prev[mask]) / n_since[mask]
    return out


def metrics_from_rollout(
    step_metrics: StepMetrics,
    num_agents: int,
    *,
    R: float = float("nan"),
    S: float = float("nan"),
    P: float = float("nan"),
    E: float = float("nan"),
    E_total: float = float("nan"),
    mean_episode_return: float = float("nan"),
    zap_success_rate: float | None = None,
    zap_attempt_rate: float | None = None,
    zap_hit_rate: float | None = None,
    n_episodes_since_log: int = 0,
    n_episodes_lifetime: int = 0,
) -> dict[str, float]:
    """Assemble the logged metric dict.

    R / S / P / E, ``mean_episode_return``, and any explicit ``zap_*`` arguments
    are **full-episode** quantities
    (Lupu & Precup Â§4.2) accumulated on-device over the ``T``-step episode by
    :func:`update_per_episode_acc` and passed in here as interval means â€” they
    are *not* derived from the ``NUM_STEPS`` rollout window. If the explicit
    action-rate args are omitted, this helper falls back to window statistics
    for offline analysis / legacy callers. ``mean_step_reward`` remains a
    window statistic.

    Note: for the individual-reward Harvest setup, the paper's collective-average
    return R equals the mean-over-agents episode return, so ``R`` and
    ``mean_episode_return`` are the same number (both kept for compatibility).
    """
    tot_r = onp.asarray(step_metrics.total_rewards)
    zap_ok = onp.asarray(step_metrics.zap_successful)
    zap_try = onp.asarray(step_metrics.zap_attempted)
    tagged = onp.asarray(step_metrics.tagged)

    def _mean_agents(arr: onp.ndarray) -> float:
        return float(onp.mean(arr))

    def _sum_agents(arr: onp.ndarray) -> float:
        return float(onp.sum(arr))

    if zap_success_rate is None:
        zap_success_rate = _mean_agents(zap_ok)
    if zap_attempt_rate is None:
        zap_attempt_rate = _mean_agents(zap_try)
    if zap_hit_rate is None:
        zap_hit_rate = _sum_agents(zap_ok) / max(_sum_agents(zap_try), 1)

    out = {
        "R": float(R),
        "S": float(S),
        "P": float(P),
        # E is 1 âˆ’ Gini over per-agent APPLE (environmental) returns only.
        # `E_apple_return` is an explicit alias so logs are unambiguous; `E` is
        # kept as the plotted key.
        "E": float(E),
        "E_apple_return": float(E),
        # E_total_return: 1 âˆ’ Raffinetti-Siletti-Vernizzi Gini over per-agent
        # TOTAL returns (may be negative when rewards are signed). Differs from
        # E by the N/(Nâˆ’1) RSV constant.
        "E_total_return": float(E_total),
        "zap_success_rate": float(zap_success_rate),
        "zap_attempt_rate": float(zap_attempt_rate),
        "zap_hit_rate": float(zap_hit_rate),
        "inactive_frac": _mean_agents(tagged),
        # Mean return of inner-episodes completed since the previous log point.
        "mean_episode_return": float(mean_episode_return),
        # Completions since previous log (for density / debugging).
        "n_episodes_completed": int(n_episodes_since_log),
        # Completions since training start â€” this is what stdout ``n_ep`` shows.
        "n_episodes_lifetime": int(n_episodes_lifetime),
        "mean_step_reward": _mean_agents(tot_r),
    }
    return out


