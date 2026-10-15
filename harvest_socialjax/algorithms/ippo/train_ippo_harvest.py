"""
Independent PPO (IPPO) for Harvest Open (Perolat et al. / SocialJax).

Separate actor-critic (params + optimizer) per agent, keyed by agent id.
Supports optional spatial reputation (ReputationWrapper, ch15/ch16) and
baseline team-sum equality shaping (OPTIMIZE_BASELINE / OPTIMIZE_SHARED_REWARDS).

AGENT_OVERRIDES (scaffolding): optional per-agent lr / entropy_coef / clip_eps.

Hydra config, CTRolloutManager + LogWrapper, jax.lax.scan rollouts,
W&B logging, checkpoints, R/S/P/E metrics (aggregated and per-agent when
LOG_PER_AGENT or ENABLE_REPUTATION).

Multi-seed (NUM_SEEDS): outer jax.vmap over seeds; W&B logs metric/seed_i.

UPDATES_PER_JIT (default LOG_EVERY): batched PPO updates on-device before host sync.

JAX backend: set JAX_PLATFORMS=cpu on CPU-only nodes; leave unset on GPU.

VERBOSE_LOGGING (default False): DEBUG_PERF prints and extra loss metrics.
LOG_PER_AGENT (default False): per-agent episode metrics (agent_i/*_ep) when
reputation is on; legacy window breakdowns when reputation is off.
"""
from __future__ import annotations

import os
import pickle
import time
import warnings

_backend = os.environ.get("JAX_PLATFORMS", "auto (gpu if available, else cpu)")
print(f"[JAX] Requested backend: {_backend}")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import harvest_socialjax.jax_compat  # noqa: F401

import logging

# Quiet JAX's verbose tracing/compilation WARNING spam (dispatch / pxla).
logging.getLogger("jax._src.dispatch").setLevel(logging.ERROR)
logging.getLogger("jax._src.interpreters.pxla").setLevel(logging.ERROR)

import chex
import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import socialjax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from harvest_socialjax import register_harvest_env
from harvest_socialjax.algorithms.ippo.episode_metrics import (
    CNN,
    StepMetrics,
    _as_seed_vec,
    _info_field,
    _interval_mean_vec,
    _stack_agent_dim,
    EpisodeAggregates,
    add_episode_aggregates,
    init_episode_aggregates,
    init_per_episode_acc,
    metrics_from_rollout,
    update_apple_count_step,
    update_per_episode_acc,
)
from harvest_socialjax.algorithms.ippo.wandb_logging import (
    finish_wandb_logger,
    init_wandb_logger,
    log_training_metrics,
)
from harvest_socialjax.algorithms.ippo.seed_log import (
    PER_SEED_TOC_METRICS as _PER_SEED_TOC_METRICS,
    pack_per_seed_series,
)
from harvest_socialjax.perf_instrumentation import (
    ShapeSignatureTracker,
    log_compiles_first_update,
    make_trace_counter,
    print_jax_device_sanity,
)
from harvest_socialjax.algorithms.render_gif import (
    GifRenderer,
    gif_update_for_chunk,
    should_render_gif,
)
from harvest_socialjax.envs.reputation import (
    ReputationConfig,
    clamp_enforcement_strictness,
)
from harvest_socialjax.wrappers import ReputationV2Wrapper, ReputationWrapper
from harvest_socialjax.wrappers.reputation_v2_wrapper import (
    seed_schedule_episode_clock,
    wandb_ep_to_per_env_completed,
)
from harvest_socialjax.envs.police import (
    apply_police_agent_settings,
    police_agent_indices,
    resolve_police_mask_from_env,
)
from socialjax.wrappers.baselines import CTRolloutManager, LogWrapper


def spatial_reputation_enabled(config: dict) -> bool:
    """True when spatial rep channels + ReputationWrapper are active.

    ``ENABLE_SPATIAL_REPUTATION`` overrides ``ENABLE_REPUTATION`` when set.
    """
    if "ENABLE_SPATIAL_REPUTATION" in config:
        return bool(config["ENABLE_SPATIAL_REPUTATION"])
    return bool(config.get("ENABLE_REPUTATION", False))


def reputation_v2_enabled(config: dict) -> bool:
    """True when burst-window enforcement TRIGGER (ReputationV2Wrapper) is on."""
    return bool(config.get("ENABLE_REPUTATION_V2", False))


def assert_reputation_modes_exclusive(config: dict) -> None:
    """Raise if classic reputation and v2 trigger are both enabled."""
    if spatial_reputation_enabled(config) and reputation_v2_enabled(config):
        raise ValueError(
            "ENABLE_REPUTATION and ENABLE_REPUTATION_V2 are mutually exclusive; "
            "enable at most one."
        )


def reputation_v2_config_from_dict(config: dict):
    """Build ``BurstWindowConfig`` from Hydra ``REPUTATION_V2`` block."""
    from harvest_socialjax.envs.harvest_open import TAG_TIMEOUT_STEPS
    from harvest_socialjax.envs.reputation_v2 import BurstWindowConfig

    block = dict(config.get("REPUTATION_V2") or {})
    kwargs: dict = {}
    int_keys = {
        "window_len",
        "max_harvests_in_window",
        "timeout_duration",
        "timeout_delay",
        "enforcement_delay",
        "enforcement_period",
        "delay_ramp_every_episodes",
        "delay_ramp_max_delay",
        "enforcement_stop_after_episodes",
        "schedule_env_multiplier",
        "graded_max_reputation",
        "delay_hold_episodes",
        "delay_hold_start_delay",
        "delay_ramp_end_episodes",
        "prob_enforcement_decay_episodes",
        "prob_enforcement_decay2_episodes",
        "peer_punish_anneal_episodes",
        "peer_punish_intro_episodes",
        "peer_punish_anneal_start_episodes",
        "peer_punish_anneal_end_episodes",
    }
    bool_keys = {
        "last_apple_timeout",
        "reset_label_on_respawn",
        "immediate_timeout",
        "enforcement_enabled",
        "delay_ramp_enabled",
        "delay_ramp_start_immediate",
        "delay_curriculum_end_immediate",
        "delay_curriculum_end_disable_enforcement",
        "graded_reputation",
        "peer_punish_reward_enabled",
        "peer_punish_anneal",
        "only_good_can_zap",
        "peer_focus_on_bad",
        "mask_reputation_obs",
        "zap_affects_label",
        "zap_to_bad_only_pure_good",
        "prob_enforcement_enabled",
        "prob_enforcement_decay_enabled",
    }
    float_keys = {
        "peer_punish_bad_reward",
        "peer_punish_good_penalty",
        "peer_punish_bad_reward_end",
        "peer_punish_good_penalty_end",
        "prob_enforcement_strictness",
        "prob_enforcement_strictness_start",
        "prob_enforcement_strictness_end",
        "prob_enforcement_strictness_final",
    }
    for k, v in block.items():
        if k in int_keys:
            kwargs[k] = int(v)
        elif k in bool_keys:
            kwargs[k] = bool(v)
        elif k in float_keys:
            kwargs[k] = float(v)
        elif k == "enforce_mode":
            kwargs[k] = str(v)
        else:
            kwargs[k] = v
    if "timeout_duration" not in kwargs:
        kwargs["timeout_duration"] = int(TAG_TIMEOUT_STEPS)
    # Align episode schedules with W&B episode_count (sum across parallel envs).
    if int(kwargs.get("schedule_env_multiplier", 1) or 1) <= 1:
        kwargs["schedule_env_multiplier"] = max(1, int(config.get("NUM_ENVS", 1)))
    return BurstWindowConfig(**kwargs)


def snapshot_device_memory() -> dict:
    """Read JAX device allocator stats. GPU-only; CPU has no ``memory_stats``.

    ``bytes_in_use`` is live allocations *right now*.
    ``peak_bytes_in_use`` is the allocator high-water mark since this process
    created the client — it is typically **not** resettable, so after the
    compile phase it already includes compile scratch. After steady-state it
    is max(compile_peak, anything allocated later), not a fresh steady peak.
    """
    devices = jax.local_devices()
    if not devices:
        return {"error": "no jax.local_devices()"}
    dev = devices[0]
    getter = getattr(dev, "memory_stats", None)
    if getter is None:
        return {
            "error": "memory_stats unavailable (CPU backend or old jaxlib)",
            "device": str(dev),
            "platform": getattr(dev, "platform", ""),
        }
    raw = getter() or {}
    out: dict = {
        "device": str(dev),
        "platform": getattr(dev, "platform", ""),
    }
    for key, val in raw.items():
        try:
            out[key] = int(val)
        except (TypeError, ValueError):
            out[key] = val
    return out


def _print_memory_snapshot(phase: str, stats: dict) -> None:
    err = stats.get("error")
    if err:
        print(f"[memory] {phase}: UNAVAILABLE ({err}) device={stats.get('device', '?')}")
        return
    used = stats.get("bytes_in_use")
    peak = stats.get("peak_bytes_in_use")
    print(
        f"[memory] {phase}: device={stats.get('device')} "
        f"bytes_in_use={used} ({_bytes_to_mib(used):.1f} MiB) "
        f"peak_bytes_in_use={peak} ({_bytes_to_mib(peak):.1f} MiB)"
    )


def _bytes_to_mib(n) -> float:
    if n is None:
        return float("nan")
    return float(n) / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Heterogeneity scaffolding (structure only — not full hetero experiments yet)
# ---------------------------------------------------------------------------
# Future plugs belong here / in resolve_agent_hparams:
#   - different observation preprocessing per agent
#   - different action sets / masks per agent
#   - different reward shaping per agent
#   - different network architectures per agent
# Currently only lr / entropy_coef / clip_eps may differ via AGENT_OVERRIDES.
# ---------------------------------------------------------------------------

_OVERRIDE_LR_KEYS = ("lr", "LR", "learning_rate")
_OVERRIDE_ENT_KEYS = ("entropy_coef", "ENT_COEF", "ent_coef")
_OVERRIDE_CLIP_KEYS = ("clip_eps", "CLIP_EPS", "clip_epsilon")


def _override_lookup(overrides: dict, agent) -> dict | None:
    """Match AGENT_OVERRIDES keys: agent_0, '0', 0, etc."""
    if not overrides:
        return None
    i = int(agent)
    candidates = (f"agent_{i}", str(i), i, agent)
    for c in candidates:
        if c in overrides:
            return dict(overrides[c])
        # OmegaConf may stringify int keys
        if str(c) in overrides:
            return dict(overrides[str(c)])
    return None


def _pick_override(ov: dict, keys: tuple[str, ...], default: float) -> float:
    for k in keys:
        if k in ov and ov[k] is not None:
            return float(ov[k])
    return float(default)


def resolve_agent_hparams(config: dict, agents) -> dict[str, dict[str, float]]:
    """Per-agent lr / entropy_coef / clip_eps; global defaults if no overrides."""
    raw = config.get("AGENT_OVERRIDES")
    if raw is None:
        overrides = {}
    elif isinstance(raw, dict):
        overrides = dict(raw)
    else:
        overrides = OmegaConf.to_container(raw, resolve=True) or {}

    global_lr = float(config["LR"])
    global_ent = float(config["ENT_COEF"])
    global_clip = float(config["CLIP_EPS"])

    resolved: dict[str, dict[str, float]] = {}
    for agent in agents:
        key = str(agent)
        ov = _override_lookup(overrides, agent) or {}
        resolved[key] = {
            "lr": _pick_override(ov, _OVERRIDE_LR_KEYS, global_lr),
            "entropy_coef": _pick_override(ov, _OVERRIDE_ENT_KEYS, global_ent),
            "clip_eps": _pick_override(ov, _OVERRIDE_CLIP_KEYS, global_clip),
        }
    return resolved


def _hparams_tuple(h: dict[str, float]) -> tuple[float, float, float]:
    return (float(h["lr"]), float(h["entropy_coef"]), float(h["clip_eps"]))


def warn_if_overrides_collapse(agent_hparams: dict[str, dict[str, float]]) -> None:
    """Warn when AGENT_OVERRIDES was set but ≥2 agents share identical resolved HPs."""
    keys = list(agent_hparams.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if _hparams_tuple(agent_hparams[keys[i]]) == _hparams_tuple(
                agent_hparams[keys[j]]
            ):
                print(
                    f"[ippo] WARNING: AGENT_OVERRIDES set but agent_{keys[i]} and "
                    f"agent_{keys[j]} have identical resolved hparams "
                    f"{agent_hparams[keys[i]]} — override may not be wired correctly."
                )


def _read_optimizer_lr(train_state: TrainState) -> float | None:
    """Best-effort read of Adam learning_rate from inject_hyperparams state."""
    opt_state = train_state.opt_state
    # optax.chain(clip, inject_hyperparams(adam)) → (clip_state, InjectHyperparamsState)
    if not isinstance(opt_state, tuple) or len(opt_state) < 2:
        return None
    adam_state = opt_state[1]
    hyperparams = getattr(adam_state, "hyperparams", None)
    if hyperparams is None or "learning_rate" not in hyperparams:
        return None
    lr = hyperparams["learning_rate"]
    # Constant float (or 0-d array); schedules are callables — skip those.
    if callable(lr):
        return None
    return float(lr)


def _set_chain_lrs(opt_state, lrs):
    """Write Adam ``learning_rate`` into ``optax.chain(clip, inject_hyperparams(adam))``.

    ``lrs`` may be a scalar (single agent) or a leading-axis vector when
    ``opt_state`` is stacked over agents.
    """
    clip_state, adam_state = opt_state
    hyperparams = dict(adam_state.hyperparams)
    hyperparams["learning_rate"] = lrs
    return (clip_state, adam_state._replace(hyperparams=hyperparams))


def _stack_agent_pytrees(*trees):
    """Stack pytrees along a new leading agent axis."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trees)


def _unstack_train_states(
    params,
    opt_state,
    steps,
    *,
    agent_keys: list[str],
    apply_fn,
    tx,
) -> dict[str, TrainState]:
    """Split stacked params/opt_state back into a per-agent ``TrainState`` dict."""
    out: dict[str, TrainState] = {}
    for i, key in enumerate(agent_keys):
        p = jax.tree_util.tree_map(lambda x, idx=i: x[idx], params)
        o = jax.tree_util.tree_map(lambda x, idx=i: x[idx], opt_state)
        step_i = steps[i] if hasattr(steps, "shape") else steps
        out[key] = TrainState(
            step=step_i,
            apply_fn=apply_fn,
            params=p,
            tx=tx,
            opt_state=o,
        )
    return out


def assert_finite_metrics(
    env_m: dict,
    mean_loss,
    update: int,
    agents,
    *,
    n_episodes_since_log: int = -1,
) -> None:
    """Fail fast if core logged scalars are NaN/Inf.

    Full-episode R/S/P/E (logged as mean_reward / sustainability / peace /
    equality) may legitimately be NaN when no inner episode has finished in
    the logging window (``n_episodes_since_log == 0``).
    """
    episode_fields = {
        "mean_reward",
        "sustainability",
        "peace",
        "system_peace",
        "equality",
        "apple_end_sum",
        "apple_late_mean_sum",
        "barren_event_sum",
        "tagged_out_frac",
        "enforcement_tagged_out_frac",
        "reward_variance_across_agents",
        "R",
        "S",
        "P",
        "E",
        "mean_episode_return",
    }
    n_completed = int(env_m.get("n_episodes_completed", n_episodes_since_log))
    skip_episode_checks = n_episodes_since_log == 0 or n_completed == 0
    checks: list[tuple[str, float]] = [
        ("mean_loss", float(mean_loss)),
    ]
    core_keys = (
        "mean_reward",
        "sustainability",
        "peace",
        "system_peace",
        "equality",
        "apple_sum",
        "tagged_out_frac",
    )
    for key, val in env_m.items():
        base = key.split("/seed_")[0] if "/seed_" in str(key) else key
        if base in core_keys:
            try:
                checks.append((str(key), float(val)))
            except (TypeError, ValueError):
                continue
            if "/seed_" in str(key):
                episode_fields.add(str(key))
    for agent in agents:
        key = f"agent_{int(agent)}/R_window"
        if key in env_m:
            checks.append((key, float(env_m[key])))
    for name, val in checks:
        if skip_episode_checks and (
            name in episode_fields or _is_episode_agent_wandb_key(name)
        ):
            continue
        if not onp.isfinite(val):
            raise RuntimeError(
                f"[ippo] non-finite metric {name!r}={val!r} at update {update} "
                f"(n_episodes_completed={n_completed})"
            )


def _fmt_metric(val: float, fmt: str) -> str:
    """Format a scalar for logs; NaN → ``'nan'``."""
    return format(val, fmt) if val == val else "nan"


def _per_agent_env_r_from_rollout(step_metrics, agents) -> dict[str, float]:
    """Per-agent env (apple) return summed over the log window — pre-shaping competence."""
    env_r = onp.asarray(step_metrics.env_rewards)
    flat = env_r.reshape(-1, len(agents))
    sums = flat.sum(axis=0)
    return {f"agent_{int(agent)}/R": float(sums[i]) for i, agent in enumerate(agents)}


# Per-agent W&B keys. Aggregate ToC metrics use all agents.
# Keys ending in ``_window`` are rollout-window diagnostics (NUM_STEPS); ``*_ep`` are
# episode-true means over inner episodes completed since the last log.
_PER_AGENT_WANDB_SUFFIXES = (
    "/R_window",
    "/env_return_window",
    "/return_window",
    "/mean_step_reward",
    "/rep_window",
    "/raw_rep_window",
    "/zap_success_rate_window",
    "/zap_attempt_rate_window",
    "/tagged_frac_window",
    "/S_window",
)

_EPISODE_AGENT_METRIC_SUFFIXES = (
    "_ep",
    "/return_ep",
)


def _is_episode_agent_wandb_key(key: str) -> bool:
    return any(key.endswith(sfx) for sfx in _EPISODE_AGENT_METRIC_SUFFIXES)


def _is_agent_window_wandb_key(key: str) -> bool:
    """True for rollout-window / count diagnostics (not episode-true)."""
    return (
        key.endswith("_window")
        or "/window" in key
        or key.endswith("/mean_step_reward")
        or key.endswith("/zap_success_count")
        or key.endswith("/zap_attempt_count")
        or key.endswith("/tags_received_count")
        or key.endswith("/tagged_steps")
    )



def _wandb_agent_metric_keys(env_m: dict, *, enable_reputation: bool, log_per_agent: bool) -> list[str]:
    """Select agent keys for W&B.

    With reputation: episode-true ``*_ep`` only — never
    ``*_window`` rollout diagnostics. Without reputation: window breakdowns
    when ``LOG_PER_AGENT`` is set.
    """
    keys: list[str] = []
    for key in env_m:
        if not key.startswith("agent_"):
            continue
        if enable_reputation:
            if _is_agent_window_wandb_key(key):
                continue
            if _is_episode_agent_wandb_key(key):
                keys.append(key)
        elif log_per_agent or any(
            key.endswith(sfx) for sfx in _PER_AGENT_WANDB_SUFFIXES
        ):
            keys.append(key)
    return keys


def _per_agent_rollout_metrics_for_wandb(
    step_metrics: StepMetrics,
    agents,
) -> dict[str, float]:
    """Window diagnostics for each agent — plot ``agent_i/*`` on W&B separately."""
    tot_r = onp.asarray(step_metrics.total_rewards)
    env_r = onp.asarray(step_metrics.env_rewards)
    tagged = onp.asarray(step_metrics.tagged)
    zap_ok = onp.asarray(step_metrics.zap_successful)
    zap_try = onp.asarray(step_metrics.zap_attempted)
    rep = onp.asarray(step_metrics.rep) if step_metrics.rep.size else None
    raw_rep = onp.asarray(step_metrics.raw_rep) if step_metrics.raw_rep.size else None

    if tot_r.ndim == 3:
        tot_flat = tot_r.reshape(-1, tot_r.shape[-1])
        env_flat = env_r.reshape(-1, env_r.shape[-1])
        tagged_flat = tagged.reshape(-1, tagged.shape[-1])
        rep_flat = rep.reshape(-1, rep.shape[-1]) if rep is not None else None
        raw_flat = raw_rep.reshape(-1, raw_rep.shape[-1]) if raw_rep is not None else None
    else:
        tot_flat = tot_r.reshape(-1, tot_r.shape[-1])
        env_flat = env_r.reshape(-1, env_r.shape[-1])
        tagged_flat = tagged.reshape(-1, tagged.shape[-1])
        rep_flat = rep.reshape(-1, rep.shape[-1]) if rep is not None else None
        raw_flat = raw_rep.reshape(-1, raw_rep.shape[-1]) if raw_rep is not None else None

    timesteps = onp.arange(tot_flat.shape[0], dtype=onp.float64)
    out: dict[str, float] = {}
    for i, agent in enumerate(agents):
        prefix = f"agent_{int(agent)}"
        out[f"{prefix}/return_window"] = float(tot_flat[:, i].sum())
        out[f"{prefix}/env_return_window"] = float(env_flat[:, i].sum())
        # Match aggregate mean_reward/R: apple-only (exclude peer-punish / police).
        out[f"{prefix}/R_window"] = float(env_flat[:, i].sum())
        out[f"{prefix}/mean_step_reward"] = float(onp.mean(tot_r[..., i]))
        out[f"{prefix}/zap_success_rate_window"] = float(onp.mean(zap_ok[..., i]))
        out[f"{prefix}/zap_attempt_rate_window"] = float(onp.mean(zap_try[..., i]))
        out[f"{prefix}/tagged_frac_window"] = float(onp.mean(tagged_flat[:, i]))
        if rep_flat is not None:
            out[f"{prefix}/rep_window_mean"] = float(onp.mean(rep_flat[:, i]))
            out[f"{prefix}/rep_window_min"] = float(onp.min(rep_flat[:, i]))
            out[f"{prefix}/rep_window_max"] = float(onp.max(rep_flat[:, i]))
            out[f"{prefix}/rep_window"] = out[f"{prefix}/rep_window_mean"]
        if raw_flat is not None:
            out[f"{prefix}/raw_rep_window"] = float(onp.mean(raw_flat[:, i]))
        out[f"{prefix}/zap_success_count"] = float(onp.sum(zap_ok[..., i]))
        out[f"{prefix}/zap_attempt_count"] = float(onp.sum(zap_try[..., i]))
        tags_recv = getattr(step_metrics, "tags_received", None)
        if tags_recv is not None and onp.asarray(tags_recv).size:
            out[f"{prefix}/tags_received_count"] = float(
                onp.sum(onp.asarray(tags_recv)[..., i])
            )
        out[f"{prefix}/tagged_steps"] = float(onp.sum(tagged_flat[:, i]))
        pos = env_flat[:, i] > 0
        if pos.any():
            out[f"{prefix}/S_window"] = float(onp.sum(timesteps[pos]) / pos.sum())
        else:
            out[f"{prefix}/S_window"] = 0.0
    return out


def _per_agent_episode_metrics_for_wandb(
    ep_life: EpisodeAggregates,
    last_agent_sums: dict[str, onp.ndarray],
    n_episodes: float,
    agents,
    *,
    seed_i: int = 0,
) -> tuple[dict[str, float], dict[str, onp.ndarray]]:
    """Mean per-agent stats over inner episodes completed since the last log.

    Each ``*_ep`` key is the arithmetic mean of that quantity over completed
    episodes in the logging interval (not a rollout-window proxy).

    Reputation (post-step ``rep`` each env step within the episode):
      * ``rep_min_ep`` — min rep over the episode
      * ``rep_max_ep`` — max rep over the episode
      * ``rep_mean_ep`` — time-average rep over the episode

    Harvest rate (reputation EMA ``rate_ema``, apples/step estimate):
      * ``harvest_rate_mean_ep`` — time-average harvest rate
      * ``harvest_rate_max_ep`` — peak harvest rate in the episode

    Burst harvest penalty (post-step ``burst_count`` / ``burst_penalty``):
      * ``burst_count_mean_ep`` / ``burst_count_max_ep``
      * ``burst_penalty_mean_ep`` / ``burst_penalty_max_ep``

    Zapping (per agent as shooter / victim):
      * ``zap_success_ep`` — timesteps with ≥1 successful zap (``zap_successful``)
      * ``zap_targets_hit_ep`` — total agents tagged (``sum_j zap_hit[i,j]``)
      * ``zap_attempts_ep`` — zap actions attempted
      * ``tags_received_ep`` — times tagged as a zap target

    Zap target reputation (shooter view, pre-step target rep at hit time):
      * ``zap_target_rep_mean_ep`` — mean target rep over all zap hits in the
        episode: ``sum_{t,j} I{zap_hit[i,j,t]} * rep_pre[j,t] / n_hits_i``

    Bad-targeting diagnostics (attempt-time vs base rate; metrics only):
      * ``target_bad_rate_attempted_ep`` — Bad frac among beam occupants when
        this agent zaps (``zap_beam_matrix``; includes eventual misses)
      * ``target_bad_rate_hit_ep`` — Bad frac among successful tags
      * ``peer_beam_hits_good_ep`` / ``peer_beam_hits_bad_ep`` — absolute beam
        hit counts on Good / Bad (pre-focus; immunity does not drop these)
      * Population ``base_rate_bad_ep`` — mean eligible Bad fraction among
        active non-self non-police agents at harvester zap-attempt time
        (compare attempted/hit rates against this for discrimination)

    Population (hit-weighted mean reputation of tagged agents):
      * ``zap_tagged_rep_mean`` — harvester shooters only:
        ``sum hits of rep_pre[victim] / n_hits``
        (from ``zap_tagged_hit_count_sum`` / ``zap_tagged_rep_total_sum``)
      * ``enforcement/police_zap_tagged_rep_mean`` — police shooters only

    Reputation enforcement (negative-rep stochastic timeout):
      * ``enforcement_hits_ep`` — times this agent was enforcement-zapped
      * ``enforcement_rep_mean_ep`` — mean ``rep_pre`` over those hits

    Apple competence (full episode sums, then interval-mean on host):
      * ``return_ep`` — environmental (apple) return only; excludes peer-punish
        zap bonuses/penalties and police zap rewards (those stay in PPO
        ``Transition.reward`` / ``return_window``)
    """
    # No completed inner episodes this log interval → omit agent_*_ep keys.
    # Emitting NaNs here used to overwrite good W&B points at the same
    # ``episode_count`` x-value (common when UPDATES_PER_JIT=1 and
    # NUM_STEPS < num_inner_steps: every other update finishes 0 episodes).
    if float(n_episodes) <= 0.0:
        return {}, dict(last_agent_sums)

    denom = float(n_episodes)
    fields = (
        ("agent_rep_min_sum", "rep_min_ep"),
        ("agent_rep_max_sum", "rep_max_ep"),
        ("agent_rep_mean_sum", "rep_mean_ep"),
        ("agent_harvest_rate_mean_sum", "harvest_rate_mean_ep"),
        ("agent_harvest_rate_max_sum", "harvest_rate_max_ep"),
        ("agent_burst_count_mean_sum", "burst_count_mean_ep"),
        ("agent_burst_count_max_sum", "burst_count_max_ep"),
        ("agent_burst_penalty_mean_sum", "burst_penalty_mean_ep"),
        ("agent_burst_penalty_max_sum", "burst_penalty_max_ep"),
        ("agent_zap_ok_ep_sum", "zap_success_ep"),
        ("agent_zap_targets_hit_ep_sum", "zap_targets_hit_ep"),
        ("agent_zap_try_ep_sum", "zap_attempts_ep"),
        ("agent_tags_received_ep_sum", "tags_received_ep"),
        ("agent_zap_target_rep_mean_sum", "zap_target_rep_mean_ep"),
        ("agent_target_bad_rate_attempted_ep_sum", "target_bad_rate_attempted_ep"),
        ("agent_target_bad_rate_hit_ep_sum", "target_bad_rate_hit_ep"),
        ("agent_peer_beam_hits_good_ep_sum", "peer_beam_hits_good_ep"),
        ("agent_peer_beam_hits_bad_ep_sum", "peer_beam_hits_bad_ep"),
        ("agent_enforcement_hits_ep_sum", "enforcement_hits_ep"),
        ("agent_enforcement_rep_mean_sum", "enforcement_rep_mean_ep"),
        ("agent_return_ep_sum", "return_ep"),
    )
    out: dict[str, float] = {}
    updated = dict(last_agent_sums)
    for field, suffix in fields:
        cur = onp.asarray(jax.device_get(getattr(ep_life, field)), dtype=onp.float64)
        if cur.ndim == 2:
            cur = cur[seed_i]
        prev = last_agent_sums[field]
        delta = cur - prev
        updated[field] = cur.copy()
        for i, agent in enumerate(agents):
            out[f"agent_{int(agent)}/{suffix}"] = float(delta[i] / denom)
    return out, updated


def _harvester_mean_rep_from_ep_metrics(
    env_m: dict,
    agents,
    police_mask,
) -> float:
    """Mean over harvesters of ``agent_i/rep_mean_ep`` (episode time-avg rep).

    Excludes police. Returns NaN if no finite harvester values.
    """
    pm = onp.asarray(police_mask, dtype=bool).reshape(-1)
    vals: list[float] = []
    for i, agent in enumerate(agents):
        if i < pm.size and bool(pm[i]):
            continue
        key = f"agent_{int(agent)}/rep_mean_ep"
        if key not in env_m:
            continue
        v = float(env_m[key])
        if onp.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan")
    return float(onp.mean(onp.asarray(vals, dtype=onp.float64)))


class ActorCritic(nn.Module):
    """Actor-critic with CNN torso (episode_metrics.CNN).

    Reputation (when enabled) is encoded only in spatial obs channels ch15/ch16.
    """

    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        # CNN torso (episode_metrics.CNN).
        embedding = CNN(activation=self.activation)(x)

        actor = act(
            nn.Dense(64, kernel_init=orthogonal(onp.sqrt(2)), bias_init=constant(0.0))(
                embedding
            )
        )
        logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor)
        pi = distrax.Categorical(logits=logits)

        critic = act(
            nn.Dense(64, kernel_init=orthogonal(onp.sqrt(2)), bias_init=constant(0.0))(
                embedding
            )
        )
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return pi, jnp.squeeze(value, axis=-1)


@chex.dataclass(frozen=True)
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray


def reputation_config_from_dict(config: dict) -> ReputationConfig:
    """Build ``ReputationConfig`` from Hydra ``REPUTATION`` block (ignore embedding_dim)."""
    block = dict(config.get("REPUTATION") or {})
    block.pop("embedding_dim", None)
    block.pop("k_pos", None)
    block.pop("k_neg", None)
    bool_keys = {"reset_rep_on_tag_respawn", "enable_enforcement"}
    kwargs: dict = {}
    for k, v in block.items():
        if k in bool_keys:
            kwargs[k] = bool(v)
        else:
            kwargs[k] = float(v)
    if "enforcement_strictness" in kwargs:
        kwargs["enforcement_strictness"] = clamp_enforcement_strictness(
            kwargs["enforcement_strictness"]
        )
    return ReputationConfig(**kwargs)


def inject_reputation_enforcement_env_kwargs(
    config: dict, rep_cfg: ReputationConfig | None = None
) -> dict:
    """When reputation is on, enable env-side negative-rep timeouts via ENV_KWARGS.

    Mutates and returns ``config``. Safe no-op when reputation is disabled.
    Punishment duration is existing ``ENV_KWARGS.tag_timeout`` (default 25).
    """
    if not spatial_reputation_enabled(config):
        return config
    if rep_cfg is None:
        rep_cfg = reputation_config_from_dict(config)
    env_kwargs = dict(config.get("ENV_KWARGS") or {})
    env_kwargs["reputation_enforcement"] = bool(rep_cfg.enable_enforcement)
    env_kwargs["reputation_enforcement_strictness"] = float(
        rep_cfg.enforcement_strictness
    )
    config["ENV_KWARGS"] = env_kwargs
    return config


def joint_optimize_rewards(individual_rewards: jnp.ndarray) -> jnp.ndarray:
    """Broadcast team-sum reward for PPO; keep per-agent shape ``(..., N)``.

    Env returns individual apple rewards. Metrics / logging keep those. When ``OPTIMIZE_SHARED_REWARDS`` is on, each agent's learning
    target is ``sum_j r_j`` (same SocialJax ``shared_rewards`` team signal),
    without rewriting the env return used for R/S/P/E.
    """
    team = jnp.sum(individual_rewards, axis=-1, keepdims=True)
    return jnp.broadcast_to(team, individual_rewards.shape)


def _episode_reset_mask(episode_done: jnp.ndarray) -> jnp.ndarray:
    """Per-env episode reset, broadcast over agents. ``episode_done`` shape ``(..., N)``."""
    per_env = jnp.max(episode_done.astype(jnp.bool_), axis=-1, keepdims=True)
    return jnp.broadcast_to(per_env, episode_done.shape)


def baseline_optimize_rewards(
    individual_rewards: jnp.ndarray,
    cum_returns: jnp.ndarray,
    *,
    equality_coef: float = 1.0,
    penalty_cap: float = 1.0,
    episode_done: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Team-sum with bounded cumulative-return equality pressure for PPO.

    Tracks per-agent cumulative return within the inner episode. Let
    ``r`` be the step's individual rewards, ``C_i = C_i^{prev} + r_i``,
    ``\bar C = mean_k C_k``, and ``d_i = |C_i - \bar C| / max(\bar C, 1)``.
    With ``T = sum_j r_j`` (team sum) and ``\lambda =`` ``equality_coef``::

        penalty_i = penalty_cap * tanh(\lambda * d_i)
        learn_r_i = T * (1 - penalty_i)

    ``penalty_i`` lies in ``[0, penalty_cap)`` so the equality term can
    never cancel more than a ``penalty_cap`` fraction of the team signal
    (asymptotically ``learn_r_i \to T * (1 - penalty_cap)``). For small
    ``d_i``, ``tanh(\lambda d_i) \approx \lambda d_i`` and with ``T=1``,
    ``penalty_cap=1`` this matches the prior additive
    ``T - \lambda d_i`` behavior.

    Multiplicative scaling is used because baseline/paper configs
    Apple-only env returns are non-negative, so ``T \ge 0``; team harvest
    is reduced by at most ``penalty_cap * 100%``.

    ``C`` resets on ``episode_done`` (inner episode boundary). Env metrics
    stay individual; only ``Transition.reward`` is shaped. Toggle with
    ``OPTIMIZE_BASELINE=true``. Tunable via ``EQUALITY_COEF``, ``PENALTY_CAP``,
    and ``EQUALITY_COEF_WARMUP_FRAC`` (linear ramp of ``EQUALITY_COEF`` after an optional
    ``EQUALITY_COEF_DELAY_FRAC`` hold at zero).
    """
    rewards = individual_rewards.astype(jnp.float32)
    cum = cum_returns.astype(jnp.float32) + rewards
    avg = jnp.mean(cum, axis=-1, keepdims=True)
    dev = jnp.abs(cum - avg)
    scale = jnp.maximum(avg, jnp.float32(1.0))
    dev_norm = dev / scale
    team = jnp.sum(rewards, axis=-1, keepdims=True)
    coef = jnp.asarray(equality_coef, dtype=rewards.dtype)
    cap = jnp.asarray(penalty_cap, dtype=rewards.dtype)
    penalty = cap * jnp.tanh(coef * dev_norm)
    # Multiplicative form needs T >= 0 (non-negative apple returns).
    team_b = jnp.broadcast_to(team, rewards.shape)
    shaped = team_b * (1.0 - penalty)
    if episode_done is not None:
        reset = _episode_reset_mask(episode_done)
        cum = jnp.where(reset, jnp.float32(0.0), cum)
    return shaped.astype(rewards.dtype), cum.astype(rewards.dtype)


def effective_equality_coef(
    update_idx,
    target_coef: float,
    warmup_frac: float,
    total_updates: int,
    delay_frac: float = 0.0,
):
    """Delay at zero, then linear ramp to ``target_coef``.

    ``delay_frac``: hold coef at 0 for the first ``delay_frac * total_updates`` indices.
    ``warmup_frac``: ramp over the next ``warmup_frac * total_updates`` indices.
    ``warmup_frac=0`` (once past delay): full ``target_coef`` immediately.
    """
    target = jnp.asarray(target_coef, dtype=jnp.float32)
    delay_frac_f = jnp.asarray(delay_frac, dtype=jnp.float32)
    warmup_frac_f = jnp.asarray(warmup_frac, dtype=jnp.float32)
    total = jnp.float32(max(int(total_updates), 1))
    delay_updates = delay_frac_f * total
    warmup_updates = warmup_frac_f * total
    u = jnp.asarray(update_idx, dtype=jnp.float32)

    past_delay = u >= delay_updates
    # When delay_frac=0 and warmup_frac>0, floor denom at 1.0 to match the
    # pre-delay schedule exactly; otherwise use 1e-8 per the delay+ramp formula.
    denom = jnp.where(
        (delay_frac_f <= jnp.float32(0.0)) & (warmup_frac_f > jnp.float32(0.0)),
        jnp.maximum(warmup_updates, jnp.float32(1.0)),
        jnp.maximum(warmup_updates, jnp.float32(1e-8)),
    )
    ramp_progress = jnp.where(
        warmup_frac_f <= jnp.float32(0.0),
        jnp.float32(1.0),
        jnp.clip(
            (u - delay_updates) / denom,
            jnp.float32(0.0),
            jnp.float32(1.0),
        ),
    )
    return jnp.where(past_delay, target * ramp_progress, jnp.float32(0.0))


def baseline_optimize_kwargs_at_update(
    *,
    update_idx,
    equality_coef: float = 1.0,
    penalty_cap: float = 1.0,
    equality_coef_warmup_frac: float = 0.0,
    equality_coef_delay_frac: float = 0.0,
    total_updates: int,
) -> dict[str, jnp.ndarray | float]:
    """Build kwargs for ``baseline_optimize_rewards`` at a PPO update index."""
    return {
        "equality_coef": effective_equality_coef(
            update_idx,
            equality_coef,
            equality_coef_warmup_frac,
            total_updates,
            delay_frac=equality_coef_delay_frac,
        ),
        "penalty_cap": penalty_cap,
    }


def validate_baseline_optimize_config(config) -> None:
    """Placeholder for future baseline PPO shaping config checks."""
    del config


def _traj_to_agent_major(traj: Transition) -> Transition:
    """``(T, E, N, ...)`` → ``(N, T, E, ...)`` for agent-vmap'd GAE / PPO."""

    def _move(x):
        return jnp.transpose(x, (2, 0, 1) + tuple(range(3, x.ndim)))

    return Transition(
        obs=_move(traj.obs),
        action=_move(traj.action),
        reward=_move(traj.reward),
        done=_move(traj.done),
        log_prob=_move(traj.log_prob),
        value=_move(traj.value),
    )


def calculate_gae(
    reward: jnp.ndarray,
    value: jnp.ndarray,
    done: jnp.ndarray,
    last_val: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
    *,
    unroll: int = 16,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """GAE matching SocialJax IPPO independent-agent path (``PARAMETER_SHARING=False``).

    Source: ``foundation_code/.../IPPO/ippo_cnn_harvest_open.py`` ``_calculate_gae``.
    Arrays are time-major ``(T, ...)``; ``last_val`` has shape ``(...)``.
    Returns ``(advantages, targets)`` with ``targets = advantages + value``.
    """

    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done_t, value_t, reward_t = transition
        delta = reward_t + gamma * next_value * (1 - done_t) - value_t
        gae = delta + gamma * gae_lambda * (1 - done_t) * gae
        return (gae, value_t), gae

    # Scan over (done, value, reward) like foundation's Transition fields used in GAE.
    traj = (done, value, reward)
    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj,
        reverse=True,
        unroll=unroll,
    )
    return advantages, advantages + value


def ppo_clip_loss_terms(
    log_prob: jnp.ndarray,
    log_prob_old: jnp.ndarray,
    value: jnp.ndarray,
    value_old: jnp.ndarray,
    advantages: jnp.ndarray,
    targets: jnp.ndarray,
    entropy: jnp.ndarray,
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Clipped PPO objective matching SocialJax IPPO ``_loss_fn`` (independent path).

    Source: ``foundation_code/.../IPPO/ippo_cnn_harvest_open.py`` lines ~247–278.
    ``entropy`` should already be reduced with ``.mean()`` (scalar), same as foundation
    ``pi.entropy().mean()``.
    """
    value_pred_clipped = value_old + (value - value_old).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_prob - log_prob_old)
    gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    total_loss = loss_actor + vf_coef * value_loss - ent_coef * entropy
    return total_loss, (value_loss, loss_actor, entropy)


_EPISODE_LIFE_FIELDS = (
    "R",
    "S",
    "P",
    "P_sys",
    "P_police",
    "E",
    "E_total",
    "count",
    "apple_sum",
    "apple_count",
    "zap_success_rate_sum",
    "zap_attempt_rate_sum",
    "zap_hit_rate_sum",
    "enforcement_hit_rate_sum",
    "enforcement_hit_count_sum",
    "enforcement_rep_total_sum",
    "zap_tagged_hit_count_sum",
    "zap_tagged_rep_total_sum",
    "police_zap_tagged_hit_count_sum",
    "police_zap_tagged_rep_total_sum",
    "apple_end_sum",
    "apple_late_mean_sum",
    "barren_t_sum",
    "barren_event_sum",
    "tagged_out_frac_sum",
    "enforcement_tagged_out_frac_sum",
    "police_tagged_out_frac_sum",
    "police_zap_attempt_rate_sum",
    "police_zap_hit_rate_sum",
    "police_reward_sum",
    "reward_variance_across_agents_sum",
    "base_rate_bad_ep_sum",
    "peer_zap_timeouts_ep_sum",
    "enforcement_timeouts_ep_sum",
    "enforcement_time_to_freeze_ep_sum",
    "bad_survival_ep_sum",
    "good_survival_ep_sum",
    "bandit_zap_count_ep_sum",
    "bandit_zap_rate_ep_sum",
)


def _checkpoint_scalar(val, *, what: str) -> float:
    """Coerce a pickled checkpoint scalar (numpy/jax/python) to float."""
    if isinstance(val, dict):
        raise TypeError(f"{what}: expected scalar, got dict")
    return float(jnp.asarray(val).item())


def _episode_life_to_dict(ep_life: EpisodeAggregates) -> dict[str, float]:
    return {
        field: float(jax.device_get(getattr(ep_life, field)))
        for field in _EPISODE_LIFE_FIELDS
    }


def _episode_life_from_dict(
    data: dict[str, float], *, num_agents: int = 1
) -> EpisodeAggregates:
    zn = jnp.zeros((num_agents,), dtype=jnp.float32)
    return EpisodeAggregates(
        **{
            field: jnp.asarray(data.get(field, 0.0), dtype=jnp.float32)
            for field in _EPISODE_LIFE_FIELDS
        },
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


def save_checkpoint(
    ckpt_dir: Path,
    train_states: dict,
    update: int,
    config: dict,
    *,
    ep_life: EpisodeAggregates | None = None,
) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_update_{update:06d}.pkl"
    payload = {
        "update": int(update),
        "params": {
            k: jax.device_get(ts.params) for k, ts in train_states.items()
        },
        "opt_state": {
            k: jax.device_get(ts.opt_state) for k, ts in train_states.items()
        },
        "steps": {
            k: int(jax.device_get(ts.step)) for k, ts in train_states.items()
        },
        "config": {
            k: v
            for k, v in config.items()
            if isinstance(v, (int, float, str, bool, type(None)))
            or (
                isinstance(v, dict)
                and all(
                    isinstance(x, (int, float, str, bool, type(None)))
                    for x in v.values()
                )
            )
        },
    }
    if ep_life is not None:
        payload["episode_life"] = _episode_life_to_dict(ep_life)
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    latest = ckpt_dir / "ckpt_latest.pkl"
    with latest.open("wb") as fh:
        pickle.dump(payload, fh)
    return path


def load_checkpoint(path: str | Path) -> dict:
    """Load a training checkpoint written by ``save_checkpoint``."""
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def resolve_warmstart_police_source_keys(
    payload: dict,
    *,
    n_dest_police: int,
    source_agent_ids: Sequence[int] | None = None,
) -> list[str]:
    """Pick source checkpoint agent keys to copy into dest police slots.

    Default: treat agents ``>= N_HARVESTERS`` in the checkpoint config as police
    (fallback: the last ``n_dest_police`` agents in the payload).
    """
    if n_dest_police < 1:
        raise ValueError("n_dest_police must be >= 1 for police warm-start")

    if source_agent_ids is not None:
        keys = [str(int(i)) for i in source_agent_ids]
    else:
        cfg = payload.get("config") or {}
        param_keys = sorted(
            (str(k) for k in payload["params"].keys()),
            key=lambda k: int(k) if k.isdigit() else k,
        )
        n_h = cfg.get("N_HARVESTERS")
        if n_h is not None and str(n_h) != "":
            n_h_i = int(n_h)
            keys = [k for k in param_keys if k.isdigit() and int(k) >= n_h_i]
        else:
            keys = param_keys[-n_dest_police:]

    missing = [k for k in keys if k not in payload["params"]]
    if missing:
        raise KeyError(
            f"WARMSTART_POLICE source agent keys missing from checkpoint: {missing}"
        )
    if len(keys) < n_dest_police:
        # One trained police → N identical frozen co-players (tile / clone).
        if len(keys) == 1:
            keys = keys * int(n_dest_police)
        else:
            raise ValueError(
                f"checkpoint has {len(keys)} police source agent(s) but dest needs "
                f"{n_dest_police}; pass WARMSTART_POLICE_SOURCE_AGENTS or use a "
                "checkpoint with enough police (single-source tiling only)"
            )
    return keys[:n_dest_police]


def replace_stacked_agent_params(
    stacked_params,
    *,
    agent_index: int,
    agent_params,
    num_seeds: int = 1,
):
    """Overwrite one agent slice in agent-stacked (optionally seed-stacked) params."""

    def _set(stack, leaf):
        leaf = jnp.asarray(leaf)
        if int(num_seeds) > 1:
            # leaves: (num_seeds, num_agents, ...)
            tiled = jnp.broadcast_to(leaf[None, ...], (int(num_seeds),) + leaf.shape)
            return stack.at[:, int(agent_index)].set(tiled)
        return stack.at[int(agent_index)].set(leaf)

    return jax.tree_util.tree_map(_set, stacked_params, agent_params)


def restore_frozen_police_slices(
    updated,
    frozen,
    police_mask: jnp.ndarray,
    *,
    num_seeds: int = 1,
):
    """Keep ``frozen`` on police agent axis; keep ``updated`` elsewhere.

    Leaves are agent-stacked as ``(N, ...)`` or seed-stacked as ``(S, N, ...)``.
    ``police_mask`` is shape ``(N,)``.
    """
    pm = jnp.asarray(police_mask, dtype=jnp.bool_)
    n_seeds = int(num_seeds)

    def _one(u, f):
        u = jnp.asarray(u)
        f = jnp.asarray(f)
        if u.shape != f.shape:
            raise ValueError(
                f"restore_frozen_police_slices shape mismatch: "
                f"updated {u.shape} vs frozen {f.shape}"
            )
        if n_seeds > 1:
            # (S, N, ...)
            mask = pm.reshape((1, pm.shape[0]) + (1,) * (u.ndim - 2))
        else:
            # (N, ...)
            mask = pm.reshape((pm.shape[0],) + (1,) * (u.ndim - 1))
        return jnp.where(mask, f, u)

    return jax.tree_util.tree_map(_one, updated, frozen)


def apply_police_param_warmstart(
    stacked_params,
    payload: dict,
    *,
    dest_police_indices: Sequence[int],
    source_agent_keys: Sequence[str],
    num_seeds: int = 1,
):
    """Copy source police weights into dest police slots; leave other agents unchanged."""
    if len(dest_police_indices) != len(source_agent_keys):
        raise ValueError(
            f"dest_police_indices ({list(dest_police_indices)}) and "
            f"source_agent_keys ({list(source_agent_keys)}) length mismatch"
        )
    out = stacked_params
    for dest_i, src_k in zip(dest_police_indices, source_agent_keys):
        src_params = jax.tree_util.tree_map(
            jnp.asarray, payload["params"][str(src_k)]
        )
        out = replace_stacked_agent_params(
            out,
            agent_index=int(dest_i),
            agent_params=src_params,
            num_seeds=int(num_seeds),
        )
    return out


def apply_harvester_param_warmstart(
    stacked_params,
    payload: dict,
    *,
    n_harvesters: int,
    num_seeds: int = 1,
):
    """Copy source agents ``0..n_harvesters-1`` into dest harvester slots."""
    n_h = int(n_harvesters)
    if n_h < 1:
        raise ValueError("n_harvesters must be >= 1 for harvester warm-start")
    out = stacked_params
    for i in range(n_h):
        key = str(i)
        if key not in payload["params"]:
            raise KeyError(
                f"WARMSTART_HARVESTERS source missing agent key {key!r} "
                f"(need 0..{n_h - 1})"
            )
        src_params = jax.tree_util.tree_map(
            jnp.asarray, payload["params"][key]
        )
        out = replace_stacked_agent_params(
            out,
            agent_index=i,
            agent_params=src_params,
            num_seeds=int(num_seeds),
        )
    return out


def _optional_ckpt_path(config: dict, key: str) -> str | None:
    path = config.get(key) or os.environ.get(key)
    if path in (None, "", "null", "None"):
        return None
    return str(path)


def restore_stacked_train_state(
    payload: dict,
    *,
    agent_keys: list[str],
) -> tuple[dict, dict, jnp.ndarray]:
    """Rebuild stacked params, opt_state, and steps from a checkpoint payload."""
    params = _stack_agent_pytrees(
        *[
            jax.tree_util.tree_map(jnp.asarray, payload["params"][k])
            for k in agent_keys
        ]
    )
    # opt_state is an optax tuple per agent — stack leaves, do not jnp.asarray the tuple.
    opt_state = _stack_agent_pytrees(
        *[payload["opt_state"][k] for k in agent_keys]
    )
    steps_raw = payload.get("steps")
    if steps_raw is None:
        steps = jnp.zeros((len(agent_keys),), dtype=jnp.int32)
    else:
        step_vals = []
        for k in agent_keys:
            raw = steps_raw[k]
            if isinstance(raw, dict):
                raise TypeError(
                    f"checkpoint steps[{k!r}] is a dict; expected a scalar step count"
                )
            step_vals.append(int(jnp.asarray(raw).item()))
        steps = jnp.array(step_vals, dtype=jnp.int32)
    return params, opt_state, steps


def metrics_from_rollout_ippo(
    step_metrics: StepMetrics,
    num_agents: int,
    agents,
    *,
    R: float = float("nan"),
    S: float = float("nan"),
    P: float = float("nan"),
    E: float = float("nan"),
    E_total: float = float("nan"),
    mean_episode_return: float = float("nan"),
    zap_success_rate: float = float("nan"),
    zap_attempt_rate: float = float("nan"),
    zap_hit_rate: float = float("nan"),
    n_episodes_since_log: int = 0,
    n_episodes_lifetime: int = 0,
) -> dict[str, float]:
    """Full-episode R/S/P/E plus per-agent breakdowns.

    R/S/P/E and mean_episode_return are per-episode quantities passed through
    from the on-device accumulator. The per-agent ``agent_i/*`` entries below
    remain *window* diagnostics (not plotted) — they summarise the current
    ``NUM_STEPS`` rollout only.
    """
    env_m = metrics_from_rollout(
        step_metrics,
        num_agents,
        R=R,
        S=S,
        P=P,
        E=E,
        E_total=E_total,
        mean_episode_return=mean_episode_return,
        zap_success_rate=zap_success_rate,
        zap_attempt_rate=zap_attempt_rate,
        zap_hit_rate=zap_hit_rate,
        n_episodes_since_log=n_episodes_since_log,
        n_episodes_lifetime=n_episodes_lifetime,
    )

    env_r = onp.asarray(step_metrics.env_rewards)
    tot_r = onp.asarray(step_metrics.total_rewards)
    tagged = onp.asarray(step_metrics.tagged)
    zap_ok = onp.asarray(step_metrics.zap_successful)
    zap_try = onp.asarray(step_metrics.zap_attempted)

    t, e, n = tot_r.shape
    tot_flat = tot_r.reshape(t * e, n)
    env_flat = env_r.reshape(t * e, n)
    tagged_flat = tagged.reshape(t * e, n)
    # Apple-only (same basis as aggregate mean_reward); tot includes peer-punish.
    per_agent_returns = env_flat.sum(axis=0)

    timesteps = onp.arange(t * e, dtype=onp.float64)
    for i, agent in enumerate(agents):
        prefix = f"agent_{int(agent)}"
        env_m[f"{prefix}/R_window"] = float(per_agent_returns[i])
        env_m[f"{prefix}/mean_step_reward"] = float(onp.mean(tot_r[..., i]))
        env_m[f"{prefix}/zap_success_rate_window"] = float(onp.mean(zap_ok[..., i]))
        env_m[f"{prefix}/zap_attempt_rate_window"] = float(onp.mean(zap_try[..., i]))
        # Per-agent sustainability S_i: same formula as compute_sustainability's
        # per-agent term (mean timestep index among env_reward>0 steps for this
        # agent). Aggregate S is nanmean of these S_i values — so agent_i/S is
        # the same statistic restricted to one agent's column, not a different metric.
        pos = env_flat[:, i] > 0
        if pos.any():
            env_m[f"{prefix}/S_window"] = float(onp.sum(timesteps[pos]) / pos.sum())
        else:
            env_m[f"{prefix}/S_window"] = 0.0
        env_m[f"{prefix}/tagged_frac_window"] = float(onp.mean(tagged_flat[:, i]))

    if hasattr(step_metrics, "rep"):
        rep = onp.asarray(step_metrics.rep)
        raw_rep = onp.asarray(step_metrics.raw_rep)
        if rep.size > 0 and rep.ndim >= 2:
            rep_flat = rep.reshape(-1, rep.shape[-1])
            raw_flat = raw_rep.reshape(-1, raw_rep.shape[-1])
            env_m["mean_rep"] = float(onp.mean(rep_flat))
            env_m["mean_raw_rep"] = float(onp.mean(raw_flat))
            for i, agent in enumerate(agents):
                prefix = f"agent_{int(agent)}"
                env_m[f"{prefix}/rep_window"] = float(onp.mean(rep_flat[:, i]))
                env_m[f"{prefix}/raw_rep_window"] = float(onp.mean(raw_flat[:, i]))

    return env_m


def make_train(config, env, agent_hparams: dict[str, dict[str, float]], ckpt_dir: Path | None = None, gif_renderer: GifRenderer | None = None):
    agents = list(env.agents)
    agent_keys = [str(a) for a in agents]
    num_agents = env.num_agents
    obs_shape = env.observation_space()[0].shape
    num_steps = int(config["NUM_STEPS"])
    num_envs = int(config["NUM_ENVS"])  # sweep-wide knob; not hardcoded
    num_seeds = max(1, int(config.get("NUM_SEEDS", 1)))
    num_inner_steps = int(config["ENV_KWARGS"]["num_inner_steps"])
    action_dim = env.action_space().n
    checkpoint_every = int(config.get("CHECKPOINT_EVERY", 0))
    use_wandb_logging = bool(config.get("USE_WANDB", False))
    use_metric_logging = use_wandb_logging
    update_epochs = int(config["UPDATE_EPOCHS"])
    num_minibatches = int(config["NUM_MINIBATCHES"])
    # Traced PPO scalars (passed into the jitted chunk — not Python-float
    # closures). Shape-static extras GAE_LAMBDA / VF_COEF / MAX_GRAD_NORM stay
    # closed-over Python floats; they are not swept.
    gamma = jnp.asarray(config["GAMMA"], dtype=jnp.float32)
    gae_lambda = float(config["GAE_LAMBDA"])
    vf_coef = float(config["VF_COEF"])
    max_grad_norm = float(config["MAX_GRAD_NORM"])
    activation = str(config.get("ACTIVATION", "relu"))
    anneal_lr = bool(config.get("ANNEAL_LR", False))
    anneal_ent = bool(config.get("ANNEAL_ENT", False))
    # When anneal on: ENT_COEF is start; ENT_COEF_END is hold value after horizon.
    ent_coef_end = float(config.get("ENT_COEF_END", config["ENT_COEF"]))
    has_agent_overrides = bool(config.get("AGENT_OVERRIDES"))
    verbose_logging = bool(config.get("VERBOSE_LOGGING", False))
    log_per_agent = bool(config.get("LOG_PER_AGENT", False))
    enable_reputation = spatial_reputation_enabled(config)
    enable_reputation_v2 = reputation_v2_enabled(config)
    # Episode/W&B reputation series: classic continuous OR v2 binary labels.
    log_reputation = enable_reputation or enable_reputation_v2
    police_mask = jnp.asarray(
        config.get("POLICE_MASK", onp.zeros((num_agents,), dtype=onp.bool_)),
        dtype=jnp.bool_,
    )
    freeze_police = bool(config.get("FREEZE_POLICE", False)) or (
        str(os.environ.get("FREEZE_POLICE", "")).lower() in ("1", "true", "yes")
    )
    debug_perf = verbose_logging and bool(config.get("DEBUG_PERF", False))
    perf_log_every = int(config.get("PERF_LOG_EVERY", 10))
    log_every = int(config["LOG_EVERY"])
    updates_per_jit = int(config.get("UPDATES_PER_JIT", log_every))
    memory_probe = bool(config.get("MEMORY_PROBE", False))
    # Train on team-sum (+ optional equality) reward; keep individual for metrics.
    # Do NOT set ENV_KWARGS.shared_rewards=true — that rewrites env rewards and
    # inflates logged mean_reward by ~N.
    optimize_shared_rewards = bool(config.get("OPTIMIZE_SHARED_REWARDS", False))
    optimize_baseline = bool(config.get("OPTIMIZE_BASELINE", False))
    equality_coef_target = float(config.get("EQUALITY_COEF", 1.0))
    penalty_cap = float(config.get("PENALTY_CAP", 1.0))
    equality_coef_warmup_frac = float(config.get("EQUALITY_COEF_WARMUP_FRAC", 0.0))
    equality_coef_delay_frac = float(config.get("EQUALITY_COEF_DELAY_FRAC", 0.0))
    if optimize_baseline and optimize_shared_rewards:
        raise ValueError(
            "Set only one of OPTIMIZE_BASELINE / OPTIMIZE_SHARED_REWARDS "
            "(baseline already includes the team-sum signal plus equality)."
        )
    if (optimize_shared_rewards or optimize_baseline) and bool(
        (config.get("ENV_KWARGS") or {}).get("shared_rewards", False)
    ):
        flag = (
            "OPTIMIZE_BASELINE"
            if optimize_baseline
            else "OPTIMIZE_SHARED_REWARDS"
        )
        raise ValueError(
            f"{flag}=true requires ENV_KWARGS.shared_rewards=false "
            "(env must return individual rewards for metrics; joint/baseline "
            "signal is applied only to the PPO Transition.reward)."
        )
    validate_baseline_optimize_config(config)

    # Per agent: NUM_ACTORS = NUM_ENVS (independent policies)
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"]) // num_steps // num_envs
    )
    num_updates_total = int(config["NUM_UPDATES"])
    # Optional LR anneal horizon (env steps). Default = this run's TOTAL_TIMESTEPS.
    # Example: TOTAL=50M + ANNEAL_LR_TOTAL_TIMESTEPS=100M → LR reaches half initial
    # at the end, matching the first half of a 100M anneal-to-zero schedule.
    _anneal_horizon = config.get("ANNEAL_LR_TOTAL_TIMESTEPS")
    if _anneal_horizon is not None and _anneal_horizon != "":
        config["ANNEAL_LR_NUM_UPDATES"] = (
            int(_anneal_horizon) // num_steps // num_envs
        )
    else:
        config["ANNEAL_LR_NUM_UPDATES"] = config["NUM_UPDATES"]
    # Optional entropy anneal horizon (env steps).
    # Priority: ANNEAL_ENT_TOTAL_TIMESTEPS > ANNEAL_ENT_FRAC * TOTAL > full TOTAL.
    # Default frac=0.25 → anneal in the first quarter of training, then hold.
    _ent_anneal_horizon = config.get("ANNEAL_ENT_TOTAL_TIMESTEPS")
    if _ent_anneal_horizon is not None and _ent_anneal_horizon != "":
        _ent_hz_steps = int(_ent_anneal_horizon)
    else:
        _ent_frac = config.get("ANNEAL_ENT_FRAC", 0.25)
        if _ent_frac is not None and _ent_frac != "":
            _ent_hz_steps = max(
                1, int(float(_ent_frac) * int(config["TOTAL_TIMESTEPS"]))
            )
        else:
            _ent_hz_steps = int(config["TOTAL_TIMESTEPS"])
    config["ANNEAL_ENT_NUM_UPDATES"] = max(
        1, _ent_hz_steps // num_steps // num_envs
    )
    # Parallel seeds do not change the per-seed update count.
    config["NUM_SEEDS"] = num_seeds
    minibatch_size = (num_envs * num_steps) // num_minibatches
    assert minibatch_size * num_minibatches == num_envs * num_steps, (
        "NUM_ENVS * NUM_STEPS must be divisible by NUM_MINIBATCHES"
    )

    # Traced per-agent loss / optimizer hparams (indexed by agent order).
    # Passed into the jitted ``_run_chunk`` so GAMMA / ENT_COEF / CLIP_EPS / LR
    # changes do not retrace the update body.
    ent_coefs = jnp.array(
        [agent_hparams[k]["entropy_coef"] for k in agent_keys], dtype=jnp.float32
    )
    clip_epses = jnp.array(
        [agent_hparams[k]["clip_eps"] for k in agent_keys], dtype=jnp.float32
    )
    lrs = jnp.array(
        [agent_hparams[k]["lr"] for k in agent_keys], dtype=jnp.float32
    )

    def train(rng):
        network = ActorCritic(
            action_dim=action_dim,
            activation=activation,
        )
        init_x = jnp.zeros((1, *obs_shape))
        # Shared transform; per-agent lr lives in inject_hyperparams state (stacked)
        # and is rewritten each update from the traced ``lrs`` argument.
        # Anneal writes numeric lrs into hyperparams (no per-agent schedule closure).
        shared_tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=1.0, eps=1e-5),
        )

        wrapped = CTRolloutManager(
            env, batch_size=num_envs, preprocess_obs=False
        )

        def _init_network(init_key):
            return network.init(init_key, init_x)

        def _init_one(rng_i):
            """One seed: stacked agent params + env batch. Vmapped when NUM_SEEDS>1."""
            params_list = []
            opt_list = []
            for key in agent_keys:
                rng_i, init_key = jax.random.split(rng_i)
                params_i = _init_network(init_key)
                ts = TrainState.create(
                    apply_fn=network.apply,
                    params=params_i,
                    tx=shared_tx,
                )
                ts = ts.replace(
                    opt_state=_set_chain_lrs(
                        ts.opt_state, agent_hparams[key]["lr"]
                    )
                )
                params_list.append(ts.params)
                opt_list.append(ts.opt_state)
            params = _stack_agent_pytrees(*params_list)
            opt_state = _stack_agent_pytrees(*opt_list)
            steps = jnp.zeros((num_agents,), dtype=jnp.int32)
            rng_i, reset_key = jax.random.split(rng_i)
            obs_dict_i, env_state_i = wrapped.batch_reset(reset_key)
            pe_acc_i = init_per_episode_acc(num_envs, num_agents, num_inner_steps)
            ep_life_i = init_episode_aggregates(num_agents)
            cum_returns_i = jnp.zeros((num_envs, num_agents), dtype=jnp.float32)
            return (
                rng_i,
                params,
                opt_state,
                steps,
                env_state_i,
                obs_dict_i,
                pe_acc_i,
                ep_life_i,
                cum_returns_i,
            )

        # Host-only verification (seed-0 init path, not vmapped).
        print("[ippo] Per-agent optimizer / loss hparam verification:")
        _vrng = jax.random.PRNGKey(0)
        for key in agent_keys:
            _vrng, init_key = jax.random.split(_vrng)
            params_i = _init_network(init_key)
            ts = TrainState.create(
                apply_fn=network.apply, params=params_i, tx=shared_tx
            )
            ts = ts.replace(
                opt_state=_set_chain_lrs(ts.opt_state, agent_hparams[key]["lr"])
            )
            hp = agent_hparams[key]
            opt_lr = _read_optimizer_lr(ts)
            opt_lr_str = f"{opt_lr}" if opt_lr is not None else "unavailable"
            print(
                f"[ippo]   agent_{key}: hparams={hp} | "
                f"optimizer_lr={opt_lr_str} | "
                f"loss uses clip_eps={hp['clip_eps']}, "
                f"entropy_coef={hp['entropy_coef']}"
            )
            if opt_lr is not None and not onp.isclose(
                opt_lr, hp["lr"], rtol=1e-5, atol=1e-9
            ):
                print(
                    f"[ippo] WARNING: agent_{key} optimizer_lr={opt_lr} "
                    f"!= resolved hparams lr={hp['lr']}"
                )
        if has_agent_overrides:
            warn_if_overrides_collapse(agent_hparams)

        if num_seeds == 1:
            (
                rng,
                params,
                opt_state,
                steps,
                env_state,
                obs_dict,
                pe_acc,
                ep_life,
                cum_returns,
            ) = _init_one(rng)
        else:
            # Independent replica RNGs: split the Hydra SEED key along the
            # vmap axis. Same key is NOT reused. _init_one then splits each
            # replica's key further for net init + env reset.
            seed_rngs = jax.random.split(rng, num_seeds)
            (
                rng,
                params,
                opt_state,
                steps,
                env_state,
                obs_dict,
                pe_acc,
                ep_life,
                cum_returns,
            ) = jax.vmap(_init_one)(seed_rngs)
            print(
                f"[multi-seed] NUM_SEEDS={num_seeds} — vmapping _run_chunk "
                f"over seeds (base SEED={config.get('SEED', '?')})"
            )

        print(
            f"[ippo] Agent-vmap enabled: stacked params over N={num_agents} "
            f"independent actor-critics; NUM_ENVS={num_envs}; "
            f"NUM_SEEDS={num_seeds}."
        )

        warmstart_police_path = _optional_ckpt_path(
            config, "WARMSTART_POLICE_CHECKPOINT"
        )
        warmstart_harv_path = _optional_ckpt_path(
            config, "WARMSTART_HARVESTERS_CHECKPOINT"
        )
        resume_ckpt_path_early = _optional_ckpt_path(config, "RESUME_CHECKPOINT")
        if freeze_police:
            if not bool(config.get("ENABLE_POLICE_AGENT", False)):
                raise ValueError("FREEZE_POLICE requires ENABLE_POLICE_AGENT=true")
            if not warmstart_police_path:
                raise ValueError(
                    "FREEZE_POLICE requires WARMSTART_POLICE_CHECKPOINT "
                    "(freezing a randomly initialized police is not supported)"
                )
            if not bool(onp.any(onp.asarray(jax.device_get(police_mask), dtype=bool))):
                raise ValueError("FREEZE_POLICE set but POLICE_MASK has no police agents")
        if (warmstart_police_path or warmstart_harv_path) and resume_ckpt_path_early:
            raise ValueError(
                "WARMSTART_*_CHECKPOINT and RESUME_CHECKPOINT are mutually "
                "exclusive (warm-start loads weights into a fresh run)"
            )
        if warmstart_harv_path:
            n_h = int(config.get("N_HARVESTERS") or 0)
            if n_h < 1:
                n_h = int(num_agents) - int(
                    onp.sum(onp.asarray(jax.device_get(police_mask), dtype=bool))
                )
            if n_h < 1:
                raise ValueError(
                    "WARMSTART_HARVESTERS_CHECKPOINT requires N_HARVESTERS >= 1"
                )
            try:
                harv_payload = load_checkpoint(warmstart_harv_path)
                params = apply_harvester_param_warmstart(
                    params,
                    harv_payload,
                    n_harvesters=n_h,
                    num_seeds=num_seeds,
                )
            except Exception as exc:
                print(
                    f"[warmstart] FATAL: failed to load harvesters from "
                    f"{warmstart_harv_path}: {type(exc).__name__}: {exc}"
                )
                raise
            print(
                f"[warmstart] harvester params from {warmstart_harv_path} "
                f"→ agents 0..{n_h - 1}; opt_state/steps stay fresh"
            )
        if warmstart_police_path:
            if not bool(config.get("ENABLE_POLICE_AGENT", False)):
                raise ValueError(
                    "WARMSTART_POLICE_CHECKPOINT requires ENABLE_POLICE_AGENT=true"
                )
            dest_police = police_agent_indices(police_mask)
            if not dest_police:
                raise ValueError(
                    "WARMSTART_POLICE_CHECKPOINT set but no police agents in env"
                )
            src_ids_cfg = config.get("WARMSTART_POLICE_SOURCE_AGENTS") or os.environ.get(
                "WARMSTART_POLICE_SOURCE_AGENTS"
            )
            src_ids: list[int] | None = None
            if src_ids_cfg not in (None, "", "null", "None"):
                if isinstance(src_ids_cfg, (list, tuple)):
                    src_ids = [int(x) for x in src_ids_cfg]
                else:
                    src_ids = [
                        int(x.strip())
                        for x in str(src_ids_cfg).split(",")
                        if x.strip()
                    ]
            try:
                ws_payload = load_checkpoint(warmstart_police_path)
                src_keys = resolve_warmstart_police_source_keys(
                    ws_payload,
                    n_dest_police=len(dest_police),
                    source_agent_ids=src_ids,
                )
                params = apply_police_param_warmstart(
                    params,
                    ws_payload,
                    dest_police_indices=dest_police,
                    source_agent_keys=src_keys,
                    num_seeds=num_seeds,
                )
            except Exception as exc:
                print(
                    f"[warmstart] FATAL: failed to load police from "
                    f"{warmstart_police_path}: {type(exc).__name__}: {exc}"
                )
                raise
            harv_note = (
                "harvesters from WARMSTART_HARVESTERS_CHECKPOINT"
                if warmstart_harv_path
                else "harvesters freshly initialized"
            )
            if freeze_police:
                print(
                    f"[warmstart] police params from {warmstart_police_path} "
                    f"source_agents={src_keys} → dest_indices={list(dest_police)}; "
                    f"{harv_note}; FREEZE_POLICE=true (fixed co-player, no PPO updates)"
                )
            else:
                print(
                    f"[warmstart] police params from {warmstart_police_path} "
                    f"source_agents={src_keys} → dest_indices={list(dest_police)}; "
                    f"{harv_note}; opt_state/steps reset; police continue learning"
                )

        def _apply_network(params_i, obs_i):
            return network.apply(params_i, obs_i)

        def _select_actions(rng, params, obs_batch):
            """Independent policies via vmap over agents. obs_batch (E, N, ...)."""
            obs_ne = jnp.swapaxes(obs_batch, 0, 1)  # (N, E, ...)
            keys = jax.random.split(rng, num_agents)

            def _one(params_i, obs_i, key):
                pi, v = _apply_network(params_i, obs_i)
                a = pi.sample(seed=key).astype(jnp.int32)
                return a, pi.log_prob(a), v

            actions_n, log_probs_n, values_n = jax.vmap(_one)(
                params, obs_ne, keys
            )
            return (
                jnp.swapaxes(actions_n, 0, 1),
                jnp.swapaxes(log_probs_n, 0, 1),
                jnp.swapaxes(values_n, 0, 1),
            )

        def _collect_rollout(rng, params, env_state, obs_dict, pe_acc, cum_returns, update_i):
            def _env_step(carry, _):
                rng, env_state, obs_dict, pe_acc, cum_returns = carry
                rng, act_key, step_key = jax.random.split(rng, 3)
                obs_batch = _stack_agent_dim(obs_dict, agents)
                actions, log_probs, values = _select_actions(
                    act_key, params, obs_batch
                )
                next_obs_dict, env_state, reward_dict, done_dict, info = (
                    wrapped.batch_step(step_key, env_state, actions)
                )
                rewards = _stack_agent_dim(reward_dict, agents)
                dones = _stack_agent_dim(done_dict, agents).astype(jnp.float32)

                env_rew = _info_field(
                    info, "environmental_rewards", (num_envs, num_agents)
                )
                tagged = _info_field(
                    info, "tagged", (num_envs, num_agents), dtype=jnp.bool_
                )
                # Peace / tagged_out_frac: agent-beam (peer) timeouts only.
                # system_peace: reputation-enforcement + police timeouts.
                # Fall back to total ``tagged`` when cause fields are absent.
                if "agent_tagged" in info:
                    agent_tagged = jnp.asarray(info["agent_tagged"])
                else:
                    agent_tagged = tagged
                enforcement_tagged = _info_field(
                    info,
                    "enforcement_tagged",
                    (num_envs, num_agents),
                    dtype=jnp.bool_,
                )
                police_tagged = _info_field(
                    info,
                    "police_tagged",
                    (num_envs, num_agents),
                    dtype=jnp.bool_,
                )
                # Police timeouts are enforcement (system_peace), not peer peace.
                # Keep ``police_tagged`` for enforcement/police_* breakdowns.
                system_tagged = jnp.logical_or(enforcement_tagged, police_tagged)
                zap_ok = _info_field(
                    info, "zap_successful", (num_envs, num_agents), dtype=jnp.bool_
                )
                zap_try = _info_field(
                    info, "zap_attempted", (num_envs, num_agents), dtype=jnp.bool_
                )
                police_zap_ok = _info_field(
                    info,
                    "police_zap_successful",
                    (num_envs, num_agents),
                    dtype=jnp.bool_,
                )
                police_zap_try = _info_field(
                    info,
                    "police_zap_attempted",
                    (num_envs, num_agents),
                    dtype=jnp.bool_,
                )
                police_reward = _info_field(
                    info, "police_zap_reward", (num_envs, num_agents)
                )
                police_zap_hit_matrix = _info_field(
                    info,
                    "police_zap_hit_matrix",
                    (num_envs, num_agents, num_agents),
                    dtype=jnp.bool_,
                )
                # Victims hit by a police beam this step → enforcement hit events.
                police_enforcement_hits = jnp.any(police_zap_hit_matrix, axis=1)
                inner_done = _info_field(
                    info, "inner_episode_done", (num_envs, num_agents), dtype=jnp.bool_
                )
                inner_t = _info_field(
                    info, "inner_t", (num_envs, num_agents), dtype=jnp.int32
                )
                ep_flag = _info_field(
                    info, "returned_episode", (num_envs, num_agents), dtype=jnp.bool_
                )
                ep_returns = _info_field(
                    info, "returned_episode_returns", (num_envs, num_agents)
                )
                # Env puts the same AppleCount on every agent; take env axis only.
                apple_info = _info_field(
                    info, "AppleCount_info", (num_envs, num_agents)
                )
                apple_e = apple_info[..., 0]
                apple_sum, apple_count = update_apple_count_step(apple_e)
                rep_info = _info_field(
                    info, "rep", (num_envs, num_agents)
                )
                rep_pre_info = _info_field(
                    info, "rep_pre", (num_envs, num_agents)
                )
                rate_ema_info = _info_field(
                    info, "rate_ema", (num_envs, num_agents)
                )
                burst_count_info = _info_field(
                    info, "burst_count", (num_envs, num_agents)
                )
                burst_penalty_info = _info_field(
                    info, "burst_penalty", (num_envs, num_agents)
                )
                raw_rep_info = _info_field(
                    info, "raw_rep", (num_envs, num_agents)
                )
                zap_hit_matrix = _info_field(
                    info,
                    "zap_hit_matrix",
                    (num_envs, num_agents, num_agents),
                    dtype=jnp.bool_,
                )
                # Attempt-time beam occupancy (pre police-focus). Fallback = hits
                # for older envs that only expose zap_hit_matrix.
                if "zap_beam_matrix" in info:
                    zap_beam_matrix = jnp.asarray(info["zap_beam_matrix"])
                else:
                    zap_beam_matrix = zap_hit_matrix
                active_agents = _info_field(
                    info, "active", (num_envs, num_agents), dtype=jnp.bool_
                )
                tags_received = jnp.any(zap_hit_matrix, axis=1).astype(jnp.float32)
                reputation_zapped = _info_field(
                    info,
                    "reputation_zapped",
                    (num_envs, num_agents),
                    dtype=jnp.bool_,
                )
                # v2 delayed/periodic Bad freezes + police zap victims → enforcement hits.
                v2_enf = (
                    _info_field(
                        info,
                        "periodic_enforcement_trigger",
                        (num_envs, num_agents),
                        dtype=jnp.bool_,
                    )
                    | _info_field(
                        info,
                        "immediate_timeout_trigger",
                        (num_envs, num_agents),
                        dtype=jnp.bool_,
                    )
                    | _info_field(
                        info,
                        "delayed_timeout_trigger",
                        (num_envs, num_agents),
                        dtype=jnp.bool_,
                    )
                )
                reputation_zapped = jnp.logical_or(
                    reputation_zapped, jnp.logical_or(v2_enf, police_enforcement_hits)
                )
                reputation_zapped = jnp.logical_or(
                    reputation_zapped,
                    _info_field(
                        info,
                        "prob_enforcement_zapped",
                        (num_envs, num_agents),
                        dtype=jnp.bool_,
                    ),
                )
                agent_beam_timeout_hit = _info_field(
                    info,
                    "agent_beam_timeout_hit",
                    (num_envs, num_agents),
                    dtype=jnp.bool_,
                )
                enforcement_timeout_hit = _info_field(
                    info,
                    "enforcement_timeout_hit",
                    (num_envs, num_agents),
                    dtype=jnp.bool_,
                )
                enforcement_freeze_bad_age = _info_field(
                    info,
                    "enforcement_freeze_bad_age",
                    (num_envs, num_agents),
                )
                bad_age_info = _info_field(
                    info, "bad_age", (num_envs, num_agents), dtype=jnp.int32
                ).astype(jnp.float32)
                good_age_info = _info_field(
                    info, "good_age", (num_envs, num_agents), dtype=jnp.int32
                ).astype(jnp.float32)

                # Metrics (R/mean_reward): apple collection only
                # (``environmental_rewards``). Never use ``rewards`` here —
                # that tensor includes peer-punish zap bonuses/penalties and
                # police zap rewards (learning signal only), and
                # shared_rewards=true would also inflate R by ~N.
                metrics_rew = env_rew.astype(jnp.float32)

                (
                    pe_acc,
                    R_sum,
                    S_sum,
                    P_sum,
                    E_sum,
                    E_total_sum,
                    ep_count,
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
                ) = update_per_episode_acc(
                    pe_acc,
                    metrics_rew,
                    env_rew.astype(jnp.float32),
                    agent_tagged.astype(jnp.bool_),
                    zap_ok.astype(jnp.bool_),
                    zap_try.astype(jnp.bool_),
                    inner_t,
                    inner_done.astype(jnp.bool_),
                    num_inner_steps,
                    num_agents,
                    apple_e.astype(jnp.float32),
                    rep=rep_info.astype(jnp.float32),
                    tags_received=tags_received,
                    rep_pre=rep_pre_info.astype(jnp.float32),
                    rate_ema=rate_ema_info.astype(jnp.float32),
                    burst_count=burst_count_info.astype(jnp.float32),
                    burst_penalty=burst_penalty_info.astype(jnp.float32),
                    zap_hit_matrix=zap_hit_matrix,
                    reputation_zapped=reputation_zapped,
                    enforcement_tagged=system_tagged.astype(jnp.bool_),
                    police_mask=police_mask,
                    police_tagged=police_tagged.astype(jnp.bool_),
                    police_zap_try=police_zap_try.astype(jnp.bool_),
                    police_zap_ok=police_zap_ok.astype(jnp.bool_),
                    police_reward=police_reward.astype(jnp.float32),
                    zap_beam_matrix=zap_beam_matrix,
                    active=active_agents.astype(jnp.bool_),
                    zap_attempted_agents=zap_try.astype(jnp.bool_),
                    agent_beam_timeout_hit=agent_beam_timeout_hit.astype(jnp.bool_),
                    enforcement_timeout_hit=enforcement_timeout_hit.astype(jnp.bool_),
                    enforcement_freeze_bad_age=enforcement_freeze_bad_age.astype(
                        jnp.float32
                    ),
                    bad_age=bad_age_info,
                    good_age=good_age_info,
                    bandit_zap_count=_info_field(
                        info, "bandit_zap_count", (num_envs,), dtype=jnp.float32
                    ),
                )

                # PPO/GAE target: optional team-sum / baseline shaping; metrics stay individual.
                learn_rewards = rewards.astype(jnp.float32)
                if optimize_baseline:
                    bo_kwargs = baseline_optimize_kwargs_at_update(
                        update_idx=update_i,
                        equality_coef=equality_coef_target,
                        penalty_cap=penalty_cap,
                        equality_coef_warmup_frac=equality_coef_warmup_frac,
                        equality_coef_delay_frac=equality_coef_delay_frac,
                        total_updates=num_updates_total,
                    )
                    learn_rewards, cum_returns = baseline_optimize_rewards(
                        learn_rewards,
                        cum_returns,
                        equality_coef=bo_kwargs["equality_coef"],
                        penalty_cap=bo_kwargs["penalty_cap"],
                        episode_done=inner_done,
                    )
                elif optimize_shared_rewards:
                    learn_rewards = joint_optimize_rewards(learn_rewards)

                transition = Transition(
                    obs=obs_batch.astype(jnp.float32),
                    action=actions.astype(jnp.int32),
                    reward=learn_rewards,
                    done=dones.astype(jnp.float32),
                    log_prob=log_probs.astype(jnp.float32),
                    value=values.astype(jnp.float32),
                )
                step_m = StepMetrics(
                    env_rewards=env_rew.astype(jnp.float32),
                    total_rewards=rewards.astype(jnp.float32),
                    tagged=tagged.astype(jnp.bool_),
                    zap_successful=zap_ok.astype(jnp.bool_),
                    zap_attempted=zap_try.astype(jnp.bool_),
                    inner_episode_done=inner_done.astype(jnp.bool_),
                    returned_episode=ep_flag.astype(jnp.bool_),
                    returned_episode_returns=ep_returns.astype(jnp.float32),
                    # IPPO has no Q-network; zeros keep shared StepMetrics / metrics_from_rollout happy.
                    mean_q=jnp.zeros((action_dim,), dtype=jnp.float32),
                    greedy_zap_rate=jnp.float32(0.0),
                    mean_q_zap=jnp.float32(0.0),
                    rep=rep_info.astype(jnp.float32),
                    raw_rep=raw_rep_info.astype(jnp.float32),
                    tags_received=tags_received,
                )
                return (
                    (rng, env_state, next_obs_dict, pe_acc, cum_returns),
                    (transition, step_m, R_sum, S_sum, P_sum, E_sum, E_total_sum,
                     ep_count,
                     zap_success_rate_sum, zap_attempt_rate_sum,
                     zap_hit_rate_sum, apple_sum, apple_count, apple_end_sum,
                     apple_late_mean_sum, barren_t_sum, barren_event_sum,
                     tagged_out_frac_sum, reward_variance_across_agents_sum,
                     agent_rep_min_sum, agent_rep_max_sum, agent_rep_mean_sum,
                     agent_zap_ok_ep_sum, agent_zap_try_ep_sum,
                     agent_zap_targets_hit_ep_sum,
                     agent_tags_received_ep_sum, agent_harvest_rate_mean_sum,
                     agent_harvest_rate_max_sum,
                     agent_burst_count_mean_sum, agent_burst_count_max_sum,
                     agent_burst_penalty_mean_sum, agent_burst_penalty_max_sum,
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
                     agent_peer_beam_hits_bad_ep_sum),
                )

            (rng, env_state, obs_dict, pe_acc, cum_returns), (
                traj,
                step_metrics,
                R_sums,
                S_sums,
                P_sums,
                E_sums,
                E_total_sums,
                ep_counts,
                zap_success_rate_sums,
                zap_attempt_rate_sums,
                zap_hit_rate_sums,
                apple_sums,
                apple_counts,
                apple_end_sums,
                apple_late_mean_sums,
                barren_t_sums,
                barren_event_sums,
                tagged_out_frac_sums,
                reward_variance_across_agents_sums,
                agent_rep_min_sums,
                agent_rep_max_sums,
                agent_rep_mean_sums,
                agent_zap_ok_ep_sums,
                agent_zap_try_ep_sums,
                agent_zap_targets_hit_ep_sums,
                agent_tags_received_ep_sums,
                agent_harvest_rate_mean_sums,
                agent_harvest_rate_max_sums,
                agent_burst_count_mean_sums,
                agent_burst_count_max_sums,
                agent_burst_penalty_mean_sums,
                agent_burst_penalty_max_sums,
                agent_zap_target_rep_mean_sums,
                agent_enforcement_hits_ep_sums,
                agent_enforcement_rep_mean_sums,
                agent_return_ep_sums,
                enforcement_hit_rate_sums,
                enforcement_hit_count_sums,
                enforcement_rep_total_sums,
                zap_tagged_hit_count_sums,
                zap_tagged_rep_total_sums,
                police_zap_tagged_hit_count_sums,
                police_zap_tagged_rep_total_sums,
                P_sys_sums,
                enforcement_tagged_out_frac_sums,
                P_police_sums,
                police_tagged_out_frac_sums,
                police_zap_attempt_rate_sums,
                police_zap_hit_rate_sums,
                police_reward_ep_sums,
                agent_target_bad_rate_attempted_ep_sums,
                agent_target_bad_rate_hit_ep_sums,
                base_rate_bad_ep_sums,
                peer_zap_timeouts_ep_sums,
                enforcement_timeouts_ep_sums,
                enforcement_time_to_freeze_ep_sums,
                bad_survival_ep_sums,
                good_survival_ep_sums,
                bandit_zap_count_ep_sums,
                bandit_zap_rate_ep_sums,
                agent_peer_beam_hits_good_ep_sums,
                agent_peer_beam_hits_bad_ep_sums,
            ) = jax.lax.scan(
                _env_step,
                (rng, env_state, obs_dict, pe_acc, cum_returns),
                None,
                length=num_steps,
            )
            window = EpisodeAggregates(
                R=jnp.sum(R_sums),
                S=jnp.sum(S_sums),
                P=jnp.sum(P_sums),
                P_sys=jnp.sum(P_sys_sums),
                P_police=jnp.sum(P_police_sums),
                E=jnp.sum(E_sums),
                E_total=jnp.sum(E_total_sums),
                count=jnp.sum(ep_counts),
                apple_sum=jnp.sum(apple_sums),
                apple_count=jnp.sum(apple_counts),
                zap_success_rate_sum=jnp.sum(zap_success_rate_sums),
                zap_attempt_rate_sum=jnp.sum(zap_attempt_rate_sums),
                zap_hit_rate_sum=jnp.sum(zap_hit_rate_sums),
                enforcement_hit_rate_sum=jnp.sum(enforcement_hit_rate_sums),
                enforcement_hit_count_sum=jnp.sum(enforcement_hit_count_sums),
                enforcement_rep_total_sum=jnp.sum(enforcement_rep_total_sums),
                zap_tagged_hit_count_sum=jnp.sum(zap_tagged_hit_count_sums),
                zap_tagged_rep_total_sum=jnp.sum(zap_tagged_rep_total_sums),
                police_zap_tagged_hit_count_sum=jnp.sum(
                    police_zap_tagged_hit_count_sums
                ),
                police_zap_tagged_rep_total_sum=jnp.sum(
                    police_zap_tagged_rep_total_sums
                ),
                apple_end_sum=jnp.sum(apple_end_sums),
                apple_late_mean_sum=jnp.sum(apple_late_mean_sums),
                barren_t_sum=jnp.sum(barren_t_sums),
                barren_event_sum=jnp.sum(barren_event_sums),
                tagged_out_frac_sum=jnp.sum(tagged_out_frac_sums),
                enforcement_tagged_out_frac_sum=jnp.sum(
                    enforcement_tagged_out_frac_sums
                ),
                police_tagged_out_frac_sum=jnp.sum(police_tagged_out_frac_sums),
                police_zap_attempt_rate_sum=jnp.sum(police_zap_attempt_rate_sums),
                police_zap_hit_rate_sum=jnp.sum(police_zap_hit_rate_sums),
                police_reward_sum=jnp.sum(police_reward_ep_sums),
                reward_variance_across_agents_sum=jnp.sum(
                    reward_variance_across_agents_sums
                ),
                base_rate_bad_ep_sum=jnp.sum(base_rate_bad_ep_sums),
                peer_zap_timeouts_ep_sum=jnp.sum(peer_zap_timeouts_ep_sums),
                enforcement_timeouts_ep_sum=jnp.sum(enforcement_timeouts_ep_sums),
                enforcement_time_to_freeze_ep_sum=jnp.sum(
                    enforcement_time_to_freeze_ep_sums
                ),
                bad_survival_ep_sum=jnp.sum(bad_survival_ep_sums),
                good_survival_ep_sum=jnp.sum(good_survival_ep_sums),
                bandit_zap_count_ep_sum=jnp.sum(bandit_zap_count_ep_sums),
                bandit_zap_rate_ep_sum=jnp.sum(bandit_zap_rate_ep_sums),
                agent_rep_min_sum=jnp.sum(agent_rep_min_sums, axis=0),
                agent_rep_max_sum=jnp.sum(agent_rep_max_sums, axis=0),
                agent_rep_mean_sum=jnp.sum(agent_rep_mean_sums, axis=0),
                agent_zap_ok_ep_sum=jnp.sum(agent_zap_ok_ep_sums, axis=0),
                agent_zap_try_ep_sum=jnp.sum(agent_zap_try_ep_sums, axis=0),
                agent_zap_targets_hit_ep_sum=jnp.sum(
                    agent_zap_targets_hit_ep_sums, axis=0
                ),
                agent_tags_received_ep_sum=jnp.sum(agent_tags_received_ep_sums, axis=0),
                agent_harvest_rate_mean_sum=jnp.sum(
                    agent_harvest_rate_mean_sums, axis=0
                ),
                agent_harvest_rate_max_sum=jnp.sum(
                    agent_harvest_rate_max_sums, axis=0
                ),
                agent_burst_count_mean_sum=jnp.sum(
                    agent_burst_count_mean_sums, axis=0
                ),
                agent_burst_count_max_sum=jnp.sum(
                    agent_burst_count_max_sums, axis=0
                ),
                agent_burst_penalty_mean_sum=jnp.sum(
                    agent_burst_penalty_mean_sums, axis=0
                ),
                agent_burst_penalty_max_sum=jnp.sum(
                    agent_burst_penalty_max_sums, axis=0
                ),
                agent_zap_target_rep_mean_sum=jnp.sum(
                    agent_zap_target_rep_mean_sums, axis=0
                ),
                agent_target_bad_rate_attempted_ep_sum=jnp.sum(
                    agent_target_bad_rate_attempted_ep_sums, axis=0
                ),
                agent_target_bad_rate_hit_ep_sum=jnp.sum(
                    agent_target_bad_rate_hit_ep_sums, axis=0
                ),
                agent_peer_beam_hits_good_ep_sum=jnp.sum(
                    agent_peer_beam_hits_good_ep_sums, axis=0
                ),
                agent_peer_beam_hits_bad_ep_sum=jnp.sum(
                    agent_peer_beam_hits_bad_ep_sums, axis=0
                ),
                agent_enforcement_hits_ep_sum=jnp.sum(
                    agent_enforcement_hits_ep_sums, axis=0
                ),
                agent_enforcement_rep_mean_sum=jnp.sum(
                    agent_enforcement_rep_mean_sums, axis=0
                ),
                agent_return_ep_sum=jnp.sum(agent_return_ep_sums, axis=0),
            )
            return (
                rng,
                env_state,
                obs_dict,
                traj,
                step_metrics,
                pe_acc,
                cum_returns,
                window,
            )

        def _calculate_gae(traj: Transition, last_val, gamma):
            """GAE over time; traj fields shaped (T, E), last_val (E,). ``gamma`` is traced."""
            return calculate_gae(
                traj.reward,
                traj.value,
                traj.done,
                last_val,
                gamma,
                gae_lambda,
                unroll=16,  # match foundation IPPO scan unroll
            )

        def _ppo_loss(params, obs, actions, log_probs_old, values_old, advantages, targets, clip_eps, ent_coef):
            pi, value = _apply_network(params, obs)
            log_prob = pi.log_prob(actions)
            entropy = pi.entropy().mean()
            return ppo_clip_loss_terms(
                log_prob,
                log_probs_old,
                value,
                values_old,
                advantages,
                targets,
                entropy,
                clip_eps,
                ent_coef,
                vf_coef,
            )

        def _update_agent(
            params, opt_state, step, traj_agent, advantages, targets, clip_eps, ent_coef, rng
        ):
            """Single-agent PPO update (vmapped over agents). traj: (T, E, ...)."""
            batch_size = num_steps * num_envs

            def _flatten(x):
                return x.reshape((batch_size,) + x.shape[2:])

            obs = _flatten(traj_agent.obs)
            actions = _flatten(traj_agent.action)
            log_probs = _flatten(traj_agent.log_prob)
            values = _flatten(traj_agent.value)
            adv = advantages.reshape((batch_size,))
            targ = targets.reshape((batch_size,))

            def _epoch(carry, _):
                params, opt_state, step, rng = carry
                rng, perm_key = jax.random.split(rng)
                permutation = jax.random.permutation(perm_key, batch_size)
                batch = (obs, actions, log_probs, values, adv, targ)
                batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape((num_minibatches, minibatch_size) + x.shape[1:]),
                    batch,
                )

                def _minibatch(carry, mb):
                    params, opt_state, step = carry
                    mb_obs, mb_a, mb_lp, mb_v, mb_adv, mb_targ = mb

                    def loss_fn(p):
                        return _ppo_loss(
                            p,
                            mb_obs,
                            mb_a,
                            mb_lp,
                            mb_v,
                            mb_adv,
                            mb_targ,
                            clip_eps,
                            ent_coef,
                        )

                    (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
                    updates, opt_state = shared_tx.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    step = step + 1
                    return (params, opt_state, step), (total, aux)

                (params, opt_state, step), loss_info = jax.lax.scan(
                    _minibatch, (params, opt_state, step), minibatches
                )
                return (params, opt_state, step, rng), loss_info

            (params, opt_state, step, rng), loss_info = jax.lax.scan(
                _epoch, (params, opt_state, step, rng), None, length=update_epochs
            )
            # Mean over UPDATE_EPOCHS × NUM_MINIBATCHES (not a single minibatch).
            mean_loss = loss_info[0].mean()
            mean_vloss = loss_info[1][0].mean()
            mean_aloss = loss_info[1][1].mean()
            mean_ent = loss_info[1][2].mean()
            return (
                params,
                opt_state,
                step,
                mean_loss,
                mean_vloss,
                mean_aloss,
                mean_ent,
                rng,
            )

        def _maybe_anneal_lrs(opt_state, steps, lrs):
            if not anneal_lr:
                return opt_state
            # steps stay in sync across agents; use agent-0 count for frac.
            # Denominator may be longer than this run (ANNEAL_LR_TOTAL_TIMESTEPS).
            count = steps[0]
            frac = (
                1.0
                - (count // (num_minibatches * update_epochs))
                / max(1, config["ANNEAL_LR_NUM_UPDATES"])
            )
            return _set_chain_lrs(opt_state, lrs * frac)

        def _maybe_anneal_ent_coefs(ent_coefs, steps):
            """Linear ENT_COEF → ENT_COEF_END over ANNEAL_ENT_NUM_UPDATES; hold."""
            if not anneal_ent:
                return ent_coefs
            count = steps[0]
            progress = jnp.clip(
                (count // (num_minibatches * update_epochs)).astype(jnp.float32)
                / jnp.float32(max(1, config["ANNEAL_ENT_NUM_UPDATES"])),
                jnp.float32(0.0),
                jnp.float32(1.0),
            )
            end = jnp.full_like(ent_coefs, jnp.float32(ent_coef_end))
            return ent_coefs + (end - ent_coefs) * progress

        def _learn(rng, params, opt_state, steps, traj, last_obs_dict, gamma, clip_epses, ent_coefs, lrs):
            # Write traced LRs into Adam hyperparams so LR is not a jit constant.
            opt_state = _set_chain_lrs(opt_state, lrs)
            opt_state = _maybe_anneal_lrs(opt_state, steps, lrs)
            ent_coefs = _maybe_anneal_ent_coefs(ent_coefs, steps)
            last_obs = _stack_agent_dim(last_obs_dict, agents)  # (E, N, ...)
            last_obs_ne = jnp.swapaxes(last_obs, 0, 1)  # (N, E, ...)

            def _last_val(params_i, obs_i):
                _, v = _apply_network(params_i, obs_i)
                return v

            last_val = jax.vmap(_last_val)(params, last_obs_ne)  # (N, E)
            traj_n = _traj_to_agent_major(traj)  # (N, T, E, ...)

            def _gae_one(traj_i, last_val_i):
                return _calculate_gae(traj_i, last_val_i, gamma)

            advantages, targets = jax.vmap(_gae_one)(traj_n, last_val)

            rng, *agent_rngs = jax.random.split(rng, num_agents + 1)
            agent_rngs = jnp.stack(agent_rngs)

            params_pre = params
            opt_state_pre = opt_state
            steps_pre = steps

            def _update_one(
                params_i, opt_i, step_i, traj_i, adv_i, targ_i, clip_i, ent_i, rng_i
            ):
                (
                    params_i,
                    opt_i,
                    step_i,
                    mean_loss_i,
                    mean_v_i,
                    mean_a_i,
                    mean_e_i,
                    rng_i,
                ) = _update_agent(
                    params_i, opt_i, step_i, traj_i, adv_i, targ_i, clip_i, ent_i, rng_i
                )
                return (
                    params_i,
                    opt_i,
                    step_i,
                    mean_loss_i,
                    mean_v_i,
                    mean_a_i,
                    mean_e_i,
                    rng_i,
                )

            (
                params,
                opt_state,
                steps,
                loss_vec,
                vloss_vec,
                aloss_vec,
                ent_vec,
                _agent_rngs,
            ) = jax.vmap(_update_one)(
                params,
                opt_state,
                steps,
                traj_n,
                advantages,
                targets,
                clip_epses,
                ent_coefs,
                agent_rngs,
            )
            if freeze_police:
                params = restore_frozen_police_slices(
                    params, params_pre, police_mask, num_seeds=1
                )
                opt_state = restore_frozen_police_slices(
                    opt_state, opt_state_pre, police_mask, num_seeds=1
                )
                steps = restore_frozen_police_slices(
                    steps, steps_pre, police_mask, num_seeds=1
                )
            # Scalars: mean over agents. loss_vec kept for per-agent WandB.
            return (
                rng,
                params,
                opt_state,
                steps,
                loss_vec,
                loss_vec.mean(),
                vloss_vec.mean(),
                aloss_vec.mean(),
                ent_vec.mean(),
            )

        def _states_dict(params, opt_state, steps, seed_i: int | None = None):
            if seed_i is not None:
                params = jax.tree_util.tree_map(lambda x: x[seed_i], params)
                opt_state = jax.tree_util.tree_map(lambda x: x[seed_i], opt_state)
                steps = steps[seed_i]
            return _unstack_train_states(
                params,
                opt_state,
                steps,
                agent_keys=agent_keys,
                apply_fn=network.apply,
                tx=shared_tx,
            )

        def _slice_step_metrics(step_metrics, seed_i: int):
            return jax.tree_util.tree_map(
                lambda x: jnp.asarray(jax.device_get(x))[seed_i],
                step_metrics,
            )

        _trace_count, _on_trace_update_chunk = make_trace_counter("_run_chunk")
        shape_tracker = ShapeSignatureTracker()
        shape_sig = (
            num_envs,
            num_steps,
            num_agents,
            num_seeds,
            tuple(obs_shape),
        )

        def _single_update(
            rng,
            params,
            opt_state,
            steps,
            env_state,
            obs_dict,
            pe_acc,
            ep_life,
            cum_returns,
            update_i,
            gamma,
            clip_epses,
            ent_coefs,
            lrs,
        ):
            (
                rng,
                env_state,
                obs_dict,
                traj,
                step_metrics,
                pe_acc,
                cum_returns,
                window,
            ) = _collect_rollout(
                rng, params, env_state, obs_dict, pe_acc, cum_returns, update_i
            )
            rng, params, opt_state, steps, loss_vec, mean_loss, mean_vloss, mean_aloss, mean_ent = _learn(
                rng, params, opt_state, steps, traj, obs_dict,
                gamma, clip_epses, ent_coefs, lrs,
            )
            ep_life = add_episode_aggregates(ep_life, window)
            return (
                rng,
                params,
                opt_state,
                steps,
                env_state,
                obs_dict,
                pe_acc,
                ep_life,
                cum_returns,
                mean_loss,
                loss_vec,
                mean_vloss,
                mean_aloss,
                mean_ent,
                step_metrics,
            )

        def _run_chunk(
            rng,
            params,
            opt_state,
            steps,
            env_state,
            obs_dict,
            pe_acc,
            ep_life,
            cum_returns,
            update_i,
            gamma,
            clip_epses,
            ent_coefs,
            lrs,
            chunk_len,
        ):
            # Placeholder metrics in carry so scan does not stack chunk×T×E tensors.
            sm0 = StepMetrics(
                env_rewards=jnp.zeros(
                    (num_steps, num_envs, num_agents), dtype=jnp.float32
                ),
                total_rewards=jnp.zeros(
                    (num_steps, num_envs, num_agents), dtype=jnp.float32
                ),
                tagged=jnp.zeros(
                    (num_steps, num_envs, num_agents), dtype=jnp.bool_
                ),
                zap_successful=jnp.zeros(
                    (num_steps, num_envs, num_agents), dtype=jnp.bool_
                ),
                zap_attempted=jnp.zeros(
                    (num_steps, num_envs, num_agents), dtype=jnp.bool_
                ),
                inner_episode_done=jnp.zeros(
                    (num_steps, num_envs, num_agents), dtype=jnp.bool_
                ),
                returned_episode=jnp.zeros(
                    (num_steps, num_envs, num_agents), dtype=jnp.bool_
                ),
                returned_episode_returns=jnp.zeros(
                    (num_steps, num_envs, num_agents), dtype=jnp.float32
                ),
                mean_q=jnp.zeros((num_steps, action_dim), dtype=jnp.float32),
                greedy_zap_rate=jnp.zeros((num_steps,), dtype=jnp.float32),
                mean_q_zap=jnp.zeros((num_steps,), dtype=jnp.float32),
                rep=jnp.zeros(
                    (num_steps, num_envs, num_agents), dtype=jnp.float32
                ),
                raw_rep=jnp.zeros(
                    (num_steps, num_envs, num_agents), dtype=jnp.float32
                ),
                tags_received=jnp.zeros(
                    (num_steps, num_envs, num_agents), dtype=jnp.float32
                ),
            )
            carry0 = (
                rng,
                params,
                opt_state,
                steps,
                env_state,
                obs_dict,
                pe_acc,
                ep_life,
                cum_returns,
                update_i,
                sm0,
            )

            def _chunk_step(carry, _):
                (
                    rng,
                    params,
                    opt_state,
                    steps,
                    env_state,
                    obs_dict,
                    pe_acc,
                    ep_life,
                    cum_returns,
                    update_i,
                    _sm,
                ) = carry
                if debug_perf:
                    _on_trace_update_chunk()
                (
                    rng,
                    params,
                    opt_state,
                    steps,
                    env_state,
                    obs_dict,
                    pe_acc,
                    ep_life,
                    cum_returns,
                    mean_loss,
                    loss_vec,
                    mean_vloss,
                    mean_aloss,
                    mean_ent,
                    step_metrics,
                ) = _single_update(
                    rng,
                    params,
                    opt_state,
                    steps,
                    env_state,
                    obs_dict,
                    pe_acc,
                    ep_life,
                    cum_returns,
                    update_i,
                    gamma,
                    clip_epses,
                    ent_coefs,
                    lrs,
                )
                new_carry = (
                    rng,
                    params,
                    opt_state,
                    steps,
                    env_state,
                    obs_dict,
                    pe_acc,
                    ep_life,
                    cum_returns,
                    update_i + jnp.int32(1),
                    step_metrics,
                )
                return new_carry, (mean_loss, loss_vec, mean_vloss, mean_aloss, mean_ent)

            carry, (losses, loss_vecs, vlosses, alosses, ents) = jax.lax.scan(
                _chunk_step, carry0, None, length=chunk_len
            )
            (
                rng,
                params,
                opt_state,
                steps,
                env_state,
                obs_dict,
                pe_acc,
                ep_life,
                cum_returns,
                update_i,
                step_metrics,
            ) = carry
            # Mean over the JIT chunk (all UPDATES_PER_JIT PPO updates), not only
            # the last update — that single snapshot was spike-noisy in WandB.
            return (
                rng,
                params,
                opt_state,
                steps,
                env_state,
                obs_dict,
                pe_acc,
                ep_life,
                cum_returns,
                update_i,
                losses.mean(),
                loss_vecs.mean(axis=0),
                vlosses.mean(),
                alosses.mean(),
                ents.mean(),
                step_metrics,
            )

        if verbose_logging:
            print(
                f"[perf] about to jax.jit(_run_chunk); "
                f"static shape signature={shape_sig}; "
                f"updates_per_jit={updates_per_jit}; num_seeds={num_seeds}"
            )
            shape_tracker.note("pre-jit", shape_sig)
        # Traced hparams: in_axes None (broadcast over seeds). chunk_len is static.
        _hparams_in_axes = (None, None, None, None)
        if num_seeds == 1:
            if verbose_logging:
                print(
                    f"[perf] calling jax.jit(_run_chunk); "
                    f"updates_per_jit={updates_per_jit} (host sync every chunk)"
                )
            run_chunk = jax.jit(_run_chunk, static_argnames=("chunk_len",))
        else:
            if verbose_logging:
                print(
                    f"[perf] calling jax.jit(jax.vmap(_run_chunk)); "
                    f"NUM_SEEDS={num_seeds}, updates_per_jit={updates_per_jit}"
                )
            run_chunk = jax.jit(
                jax.vmap(
                    _run_chunk,
                    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0) + _hparams_in_axes + (None,),
                ),
                static_argnames=("chunk_len",),
            )

        num_updates = config["NUM_UPDATES"]
        timesteps_per_update = num_steps * num_envs
        if num_seeds == 1:
            update_i = jnp.int32(0)
        else:
            update_i = jnp.zeros((num_seeds,), dtype=jnp.int32)

        def _z() -> onp.ndarray:
            return onp.zeros((num_seeds,), dtype=onp.float64)

        def _seed_vec(val) -> onp.ndarray:
            """Scalar episode-life baseline broadcast across seeds (resume / W&B)."""
            return onp.full((num_seeds,), float(val), dtype=onp.float64)

        last_log = {
            "R": _z(),
            "S": _z(),
            "P": _z(),
            "P_sys": _z(),
            "P_police": _z(),
            "E": _z(),
            "E_total": _z(),
            "count": _z(),
            "apple_sum": _z(),
            "apple_count": _z(),
            "zap_success_rate_sum": _z(),
            "zap_attempt_rate_sum": _z(),
            "zap_hit_rate_sum": _z(),
            "enforcement_hit_rate_sum": _z(),
            "enforcement_hit_count_sum": _z(),
            "enforcement_rep_total_sum": _z(),
            "zap_tagged_hit_count_sum": _z(),
            "zap_tagged_rep_total_sum": _z(),
            "police_zap_tagged_hit_count_sum": _z(),
            "police_zap_tagged_rep_total_sum": _z(),
            "apple_end_sum": _z(),
            "apple_late_mean_sum": _z(),
            "barren_t_sum": _z(),
            "barren_event_sum": _z(),
            "tagged_out_frac_sum": _z(),
            "enforcement_tagged_out_frac_sum": _z(),
            "police_tagged_out_frac_sum": _z(),
            "police_zap_attempt_rate_sum": _z(),
            "police_zap_hit_rate_sum": _z(),
            "police_reward_sum": _z(),
            "reward_variance_across_agents_sum": _z(),
            "base_rate_bad_ep_sum": _z(),
            "peer_zap_timeouts_ep_sum": _z(),
            "enforcement_timeouts_ep_sum": _z(),
            "enforcement_time_to_freeze_ep_sum": _z(),
            "bad_survival_ep_sum": _z(),
            "good_survival_ep_sum": _z(),
            "bandit_zap_count_ep_sum": _z(),
            "bandit_zap_rate_ep_sum": _z(),
        }
        _agent_ep_fields = (
            "agent_rep_min_sum",
            "agent_rep_max_sum",
            "agent_rep_mean_sum",
            "agent_harvest_rate_mean_sum",
            "agent_harvest_rate_max_sum",
            "agent_burst_count_mean_sum",
            "agent_burst_count_max_sum",
            "agent_burst_penalty_mean_sum",
            "agent_burst_penalty_max_sum",
            "agent_zap_ok_ep_sum",
            "agent_zap_try_ep_sum",
            "agent_zap_targets_hit_ep_sum",
            "agent_tags_received_ep_sum",
            "agent_zap_target_rep_mean_sum",
            "agent_target_bad_rate_attempted_ep_sum",
            "agent_target_bad_rate_hit_ep_sum",
            "agent_peer_beam_hits_good_ep_sum",
            "agent_peer_beam_hits_bad_ep_sum",
            "agent_enforcement_hits_ep_sum",
            "agent_enforcement_rep_mean_sum",
            "agent_return_ep_sum",
        )
        last_log_agent = {
            f: onp.zeros((num_agents,), dtype=onp.float64) for f in _agent_ep_fields
        }

        def _log(
            update,
            mean_loss,
            loss_vec,
            mean_vloss,
            mean_aloss,
            mean_ent,
            step_metrics,
            ep_life,
            perf_extra=None):
            nonlocal last_log_agent
            life_count = _as_seed_vec(ep_life.count, num_seeds)
            n_since = life_count - last_log["count"]

            def _interval_mean(field):
                cur = _as_seed_vec(getattr(ep_life, field), num_seeds)
                val = _interval_mean_vec(cur, last_log[field], n_since)
                last_log[field] = cur
                return val

            R_v = _interval_mean("R")
            S_v = _interval_mean("S")
            P_v = _interval_mean("P")
            P_sys_v = _interval_mean("P_sys")
            P_police_v = _interval_mean("P_police")
            E_v = _interval_mean("E")
            E_total_v = _interval_mean("E_total")
            mean_ep_return_v = R_v
            last_log["count"] = life_count

            apple_sum = _as_seed_vec(ep_life.apple_sum, num_seeds)
            apple_count = _as_seed_vec(ep_life.apple_count, num_seeds)
            apple_n_since = apple_count - last_log["apple_count"]
            mean_apple_count_v = onp.full((num_seeds), onp.nan, dtype=onp.float64)
            mask_a = apple_n_since > 0
            mean_apple_count_v[mask_a] = (
                (apple_sum - last_log["apple_sum"])[mask_a] / apple_n_since[mask_a]
            )
            last_log["apple_sum"] = apple_sum
            last_log["apple_count"] = apple_count

            apple_end_v = _interval_mean("apple_end_sum")
            apple_late_v = _interval_mean("apple_late_mean_sum")
            barren_t_v = _interval_mean("barren_t_sum")
            barren_frac_v = _interval_mean("barren_event_sum")
            zap_success_rate_v = _interval_mean("zap_success_rate_sum")
            zap_attempt_rate_v = _interval_mean("zap_attempt_rate_sum")
            zap_hit_rate_v = _interval_mean("zap_hit_rate_sum")
            enforcement_hit_rate_v = _interval_mean("enforcement_hit_rate_sum")
            enf_hit_count_cur = _as_seed_vec(
                ep_life.enforcement_hit_count_sum, num_seeds
            )
            enf_rep_total_cur = _as_seed_vec(
                ep_life.enforcement_rep_total_sum, num_seeds
            )
            enf_hit_delta = enf_hit_count_cur - last_log["enforcement_hit_count_sum"]
            enf_rep_delta = enf_rep_total_cur - last_log["enforcement_rep_total_sum"]
            enforcement_rep_mean_v = onp.where(
                enf_hit_delta > 0,
                enf_rep_delta / enf_hit_delta,
                0.0,
            )
            # Mean enforcement timeouts per completed episode (all agents).
            enforcement_hits_v = _interval_mean_vec(
                enf_hit_count_cur, last_log["enforcement_hit_count_sum"], n_since
            )
            last_log["enforcement_hit_count_sum"] = enf_hit_count_cur
            last_log["enforcement_rep_total_sum"] = enf_rep_total_cur

            # Hit-weighted mean reputation of zap-tagged victims (harvesters only).
            zap_tag_hit_cur = _as_seed_vec(
                ep_life.zap_tagged_hit_count_sum, num_seeds
            )
            zap_tag_rep_cur = _as_seed_vec(
                ep_life.zap_tagged_rep_total_sum, num_seeds
            )
            zap_tag_hit_delta = zap_tag_hit_cur - last_log["zap_tagged_hit_count_sum"]
            zap_tag_rep_delta = zap_tag_rep_cur - last_log["zap_tagged_rep_total_sum"]
            zap_tagged_rep_mean_v = onp.where(
                zap_tag_hit_delta > 0,
                zap_tag_rep_delta / zap_tag_hit_delta,
                onp.nan,
            )
            last_log["zap_tagged_hit_count_sum"] = zap_tag_hit_cur
            last_log["zap_tagged_rep_total_sum"] = zap_tag_rep_cur

            # Police shooters only → enforcement/police_zap_tagged_rep_mean.
            pol_tag_hit_cur = _as_seed_vec(
                ep_life.police_zap_tagged_hit_count_sum, num_seeds
            )
            pol_tag_rep_cur = _as_seed_vec(
                ep_life.police_zap_tagged_rep_total_sum, num_seeds
            )
            pol_tag_hit_delta = (
                pol_tag_hit_cur - last_log["police_zap_tagged_hit_count_sum"]
            )
            pol_tag_rep_delta = (
                pol_tag_rep_cur - last_log["police_zap_tagged_rep_total_sum"]
            )
            police_zap_tagged_rep_mean_v = onp.where(
                pol_tag_hit_delta > 0,
                pol_tag_rep_delta / pol_tag_hit_delta,
                onp.nan,
            )
            last_log["police_zap_tagged_hit_count_sum"] = pol_tag_hit_cur
            last_log["police_zap_tagged_rep_total_sum"] = pol_tag_rep_cur

            tagged_out_frac_v = _interval_mean("tagged_out_frac_sum")
            enforcement_tagged_out_frac_v = _interval_mean(
                "enforcement_tagged_out_frac_sum"
            )
            police_tagged_out_frac_v = _interval_mean("police_tagged_out_frac_sum")
            police_zap_attempt_rate_v = _interval_mean("police_zap_attempt_rate_sum")
            police_zap_hit_rate_v = _interval_mean("police_zap_hit_rate_sum")
            police_reward_v = _interval_mean("police_reward_sum")
            reward_var_v = _interval_mean("reward_variance_across_agents_sum")
            base_rate_bad_v = _interval_mean("base_rate_bad_ep_sum")
            peer_zap_timeouts_v = _interval_mean("peer_zap_timeouts_ep_sum")
            enforcement_timeouts_v = _interval_mean("enforcement_timeouts_ep_sum")
            enforcement_time_to_freeze_v = _interval_mean(
                "enforcement_time_to_freeze_ep_sum"
            )
            bad_survival_v = _interval_mean("bad_survival_ep_sum")
            good_survival_v = _interval_mean("good_survival_ep_sum")
            bandit_zap_count_v = _interval_mean("bandit_zap_count_ep_sum")
            bandit_zap_rate_v = _interval_mean("bandit_zap_rate_ep_sum")

            # Seed-0 only for non-ToC payload (apples / zap rates).
            # The five ToC series below use pack_per_seed_series (all replicas).
            def _run_scalar(vec) -> float:
                return float(_as_seed_vec(vec, num_seeds)[0])

            env_m = {
                "apple_sum": _run_scalar(mean_apple_count_v),
                "barren_event_sum": _run_scalar(barren_frac_v),
                "apple_end_sum": _run_scalar(apple_end_v),
                "apple_late_mean_sum": _run_scalar(apple_late_v),
                "reward_variance_across_agents": _run_scalar(reward_var_v),
                "zap_attempt_rate": _run_scalar(zap_attempt_rate_v),
                "zap_success_rate": _run_scalar(zap_success_rate_v),
                "zap_hit_rate": _run_scalar(zap_hit_rate_v),
                "enforcement_hit_rate": _run_scalar(enforcement_hit_rate_v),
                "enforcement_rep_mean": _run_scalar(enforcement_rep_mean_v),
                "enforcement_hits": _run_scalar(enforcement_hits_v),
                "enforcement_tagged_out_frac": _run_scalar(
                    enforcement_tagged_out_frac_v
                ),
                "enforcement/police_peace": _run_scalar(P_police_v),
                "enforcement/police_tagged_out_frac": _run_scalar(
                    police_tagged_out_frac_v
                ),
                "enforcement/police_zap_attempt_rate": _run_scalar(
                    police_zap_attempt_rate_v
                ),
                "enforcement/police_zap_hit_rate": _run_scalar(police_zap_hit_rate_v),
                "enforcement/police_reward": _run_scalar(police_reward_v),
                "enforcement/police_zap_tagged_rep_mean": _run_scalar(
                    police_zap_tagged_rep_mean_v
                ),
                "zap_tagged_rep_mean": _run_scalar(zap_tagged_rep_mean_v),
                "base_rate_bad_ep": _run_scalar(base_rate_bad_v),
                "peer_zap_timeouts_ep": _run_scalar(peer_zap_timeouts_v),
                "enforcement_timeouts_ep": _run_scalar(enforcement_timeouts_v),
                "enforcement_time_to_freeze_ep": _run_scalar(
                    enforcement_time_to_freeze_v
                ),
                "bad_survival_ep": _run_scalar(bad_survival_v),
                "good_survival_ep": _run_scalar(good_survival_v),
                "bandit_zap_count": _run_scalar(bandit_zap_count_v),
                "bandit_zap_rate": _run_scalar(bandit_zap_rate_v),
                "n_episodes_completed": int(n_since[0]),
                "n_episodes_lifetime": int(life_count[0]),
            }
            # peace = agent-beam timeouts; system_peace = enforcement timeouts.
            toc_vecs = {
                "mean_reward": R_v,
                "peace": P_v,
                "system_peace": P_sys_v,
                "sustainability": S_v,
                "equality": E_v,
                "tagged_out_frac": tagged_out_frac_v,
            }
            for _name, _vec in toc_vecs.items():
                env_m.update(
                    pack_per_seed_series(_name, _as_seed_vec(_vec, num_seeds), num_seeds)
                )

            if optimize_baseline and step_metrics is not None:
                sm_baseline = (
                    step_metrics
                    if num_seeds == 1
                    else _slice_step_metrics(step_metrics, 0)
                )
                env_m.update(_per_agent_env_r_from_rollout(sm_baseline, agents))

            if log_per_agent:
                sm_0 = (
                    step_metrics
                    if num_seeds == 1
                    else _slice_step_metrics(step_metrics, 0)
                )
                per_agent = metrics_from_rollout_ippo(
                    sm_0,
                    num_agents,
                    agents,
                    R=float(R_v[0]),
                    S=float(S_v[0]),
                    P=float(P_v[0]),
                    E=float(E_v[0]),
                    E_total=float(E_total_v[0]),
                    mean_episode_return=float(mean_ep_return_v[0]),
                    zap_success_rate=float(zap_success_rate_v[0]),
                    zap_attempt_rate=float(zap_attempt_rate_v[0]),
                    zap_hit_rate=float(zap_hit_rate_v[0]),
                    n_episodes_since_log=int(n_since[0]),
                    n_episodes_lifetime=int(life_count[0]))
                for key, val in per_agent.items():
                    if not key.startswith("agent_"):
                        continue
                    # Reputation runs: keep episode metrics only in W&B — skip
                    # noisy rollout-window agent_* keys from the rollout packer.
                    if log_reputation and _is_agent_window_wandb_key(key):
                        continue
                    env_m[key] = float(val)
                if log_reputation and step_metrics is not None:
                    ep_agent_metrics, last_log_agent = (
                        _per_agent_episode_metrics_for_wandb(
                            ep_life,
                            last_log_agent,
                            float(n_since[0]),
                            agents,
                            seed_i=0)
                    )
                    env_m.update(ep_agent_metrics)
                    # Episode-average rep (harvesters), not NUM_STEPS rollout window.
                    mean_rep_ep = _harvester_mean_rep_from_ep_metrics(
                        env_m, agents, police_mask
                    )
                    env_m["mean_rep"] = mean_rep_ep
                    env_m["mean_raw_rep"] = mean_rep_ep
            elif log_reputation and step_metrics is not None:
                # Episode-true agent metrics only (no *_window rollout pack).
                ep_agent_metrics, last_log_agent = _per_agent_episode_metrics_for_wandb(
                    ep_life,
                    last_log_agent,
                    float(n_since[0]),
                    agents,
                    seed_i=0)
                env_m.update(ep_agent_metrics)
                mean_rep_ep = _harvester_mean_rep_from_ep_metrics(
                    env_m, agents, police_mask
                )
                env_m["mean_rep"] = mean_rep_ep
                env_m["mean_raw_rep"] = mean_rep_ep

            loss_v = _as_seed_vec(mean_loss, num_seeds)
            loss_mean = float(loss_v[0])
            vloss_mean = float(_as_seed_vec(mean_vloss, num_seeds)[0])
            aloss_mean = float(_as_seed_vec(mean_aloss, num_seeds)[0])
            ent_mean = float(_as_seed_vec(mean_ent, num_seeds)[0])
            n_since_sum = int(n_since[0])
            assert_finite_metrics(
                env_m,
                loss_mean,
                update,
                agents,
                n_episodes_since_log=n_since_sum)

            r_str = _fmt_metric(env_m.get("mean_reward", env_m.get("mean_reward/seed_0", float("nan"))), ".2f")
            apple_str = _fmt_metric(env_m["apple_sum"], ".1f")
            bfrac_str = _fmt_metric(env_m["barren_event_sum"], ".2f")
            s_str = _fmt_metric(env_m.get("sustainability", env_m.get("sustainability/seed_0", float("nan"))), ".1f")
            e_str = _fmt_metric(env_m.get("equality", env_m.get("equality/seed_0", float("nan"))), ".3f")
            p_str = _fmt_metric(env_m.get("peace", env_m.get("peace/seed_0", float("nan"))), ".2f")
            psys_str = _fmt_metric(
                env_m.get("system_peace", env_m.get("system_peace/seed_0", float("nan"))),
                ".2f",
            )
            t_str = _fmt_metric(env_m.get("tagged_out_frac", env_m.get("tagged_out_frac/seed_0", float("nan"))), ".3f")
            seed_tag = f" NUM_SEEDS={num_seeds}" if num_seeds > 1 else ""
            print(
                f"update {update}/{num_updates}{seed_tag} "
                f"episode_count={env_m['n_episodes_lifetime']} "
                f"mean_reward={r_str} apple_sum={apple_str} "
                f"barren_event_sum={bfrac_str} S={s_str} E={e_str} "
                f"P={p_str} tagged_out_frac={t_str} "
                f"(+{env_m['n_episodes_completed']} since log)"
            )
            if enable_reputation:
                enf_rate_str = _fmt_metric(env_m.get("enforcement_hit_rate", float("nan")), ".4f")
                enf_rep_str = _fmt_metric(env_m.get("enforcement_rep_mean", float("nan")), ".3f")
                enf_hits_str = _fmt_metric(env_m.get("enforcement_hits", float("nan")), ".2f")
                enf_tof_str = _fmt_metric(
                    env_m.get("enforcement_tagged_out_frac", float("nan")), ".3f"
                )
                print(
                    f"  system_peace={psys_str} "
                    f"enforcement_tagged_out_frac={enf_tof_str} "
                    f"enforcement_hit_rate={enf_rate_str} "
                    f"enforcement_hits/ep={enf_hits_str} "
                    f"enforcement_rep_mean={enf_rep_str}"
                )
            if log_reputation:
                mean_rep_str = _fmt_metric(env_m.get("mean_rep", float("nan")), ".3f")
                ztr_str = _fmt_metric(
                    env_m.get("zap_tagged_rep_mean", float("nan")), ".3f"
                )
                print(
                    f"  mean_rep={mean_rep_str} "
                    f"zap_tagged_rep_mean={ztr_str}"
                )
            if num_seeds > 1:
                parts = []
                for _name in _PER_SEED_TOC_METRICS:
                    vals = [
                        _fmt_metric(env_m[f"{_name}/seed_{i}"], ".3g")
                        for i in range(num_seeds)
                    ]
                    parts.append(f"{_name}=[{','.join(vals)}]")
                print(f"  per-seed: {' '.join(parts)}")
            if use_metric_logging:
                # Episode-true ToC / rates are NaN when no inner episode finished
                # this interval. Omit them (do not log NaN) so W&B step_metric=
                # episode_count cannot overwrite the last finite point — this
                # happens every other update with UPDATES_PER_JIT=1 and
                # NUM_STEPS < num_inner_steps (N=10 Perolat A100 setup).
                episodes_this_log = int(n_since[0]) > 0
                payload = {
                    "update": update,
                    # Lifetime inner-episode completions on replica 0 (shared
                    # W&B x-axis). Per-seed ToC series are metric/seed_i.
                    "episode_count": int(life_count[0]),
                    "apple_sum": env_m["apple_sum"],
                }
                if episodes_this_log:
                    payload["barren_event_sum"] = env_m["barren_event_sum"]
                    payload["apple_end_sum"] = env_m["apple_end_sum"]
                    payload["apple_late_mean_sum"] = env_m["apple_late_mean_sum"]
                    payload["reward_variance_across_agents"] = env_m[
                        "reward_variance_across_agents"
                    ]
                    payload["zap_attempt_rate"] = env_m["zap_attempt_rate"]
                    payload["zap_success_rate"] = env_m["zap_success_rate"]
                    payload["zap_hit_rate"] = env_m["zap_hit_rate"]
                    if enable_reputation:
                        payload["enforcement_hit_rate"] = env_m["enforcement_hit_rate"]
                        payload["enforcement_rep_mean"] = env_m["enforcement_rep_mean"]
                        payload["enforcement_hits"] = env_m["enforcement_hits"]
                        payload["enforcement_tagged_out_frac"] = env_m[
                            "enforcement_tagged_out_frac"
                        ]
                    for _name in _PER_SEED_TOC_METRICS:
                        payload.update(
                            {
                                k: env_m[k]
                                for k in env_m
                                if k == _name or k.startswith(f"{_name}/seed_")
                            }
                        )
                if verbose_logging:
                    payload["loss"] = loss_mean
                    payload["value_loss"] = vloss_mean
                    payload["actor_loss"] = aloss_mean
                    payload["entropy"] = ent_mean
                    if perf_extra:
                        payload.update(perf_extra)
                if episodes_this_log and (log_per_agent or log_reputation):
                    for key in _wandb_agent_metric_keys(
                        env_m,
                        enable_reputation=log_reputation,
                        log_per_agent=log_per_agent):
                        payload[key] = env_m[key]
                if episodes_this_log and log_reputation and not log_per_agent:
                    if "mean_rep" in env_m:
                        payload["mean_rep"] = env_m["mean_rep"]
                        payload["mean_raw_rep"] = env_m["mean_raw_rep"]
                    # Harvester shooters only (exclude police agent_* rows).
                    _pm = onp.asarray(police_mask, dtype=bool).reshape(-1)

                    def _harvester_nanmean(suffix: str) -> float | None:
                        keys = []
                        for k in env_m:
                            if not (
                                k.startswith("agent_") and k.endswith(f"/{suffix}")
                            ):
                                continue
                            try:
                                aid = int(k.split("_", 1)[1].split("/", 1)[0])
                            except (ValueError, IndexError):
                                continue
                            if aid < _pm.size and bool(_pm[aid]):
                                continue
                            keys.append(k)
                        if not keys:
                            return None
                        vals = onp.asarray(
                            [float(env_m[k]) for k in keys], dtype=onp.float64
                        )
                        if not onp.isfinite(vals).any():
                            return None
                        return float(onp.nanmean(vals))

                    ztr = _harvester_nanmean("zap_target_rep_mean_ep")
                    if ztr is not None and onp.isfinite(ztr):
                        payload["zap_target_rep_mean_ep"] = ztr
                    tba = _harvester_nanmean("target_bad_rate_attempted_ep")
                    if tba is not None and onp.isfinite(tba):
                        payload["target_bad_rate_attempted_ep"] = tba
                    tbh = _harvester_nanmean("target_bad_rate_hit_ep")
                    if tbh is not None and onp.isfinite(tbh):
                        payload["target_bad_rate_hit_ep"] = tbh
                if episodes_this_log and log_reputation and "zap_tagged_rep_mean" in env_m:
                    # Hit-weighted mean reputation of zap victims (harvester shooters).
                    payload["zap_tagged_rep_mean"] = env_m["zap_tagged_rep_mean"]
                if episodes_this_log and log_reputation and "base_rate_bad_ep" in env_m:
                    payload["base_rate_bad_ep"] = env_m["base_rate_bad_ep"]
                if episodes_this_log and log_reputation and "peer_zap_timeouts_ep" in env_m:
                    payload["peer_zap_timeouts_ep"] = env_m["peer_zap_timeouts_ep"]
                # Resolved peer Bad-zap magnitude (anneal schedule) for W&B plots.
                if enable_reputation_v2:
                    try:
                        from harvest_socialjax.envs.reputation_v2 import (
                            resolve_peer_punish_magnitudes,
                        )

                        _rv = reputation_v2_config_from_dict(config)
                        if bool(_rv.peer_punish_reward_enabled):
                            _mult = max(int(_rv.schedule_env_multiplier), 1)
                            _per_env = wandb_ep_to_per_env_completed(
                                int(life_count[0]), _mult
                            )
                            _bad, _good = resolve_peer_punish_magnitudes(
                                jnp.int32(_per_env), _rv
                            )
                            payload["peer_punish_bad_reward_sched"] = float(
                                jax.device_get(_bad)
                            )
                            payload["peer_punish_good_penalty_sched"] = float(
                                jax.device_get(_good)
                            )
                    except Exception:
                        pass
                if episodes_this_log and log_reputation and "enforcement_timeouts_ep" in env_m:
                    payload["enforcement_timeouts_ep"] = env_m[
                        "enforcement_timeouts_ep"
                    ]
                if episodes_this_log and log_reputation and "enforcement_time_to_freeze_ep" in env_m:
                    payload["enforcement_time_to_freeze_ep"] = env_m[
                        "enforcement_time_to_freeze_ep"
                    ]
                if episodes_this_log and log_reputation and "bad_survival_ep" in env_m:
                    payload["bad_survival_ep"] = env_m["bad_survival_ep"]
                if episodes_this_log and log_reputation and "good_survival_ep" in env_m:
                    payload["good_survival_ep"] = env_m["good_survival_ep"]
                if episodes_this_log and "bandit_zap_count" in env_m:
                    payload["bandit_zap_count"] = env_m["bandit_zap_count"]
                if episodes_this_log and "bandit_zap_rate" in env_m:
                    payload["bandit_zap_rate"] = env_m["bandit_zap_rate"]
                # Police-only series (always under enforcement/*).
                if episodes_this_log:
                    for k, v in env_m.items():
                        if str(k).startswith("enforcement/"):
                            payload[k] = v
                if optimize_baseline:
                    payload["effective_equality_coef"] = float(
                        effective_equality_coef(
                            jnp.int32(update),
                            equality_coef_target,
                            equality_coef_warmup_frac,
                            num_updates_total,
                            delay_frac=equality_coef_delay_frac)
                    )
                    for agent in agents:
                        key = f"agent_{int(agent)}/R_window"
                        if key in env_m:
                            payload[key] = env_m[key]
                log_training_metrics(payload)
            return env_m

        def _maybe_checkpoint(update, params, opt_state, steps, ep_life=None):
            if ckpt_dir is None or checkpoint_every <= 0:
                return None
            if update % checkpoint_every != 0 and update != num_updates - 1:
                return None
            if num_seeds == 1:
                path = save_checkpoint(
                    ckpt_dir,
                    _states_dict(params, opt_state, steps),
                    update,
                    config,
                    ep_life=ep_life,
                )
                print(f"[checkpoint] saved {path}")
                return path
            for s in range(num_seeds):
                sub = ckpt_dir / f"seed_{s}"
                save_checkpoint(
                    sub,
                    _states_dict(params, opt_state, steps, seed_i=s),
                    update,
                    config,
                    ep_life=ep_life,
                )
            path0 = save_checkpoint(
                ckpt_dir,
                _states_dict(params, opt_state, steps, seed_i=0),
                update,
                config,
                ep_life=ep_life,
            )
            print(f"[checkpoint] saved {num_seeds} seeds; seed_0 → {path0}")
            return path0

        def _note_shapes(where: str):
            if debug_perf:
                shape_tracker.note(where, shape_sig)

        resume_ckpt_path = config.get("RESUME_CHECKPOINT") or os.environ.get(
            "RESUME_CHECKPOINT"
        )
        resume_payload = None
        start_update = 0
        if resume_ckpt_path:
            try:
                resume_payload = load_checkpoint(resume_ckpt_path)
                start_update = int(resume_payload["update"]) + 1
                r_params, r_opt_state, r_steps = restore_stacked_train_state(
                    resume_payload, agent_keys=agent_keys
                )
            except Exception as exc:
                print(
                    f"[resume] FATAL: failed to load {resume_ckpt_path}: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise
            params = r_params
            opt_state = r_opt_state
            steps = r_steps
            print(
                f"[resume] loaded {resume_ckpt_path} "
                f"(completed update {resume_payload['update']}, "
                f"resuming at {start_update}/{num_updates})"
            )
            if resume_payload.get("episode_life") and num_seeds == 1:
                try:
                    ep_life = _episode_life_from_dict(
                        resume_payload["episode_life"], num_agents=num_agents
                    )
                    for field in _EPISODE_LIFE_FIELDS:
                        last_log[field] = _seed_vec(
                            _checkpoint_scalar(
                                jax.device_get(getattr(ep_life, field)),
                                what=f"episode_life.{field}",
                            )
                        )
                except (TypeError, KeyError, ValueError) as exc:
                    print(
                        f"[resume] WARNING: episode_life restore failed ({exc}); "
                        "falling back to WANDB_RESUME_EPISODE_COUNT"
                    )
                    resume_ep = config.get("WANDB_RESUME_EPISODE_COUNT") or os.environ.get(
                        "WANDB_RESUME_EPISODE_COUNT"
                    )
                    if resume_ep is not None:
                        ep_life = init_episode_aggregates(num_agents).replace(
                            count=jnp.float32(resume_ep)
                        )
                        last_log["count"] = _seed_vec(resume_ep)
            if resume_payload.get("episode_life"):
                print(
                    f"[resume] episode_count={float(jax.device_get(ep_life.count)):.0f}"
                )
            if start_update >= num_updates:
                raise ValueError(
                    f"RESUME_CHECKPOINT already at update {resume_payload['update']} "
                    f">= NUM_UPDATES-1 ({num_updates - 1}); increase TOTAL_TIMESTEPS"
                )

        # ---- first chunk (len=1): compile + execute (skipped when resuming) ----
        timesteps_per_update_all = timesteps_per_update * num_seeds
        if start_update == 0:
            _note_shapes("update_0_pre")
            t0 = time.perf_counter()
            with log_compiles_first_update(debug_perf):
                (
                    rng,
                    params,
                    opt_state,
                    steps,
                    env_state,
                    obs_dict,
                    pe_acc,
                    ep_life,
                    cum_returns,
                    update_i,
                    mean_loss,
                    loss_vec,
                    mean_vloss,
                    mean_aloss,
                    mean_ent,
                    step_metrics,
                ) = run_chunk(
                    rng,
                    params,
                    opt_state,
                    steps,
                    env_state,
                    obs_dict,
                    pe_acc,
                    ep_life,
                    cum_returns,
                    update_i,
                    gamma,
                    clip_epses,
                    ent_coefs,
                    lrs,
                    1,
                )
            jax.block_until_ready(mean_loss)
            t_first = time.perf_counter() - t0
            if memory_probe:
                compile_stats = snapshot_device_memory()
                config.setdefault("_MEMORY_PROBE_STATS", {})["compile"] = compile_stats
                _print_memory_snapshot("COMPILE", compile_stats)
            sps_first = timesteps_per_update_all / t_first
            if verbose_logging:
                print(
                    f"[perf] compile+first_step_time={t_first:.3f}s "
                    f"steps_per_second={sps_first:.1f} "
                    f"(timesteps_per_update={timesteps_per_update}"
                    f"{f' ×{num_seeds} seeds' if num_seeds > 1 else ''})"
                )
                print(
                    f"[profile] first update (incl. compile): {t_first:.3f}s "
                    f"({sps_first:.1f} env-steps/s)"
                )
            if debug_perf:
                print(f"[perf] JIT trace count after first update: {_trace_count[0]}")

            perf0 = {
                "compile+first_step_time": float(t_first),
                "steps_per_second": float(sps_first),
            }
            if (not memory_probe) and 0 % log_every == 0:
                _log(
                    0,
                    mean_loss,
                    loss_vec,
                    mean_vloss,
                    mean_aloss,
                    mean_ent,
                    step_metrics,
                    ep_life,
                    perf_extra=perf0 if debug_perf else None,
                )
            if not memory_probe:
                _maybe_checkpoint(0, params, opt_state, steps, ep_life)
            if not memory_probe and gif_renderer is not None and should_render_gif(0, config):
                gif_renderer.render_ippo(
                    params, 0, network.apply, seed_i=0, label="ippo",
                )
            chunk_start = 1
        else:
            rng, reset_key = jax.random.split(rng)
            obs_dict, env_state = wrapped.batch_reset(reset_key)
            pe_acc = init_per_episode_acc(num_envs, num_agents, num_inner_steps)
            cum_returns = jnp.zeros((num_envs, num_agents), dtype=jnp.float32)
            if not resume_payload.get("episode_life"):
                resume_ep = config.get("WANDB_RESUME_EPISODE_COUNT") or os.environ.get(
                    "WANDB_RESUME_EPISODE_COUNT"
                )
                if resume_ep is not None:
                    ep_life = init_episode_aggregates(num_agents).replace(
                        count=jnp.float32(resume_ep)
                    )
                    last_log["count"] = _seed_vec(resume_ep)
            # Peer / bandit schedules live on env episodes_completed, which
            # batch_reset zeros. Seed from W&B episode_count so absolute
            # anneal thresholds (e.g. start=50k) still fire after resume.
            resume_ep_clock = config.get("WANDB_RESUME_EPISODE_COUNT") or os.environ.get(
                "WANDB_RESUME_EPISODE_COUNT"
            )
            if resume_ep_clock is None and resume_payload.get("episode_life"):
                try:
                    resume_ep_clock = float(
                        jax.device_get(ep_life.count)
                    )
                except Exception:
                    resume_ep_clock = None
            if resume_ep_clock is not None:
                sched_mult = max(1, int(config.get("NUM_ENVS", 1)))
                if enable_reputation_v2:
                    try:
                        _rv = reputation_v2_config_from_dict(config)
                        sched_mult = max(int(_rv.schedule_env_multiplier), 1)
                    except Exception:
                        pass
                per_env_eps = wandb_ep_to_per_env_completed(
                    int(resume_ep_clock), sched_mult
                )
                env_state = seed_schedule_episode_clock(env_state, per_env_eps)
                print(
                    f"[resume] seeded schedule clock: wandb_ep={int(resume_ep_clock)} "
                    f"→ per_env_completed={per_env_eps} "
                    f"(×mult={sched_mult} → global≈{per_env_eps * sched_mult})"
                )
            if resume_payload.get("steps") is None:
                completed = int(resume_payload["update"]) + 1
                steps = jnp.full(
                    (num_agents,),
                    completed * update_epochs * num_minibatches,
                    dtype=jnp.int32,
                )
            chunk_start = start_update

        t_steady_start = time.perf_counter()
        steady_updates = 0
        while chunk_start < num_updates:
            chunk_len = min(updates_per_jit, num_updates - chunk_start)
            _note_shapes(f"chunk_{chunk_start}_pre")
            t_u0 = time.perf_counter()
            if num_seeds == 1:
                update_i_in = jnp.int32(chunk_start)
            else:
                update_i_in = jnp.full((num_seeds,), chunk_start, dtype=jnp.int32)
            (
                rng,
                params,
                opt_state,
                steps,
                env_state,
                obs_dict,
                pe_acc,
                ep_life,
                cum_returns,
                update_i,
                mean_loss,
                loss_vec,
                mean_vloss,
                mean_aloss,
                mean_ent,
                step_metrics,
            ) = run_chunk(
                rng,
                params,
                opt_state,
                steps,
                env_state,
                obs_dict,
                pe_acc,
                ep_life,
                cum_returns,
                update_i_in,
                gamma,
                clip_epses,
                ent_coefs,
                lrs,
                chunk_len,
            )
            last_update = chunk_start + chunk_len - 1
            chunk_timesteps = chunk_len * timesteps_per_update_all

            do_log = (not memory_probe) and (
                last_update % log_every == 0 or last_update == num_updates - 1
            )
            do_perf = verbose_logging and (
                last_update % perf_log_every == 0 or do_log
            )
            perf_extra = None
            if do_perf or do_log:
                jax.block_until_ready(mean_loss)
            if do_perf:
                dt = time.perf_counter() - t_u0
                sps = chunk_timesteps / dt
                print(
                    f"[perf] chunk_end_update={last_update} "
                    f"chunk_len={chunk_len} "
                    f"steady_state_step_time={dt:.3f}s "
                    f"steps_per_second={sps:.1f}"
                )
                perf_extra = {
                    "steady_state_step_time": float(dt),
                    "steps_per_second": float(sps),
                }

            if do_log:
                _log(
                    last_update,
                    mean_loss,
                    loss_vec,
                    mean_vloss,
                    mean_aloss,
                    mean_ent,
                    step_metrics,
                    ep_life,
                    perf_extra=perf_extra,
                )

            for u in range(chunk_start, chunk_start + chunk_len):
                if not memory_probe:
                    _maybe_checkpoint(u, params, opt_state, steps, ep_life)

            gif_u = gif_update_for_chunk(chunk_start, last_update, config)
            if not memory_probe and gif_renderer is not None and gif_u is not None:
                gif_renderer.render_ippo(
                    params, gif_u, network.apply,
                    seed_i=0, label="ippo",
                )

            steady_updates += chunk_len
            chunk_start += chunk_len

        if memory_probe:
            jax.block_until_ready(mean_loss)
            steady_stats = snapshot_device_memory()
            config.setdefault("_MEMORY_PROBE_STATS", {})["steady"] = steady_stats
            _print_memory_snapshot("STEADY", steady_stats)
        elif steady_updates > 0:
            jax.block_until_ready(mean_loss)
            t_steady = time.perf_counter() - t_steady_start
            steady_steps = steady_updates * timesteps_per_update_all
            if verbose_logging:
                print(
                    f"[profile] steady-state ({steady_updates} updates): "
                    f"{t_steady:.3f}s, {steady_steps / t_steady:.1f} env-steps/s "
                    f"({steady_updates / t_steady:.2f} updates/s)"
                )
            if debug_perf:
                print(
                    f"[perf] final JIT trace count={_trace_count[0]} "
                    f"(1 = no retrace; >1 = shape/static churn)"
                )
        elif verbose_logging:
            print(
                "[profile] only 1 update total — steady-state rate N/A "
                "(increase TOTAL_TIMESTEPS for a meaningful estimate)"
            )

        if num_seeds == 1:
            train_states = _states_dict(params, opt_state, steps)
        else:
            train_states = {
                f"seed_{s}": _states_dict(params, opt_state, steps, seed_i=s)
                for s in range(num_seeds)
            }
        if (not memory_probe) and ckpt_dir is not None and checkpoint_every > 0:
            _maybe_checkpoint(num_updates - 1, params, opt_state, steps, ep_life)

        return train_states, mean_loss

    return train


@hydra.main(
    version_base=None,
    config_path="../../configs/ippo",
    config_name="ippo_tiny",
)
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True)
    from harvest_socialjax.envs.env_scale_presets import (
        apply_env_scale,
        print_run_env_banner,
    )
    apply_env_scale(config)
    if bool(config.get("VERBOSE_LOGGING", False)):
        print_jax_device_sanity()
    register_harvest_env()
    assert_reputation_modes_exclusive(config)
    apply_police_agent_settings(config)
    enable_reputation = spatial_reputation_enabled(config)
    enable_reputation_v2 = reputation_v2_enabled(config)
    rep_cfg = reputation_config_from_dict(config) if enable_reputation else None
    rep_v2_cfg = (
        reputation_v2_config_from_dict(config) if enable_reputation_v2 else None
    )
    if enable_reputation:
        inject_reputation_enforcement_env_kwargs(config, rep_cfg)
    # Bandit anneal schedule uses W&B episode_count units via NUM_ENVS multiplier.
    env_kwargs = dict(config.get("ENV_KWARGS") or {})
    if int(env_kwargs.get("bandit_schedule_env_multiplier", 1) or 1) <= 1:
        env_kwargs["bandit_schedule_env_multiplier"] = max(
            1, int(config.get("NUM_ENVS", 1))
        )
    # Peer zap freeze filter (Good immune) lives on Harvest_open.
    if enable_reputation_v2 and rep_v2_cfg is not None:
        env_kwargs["peer_focus_on_bad"] = bool(rep_v2_cfg.peer_focus_on_bad)
        env_kwargs["mask_reputation_obs"] = bool(rep_v2_cfg.mask_reputation_obs)
    config["ENV_KWARGS"] = env_kwargs
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if enable_reputation:
        env = ReputationWrapper(env, rep_cfg)
        print(
            f"[ippo] ENABLE_REPUTATION=true (spatial ch15/ch16 only) "
            f"r_star={rep_cfg.r_star} ema_beta={rep_cfg.ema_beta} "
            f"decay={rep_cfg.decay} alpha_h={rep_cfg.alpha_h} "
            f"alpha_z={rep_cfg.alpha_z} "
            f"burst_decay={rep_cfg.burst_decay} alpha_burst={rep_cfg.alpha_burst} "
            f"reset_rep_on_tag_respawn={rep_cfg.reset_rep_on_tag_respawn} "
            f"enforcement={rep_cfg.enable_enforcement} "
            f"strictness={rep_cfg.enforcement_strictness}"
        )
    elif enable_reputation_v2:
        env = ReputationV2Wrapper(env, rep_v2_cfg)
        print(
            f"[ippo] ENABLE_REPUTATION_V2=true "
            f"({'graded Good(+1,+2,…)/Bad(-1)' if rep_v2_cfg.graded_reputation else 'binary Good(+1)/Bad(-1)'} "
            f"ch15/ch16) "
            f"window_len={rep_v2_cfg.window_len} "
            f"max_harvests_in_window={rep_v2_cfg.max_harvests_in_window} "
            f"timeout_duration={rep_v2_cfg.timeout_duration} "
            f"last_apple_timeout={rep_v2_cfg.last_apple_timeout} "
            f"immediate_timeout={rep_v2_cfg.immediate_timeout} "
            f"timeout_delay={rep_v2_cfg.timeout_delay} "
            f"enforcement_enabled={rep_v2_cfg.enforcement_enabled} "
            f"enforcement_delay={rep_v2_cfg.enforcement_delay} "
            f"enforcement_period={rep_v2_cfg.enforcement_period} "
            f"delay_ramp={rep_v2_cfg.delay_ramp_enabled} "
            f"(every={rep_v2_cfg.delay_ramp_every_episodes} "
            f"max_delay={rep_v2_cfg.delay_ramp_max_delay} "
            f"start_immediate={rep_v2_cfg.delay_ramp_start_immediate} "
            f"hold={rep_v2_cfg.delay_hold_episodes}@{rep_v2_cfg.delay_hold_start_delay} "
            f"end_at={rep_v2_cfg.delay_ramp_end_episodes} "
            f"end_immediate={rep_v2_cfg.delay_curriculum_end_immediate} "
            f"end_disable_enf={rep_v2_cfg.delay_curriculum_end_disable_enforcement}) "
            f"stop_after_eps={rep_v2_cfg.enforcement_stop_after_episodes} "
            f"schedule_mult={rep_v2_cfg.schedule_env_multiplier} "
            f"graded={rep_v2_cfg.graded_reputation} "
            f"graded_max={rep_v2_cfg.graded_max_reputation} "
            f"peer_punish={rep_v2_cfg.peer_punish_reward_enabled} "
            f"(bad={rep_v2_cfg.peer_punish_bad_reward} "
            f"good_pen={rep_v2_cfg.peer_punish_good_penalty}) "
            f"only_good_can_zap={rep_v2_cfg.only_good_can_zap} "
            f"peer_focus_on_bad={rep_v2_cfg.peer_focus_on_bad} "
            f"mask_reputation_obs={rep_v2_cfg.mask_reputation_obs} "
            f"zap_affects_label={rep_v2_cfg.zap_affects_label} "
            f"zap_to_bad_only_pure_good={rep_v2_cfg.zap_to_bad_only_pure_good} "
            f"enforce_mode={rep_v2_cfg.enforce_mode}"
        )
    else:
        print("[ippo] ENABLE_REPUTATION=false ENABLE_REPUTATION_V2=false (baseline)")
    env = LogWrapper(env, replace_info=False)
    police_mask = resolve_police_mask_from_env(env, env.num_agents)
    config["POLICE_MASK"] = onp.asarray(jax.device_get(police_mask), dtype=onp.bool_)
    if config.get("N_HARVESTERS") is None:
        config["N_HARVESTERS"] = int(env.num_agents) - int(
            onp.sum(config["POLICE_MASK"])
        )
    _police_ids = police_agent_indices(police_mask)
    if bool(config.get("ENABLE_POLICE_AGENT", False)) or _police_ids:
        print(
            f"[ippo] ENABLE_POLICE_AGENT="
            f"{bool(config.get('ENABLE_POLICE_AGENT', False))} "
            f"police agents={_police_ids} "
            f"N_harvesters={config['N_HARVESTERS']} num_agents={env.num_agents}"
        )
        if enable_reputation_v2 and rep_v2_cfg is not None and bool(
            rep_v2_cfg.mask_reputation_obs
        ):
            print(
                "[ippo] mask_reputation_obs=true → police CNN also gets "
                "ch15/ch16=0 (labels remain for police zap reward / focus only)",
                flush=True,
            )
    print_run_env_banner(config, env)

    # One-shot obs shape check (CNN vs MLP path).
    _obs_space = env.observation_space()
    _obs_box = _obs_space[0] if isinstance(_obs_space, tuple) else _obs_space
    print(
        f"[ippo] observation_space shape={_obs_box.shape} "
        f"cnn={config['ENV_KWARGS'].get('cnn')} "
        f"num_agents={env.num_agents} action_dim={env.action_space().n}"
    )

    agent_hparams = resolve_agent_hparams(config, env.agents)
    print("[ippo] Independent per-agent actor-critics (no parameter sharing).")
    _opt_shared = bool(config.get("OPTIMIZE_SHARED_REWARDS", False))
    _opt_baseline = bool(config.get("OPTIMIZE_BASELINE", False))
    _eq_coef = float(config.get("EQUALITY_COEF", 1.0))
    _penalty_cap = float(config.get("PENALTY_CAP", 1.0))
    _eq_warmup_frac = float(config.get("EQUALITY_COEF_WARMUP_FRAC", 0.0))
    _eq_delay_frac = float(config.get("EQUALITY_COEF_DELAY_FRAC", 0.0))
    _env_shared = bool(getattr(env, "shared_rewards", False))
    print(
        f"[ippo] reward: OPTIMIZE_SHARED_REWARDS={_opt_shared} "
        f"OPTIMIZE_BASELINE={_opt_baseline} EQUALITY_COEF={_eq_coef} "
        f"PENALTY_CAP={_penalty_cap} "
        f"EQUALITY_COEF_WARMUP_FRAC={_eq_warmup_frac} "
        f"EQUALITY_COEF_DELAY_FRAC={_eq_delay_frac} "
        f"env.shared_rewards={_env_shared} "
        "(logged mean_reward/R = individual apple env_rew only — excludes "
        "peer-punish and police zap rewards; those affect PPO only; "
        "PPO uses team-sum when OPTIMIZE_SHARED_REWARDS, "
        "bounded team-sum equality penalty when OPTIMIZE_BASELINE)"
    )
    _ent_msg = f"ENT_COEF={config['ENT_COEF']}"
    if bool(config.get("ANNEAL_ENT", False)):
        _ent_end = config.get("ENT_COEF_END", config["ENT_COEF"])
        _ent_hz = config.get("ANNEAL_ENT_TOTAL_TIMESTEPS")
        if _ent_hz is None or _ent_hz == "":
            _ent_frac = config.get("ANNEAL_ENT_FRAC", 0.25)
            _ent_hz = (
                f"{float(_ent_frac):.2f}*TOTAL={int(float(_ent_frac) * int(config['TOTAL_TIMESTEPS']))}"
            )
        _ent_msg = (
            f"ENT_COEF={config['ENT_COEF']}→{_ent_end} "
            f"(ANNEAL_ENT over {_ent_hz} steps)"
        )
    print(
        f"[ippo] traced hparams GAMMA={config['GAMMA']} {_ent_msg} "
        f"CLIP_EPS={config['CLIP_EPS']} LR={config['LR']} "
        f"NUM_ENVS={config['NUM_ENVS']}"
    )
    print(f"[ippo] Resolved agent hparams: {agent_hparams}")
    print(
        "[ippo] W&B x-axis = episode_count "
        "(lifetime inner-episode completions across NUM_ENVS, not PPO updates)"
    )
    if config.get("AGENT_OVERRIDES"):
        print(
            "[ippo] AGENT_OVERRIDES present - scaffolding only; "
            "env/obs/action heterogeneity not implemented yet."
        )
        # Early pairwise check (optimizer readback happens inside make_train).
        warn_if_overrides_collapse(agent_hparams)
    else:
        print("[ippo] No AGENT_OVERRIDES - fully homogeneous baseline.")

    ckpt_rel = config.get("CHECKPOINT_DIR", "checkpoints")
    ckpt_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / (
        ckpt_rel or "checkpoints"
    )

    resume_ckpt_path = config.get("RESUME_CHECKPOINT") or os.environ.get(
        "RESUME_CHECKPOINT"
    )
    if resume_ckpt_path:
        resume_payload = load_checkpoint(resume_ckpt_path)
        ep_data = resume_payload.get("episode_life")
        if ep_data and "count" in ep_data:
            config["WANDB_RESUME_EPISODE_COUNT"] = int(
                _checkpoint_scalar(ep_data["count"], what="episode_life.count")
            )
        elif os.environ.get("WANDB_RESUME_EPISODE_COUNT"):
            config["WANDB_RESUME_EPISODE_COUNT"] = int(
                os.environ["WANDB_RESUME_EPISODE_COUNT"]
            )
        resume_wandb = os.environ.get("WANDB_RESUME_ID") or config.get(
            "WANDB_RESUME_ID"
        )
        if resume_wandb:
            print(
                f"[resume] W&B continue id={resume_wandb} "
                f"episode_count={config.get('WANDB_RESUME_EPISODE_COUNT', '?')}"
            )

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_name = str(
        getattr(getattr(hydra_cfg, "job", None), "config_name", None)
        or "ippo_run"
    )
    init_wandb_logger(
        config,
        cfg_name=cfg_name,
    )

    warnings.filterwarnings(
        "error",
        message=r".*scatter inputs have incompatible types.*",
        category=FutureWarning,
    )

    gif_renderer = None
    if config.get("RENDER_GIF", False):
        raw_env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        if enable_reputation:
            raw_env = ReputationWrapper(
                raw_env,
                reputation_config_from_dict(config),
            )
        elif enable_reputation_v2:
            raw_env = ReputationV2Wrapper(
                raw_env,
                reputation_v2_config_from_dict(config),
            )
        gif_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "gifs"
        gif_renderer = GifRenderer(config, raw_env, gif_dir, cfg_name=cfg_name)

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn = make_train(config, env, agent_hparams, ckpt_dir=ckpt_dir, gif_renderer=gif_renderer)
    train_states, final_loss = train_fn(rng)
    n_seeds = int(config.get("NUM_SEEDS", 1))
    if n_seeds == 1:
        agent_list = list(train_states.keys())
    else:
        agent_list = list(train_states["seed_0"].keys())
    loss_arr = onp.asarray(jax.device_get(final_loss)).reshape(-1)
    print(
        f"Training finished. agents={agent_list} "
        f"NUM_SEEDS={n_seeds} "
        f"final_mean_loss={float(onp.nanmean(loss_arr)):.4f}"
    )
    print(f"Checkpoints directory: {ckpt_dir}")
    finish_wandb_logger()


if __name__ == "__main__":
    main()
