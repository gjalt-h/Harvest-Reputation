"""W&B-only training logger for IPPO (no local dashboard)."""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any

_active_config: dict[str, Any] | None = None
_wandb_active = False
_episode_axis_defined = False
_last_wandb_episode_step: int | None = None

_SKIP_LOG_KEYS = frozenset({"_step", "_runtime", "_timestamp"})

_PARAM_TOP_LEVEL = (
    "SEED",
    "NUM_SEEDS",
    "LR",
    "ENT_COEF",
    "ANNEAL_ENT",
    "ENT_COEF_END",
    "ANNEAL_ENT_FRAC",
    "CLIP_EPS",
    "GAMMA",
    "GAE_LAMBDA",
    "VF_COEF",
    "MAX_GRAD_NORM",
    "NUM_ENVS",
    "NUM_STEPS",
    "TOTAL_TIMESTEPS",
    "LOG_EVERY",
    "ENV_NAME",
    "ENV_SCALE",
    "ENV_NUM_AGENTS",
    "PROJECT",
    "UPDATE_EPOCHS",
    "NUM_MINIBATCHES",
)

_ENV_KWARG_KEYS = (
    "num_inner_steps",
    "shared_rewards",
    "cnn",
)


def _scalar(value: Any) -> float | int | bool | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value != value:
            return None
        return value
    try:
        out = float(value)
        if out != out:
            return None
        return out
    except (TypeError, ValueError):
        return None


def sanitize_wandb_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop non-finite floats so NaNs cannot overwrite prior W&B history.

    With ``define_metric(*, step_metric=episode_count)``, logging
    ``mean_reward=NaN`` at the same ``episode_count`` as a previous finite
    point replaces the good value. Interval means are legitimately NaN when
    no inner episode finished since the last log (e.g. jit1 + NUM_STEPS < T);
    those keys must be omitted, not written as NaN.
    """
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _SKIP_LOG_KEYS:
            continue
        if isinstance(value, bool) or value is None:
            out[key] = value
            continue
        if hasattr(value, "item") and not isinstance(value, (bytes, str)):
            try:
                value = value.item()
            except (ValueError, TypeError):
                continue
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            out[key] = value
            continue
        out[key] = value
    return out


def env_steps_from_update(config: dict[str, Any] | None, update: int) -> int:
    if config is None:
        return int(update) + 1
    num_envs = int(config.get("NUM_ENVS", 1) or 1)
    num_steps = int(config.get("NUM_STEPS", 1) or 1)
    return (int(update) + 1) * num_envs * num_steps


def _fmt_group_float(value: Any, *, digits: int = 6) -> str:
    """Stable float token for W&B group ids (``1.36111`` → ``1p36111``)."""
    text = f"{float(value):.{digits}g}"
    return text.replace("-", "m").replace(".", "p")


def _fmt_timesteps(ts: int) -> str:
    if ts >= 1_000_000 and ts % 1_000_000 == 0:
        return f"{ts // 1_000_000}m"
    if ts >= 1_000 and ts % 1_000 == 0:
        return f"{ts // 1_000}k"
    return str(ts)


def resolve_wandb_group(config: dict[str, Any]) -> str:
    """Seed-agnostic W&B ``group`` so replicate seeds share one family for mean±std.

    Identity includes reward mode, N, apple density/layout/regrowth, horizon, and
    other training knobs that define the experimental cell — never ``SEED``.

    Default mode is ``seed_family`` (this function). Set ``WANDB_GROUP_MODE=manual``
    to keep a legacy sweep-wide ``WANDB_RUN_GROUP`` as the W&B group instead.
    """
    mode = str(
        os.environ.get("WANDB_GROUP_MODE")
        or config.get("WANDB_GROUP_MODE")
        or "seed_family"
    ).lower()
    manual = (
        os.environ.get("WANDB_RUN_GROUP")
        or config.get("WANDB_RUN_GROUP")
        or config.get("WANDB_GROUP")
    )
    if mode in ("manual", "legacy") and manual:
        return str(manual)

    env = dict(config.get("ENV_KWARGS") or {})
    n = int(config.get("ENV_NUM_AGENTS") or env.get("num_agents") or 0)
    shared = bool(config.get("OPTIMIZE_SHARED_REWARDS", False)) or bool(
        env.get("shared_rewards", False)
    )
    reward_mode = "shared" if shared else "indep"
    layout = str(env.get("apple_layout_mode") or "auto")
    dens = env.get("apple_density")
    dens_s = _fmt_group_float(dens) if dens is not None else "na"
    rg = env.get("regrowth_scale", 1.0)
    rg_s = _fmt_group_float(rg, digits=4)
    ts = int(float(config.get("TOTAL_TIMESTEPS") or 0))
    parts = [
        "ippo",
        reward_mode,
        f"n{n}",
        f"d{dens_s}",
        layout,
        f"rg{rg_s}",
        f"t{_fmt_timesteps(ts)}",
    ]
    gamma = config.get("GAMMA")
    if gamma is not None:
        parts.append(f"g{_fmt_group_float(gamma, digits=4)}")
    ent = config.get("ENT_COEF")
    if ent is not None:
        parts.append(f"ent{_fmt_group_float(ent, digits=4)}")
    eq = float(config.get("EQUALITY_COEF") or 0.0)
    if eq:
        parts.append(f"eq{_fmt_group_float(eq, digits=4)}")
    for key, label in (
        ("EQUALITY_COEF_DELAY_FRAC", "eqd"),
        ("EQUALITY_COEF_WARMUP_FRAC", "eqw"),
    ):
        raw = config.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val:
            parts.append(f"{label}{_fmt_group_float(val, digits=4)}")
    if bool(config.get("ENABLE_REPUTATION", False)):
        parts.append("rep")
    return "_".join(parts)


def resolve_run_name(config: dict[str, Any], cfg_name: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = config.get("RUN_NAME_SUFFIX")
    base = f"{cfg_name}_{timestamp}"
    if suffix:
        base = f"{cfg_name}_{suffix}_{timestamp}"
    name = os.environ.get("RUN_NAME") or os.environ.get("WANDB_NAME", base)
    seed = config.get("SEED")
    if seed is None:
        return name
    seed_token = f"seed{int(seed)}"
    # Avoid doubling when sbatch already baked seed into WANDB_NAME.
    compact = name.lower().replace("-", "_")
    if seed_token in compact or f"_{seed_token}_" in f"_{compact}_":
        return name
    return f"{name}_{seed_token}"


def _import_wandb():
    blocked = {Path.cwd().resolve()}
    here = Path(__file__).resolve()
    for parent in here.parents[:5]:
        blocked.add(parent)
    cleaned = []
    for entry in sys.path:
        if entry in ("", "."):
            continue
        try:
            path = Path(entry).resolve()
        except OSError:
            cleaned.append(entry)
            continue
        if path in blocked and (path / "wandb").is_dir() and not (
            path / "wandb" / "__init__.py"
        ).exists():
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned
    import wandb

    return wandb


def _define_episode_count_axis() -> None:
    global _episode_axis_defined
    if _episode_axis_defined:
        return
    wandb = _import_wandb()
    wandb.define_metric("episode_count")
    wandb.define_metric("*", step_metric="episode_count")
    _episode_axis_defined = True
    print("[wandb] x-axis: episode_count (define_metric * step_metric=episode_count)")


def init_wandb_logger(
    config: dict[str, Any],
    *,
    cfg_name: str,
) -> bool:
    """Initialize W&B when ``USE_WANDB`` is true."""
    global _active_config, _wandb_active, _episode_axis_defined, _last_wandb_episode_step

    _active_config = config
    _wandb_active = False
    _episode_axis_defined = False
    _last_wandb_episode_step = None

    if not bool(config.get("USE_WANDB", False)):
        print("[wandb] USE_WANDB=false — skipping W&B init")
        return False

    try:
        wandb = _import_wandb()
    except ImportError as exc:
        print(f"[wandb] package not available ({exc!r}); continuing without WandB")
        return False

    run_name = resolve_run_name(config, cfg_name)
    tags = [
        t.strip()
        for t in os.environ.get("WANDB_TAGS", "").split(",")
        if t.strip()
    ]
    seed = config.get("SEED")
    if seed is not None:
        seed_tag = f"seed{int(seed)}"
        if seed_tag not in tags:
            tags.append(seed_tag)
    notes = os.environ.get("WANDB_NOTES")
    wandb_mode = config.get("WANDB_MODE", os.environ.get("WANDB_MODE", "online"))
    # Seed-agnostic group for mean±std; optional WANDB_RUN_GROUP kept as job_type /
    # tag so old sweep-family labels remain filterable without mixing configs.
    wandb_group = resolve_wandb_group(config)
    sweep_family = (
        os.environ.get("WANDB_RUN_GROUP")
        or config.get("WANDB_RUN_GROUP")
        or config.get("WANDB_GROUP")
    )
    if sweep_family and str(sweep_family) != str(wandb_group):
        family_tag = f"sweep_family:{sweep_family}"
        if family_tag not in tags:
            tags.append(family_tag)
    resume_id = os.environ.get("WANDB_RESUME_ID") or config.get("WANDB_RESUME_ID")
    try:
        from harvest_socialjax.wandb_projects import (
            IPPO_REPUTATION_WANDB_PROJECT,
            IPPO_WANDB_PROJECT,
        )

        if "ENABLE_SPATIAL_REPUTATION" in config:
            enable_reputation = bool(config["ENABLE_SPATIAL_REPUTATION"])
        else:
            # Classic scalar reputation OR v2 burst-window enforcement TRIGGER.
            enable_reputation = bool(config.get("ENABLE_REPUTATION", False)) or bool(
                config.get("ENABLE_REPUTATION_V2", False)
            )
        explicit = config.get("PROJECT")
        if enable_reputation:
            # Reputation runs go to harvest_ippo_rep. An explicit PROJECT wins
            # unless it is still the baseline default (legacy configs).
            if explicit and explicit not in (IPPO_WANDB_PROJECT, "harvest_ippo"):
                project = explicit
            else:
                project = IPPO_REPUTATION_WANDB_PROJECT
        else:
            project = explicit or IPPO_WANDB_PROJECT
    except ImportError:
        if bool(config.get("ENABLE_REPUTATION", False)) or bool(
            config.get("ENABLE_REPUTATION_V2", False)
        ):
            project = config.get("PROJECT") or "harvest_ippo_rep"
        else:
            project = config.get("PROJECT") or "harvest_ippo"

    entity = os.environ.get("WANDB_ENTITY") or None
    # W&B projects cap unique JobType values at 64. This project already hit
    # that ceiling from old long sweep-family job_types, so default is to omit
    # job_type entirely (group + tags carry the experiment identity).
    # Set WANDB_JOB_TYPE / config WANDB_JOB_TYPE to force a short type (e.g. "ippo").
    job_type_raw = os.environ.get("WANDB_JOB_TYPE")
    if job_type_raw is None or str(job_type_raw).strip() == "":
        job_type_raw = config.get("WANDB_JOB_TYPE")
    job_type = (
        str(job_type_raw).strip()
        if job_type_raw is not None and str(job_type_raw).strip() != ""
        else None
    )
    init_kwargs = {
        "project": project,
        "entity": entity,
        "config": config,
        "mode": wandb_mode,
        "name": run_name,
        "group": wandb_group,
        "dir": os.environ.get("WANDB_DIR"),
        "notes": notes,
        "tags": tags or None,
    }
    if job_type is not None:
        init_kwargs["job_type"] = job_type
    if resume_id:
        init_kwargs["id"] = resume_id
        init_kwargs["resume"] = "must"
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    resume_ep = config.get("WANDB_RESUME_EPISODE_COUNT") or os.environ.get(
        "WANDB_RESUME_EPISODE_COUNT"
    )
    if resume_ep is not None:
        _last_wandb_episode_step = int(resume_ep)

    try:
        wandb.init(**init_kwargs)
        _wandb_active = wandb.run is not None
        if _wandb_active:
            print(
                f"[wandb] init mode={wandb_mode} project={wandb.run.project} "
                f"id={wandb.run.id} name={wandb.run.name} "
                f"group={wandb.run.group} seed={config.get('SEED', '?')}"
            )
            _define_episode_count_axis()
        return _wandb_active
    except Exception as exc:
        msg = f"{exc!s} {exc!r}".lower()
        # Retry once without job_type if the project hit W&B's 64 JobType cap.
        if "job_type" in init_kwargs and (
            "jobtype" in msg or "job_type" in msg or "64 limit" in msg
        ):
            print(
                f"[wandb] WARNING: init failed ({exc!r}); retrying without job_type"
            )
            retry = dict(init_kwargs)
            retry.pop("job_type", None)
            try:
                wandb.init(**retry)
                _wandb_active = wandb.run is not None
                if _wandb_active:
                    print(
                        f"[wandb] init mode={wandb_mode} project={wandb.run.project} "
                        f"id={wandb.run.id} name={wandb.run.name} "
                        f"group={wandb.run.group} seed={config.get('SEED', '?')} "
                        "(job_type omitted)"
                    )
                    _define_episode_count_axis()
                return _wandb_active
            except Exception as exc2:
                print(f"[wandb] ERROR: wandb.init() retry failed: {exc2!r}")
                _wandb_active = False
                return False
        print(f"[wandb] ERROR: wandb.init() failed: {exc!r}")
        _wandb_active = False
        return False


def log_training_metrics(payload: dict[str, Any]) -> None:
    """Log one training interval to W&B (x-axis ``episode_count`` when present)."""
    global _last_wandb_episode_step

    if not _wandb_active:
        return

    update = payload.get("update", payload.get("step", payload.get("_step", 0)))
    env_steps = payload.get("env_steps")
    if env_steps is None:
        env_steps = env_steps_from_update(_active_config, int(update))
    enriched = sanitize_wandb_payload(dict(payload))
    enriched["env_steps"] = int(env_steps)

    episode_count = payload.get("episode_count")
    if episode_count is not None:
        episode_count = int(episode_count)
        enriched["episode_count"] = episode_count

    try:
        wandb = _import_wandb()
        if wandb.run is None:
            return
        if episode_count is not None:
            if not _episode_axis_defined:
                _define_episode_count_axis()
            log_kwargs: dict[str, Any] = {}
            if (
                _last_wandb_episode_step is None
                or episode_count > _last_wandb_episode_step
            ):
                log_kwargs["step"] = episode_count
                _last_wandb_episode_step = episode_count
            wandb.log(dict(enriched), **log_kwargs)
        else:
            wandb.log(dict(enriched), step=int(env_steps))
    except Exception as exc:
        print(f"[wandb] log failed: {exc!r}")


def finish_wandb_logger(*, status: str = "finished") -> None:
    global _active_config, _wandb_active, _episode_axis_defined, _last_wandb_episode_step
    del status
    if _wandb_active:
        try:
            wandb = _import_wandb()
            if wandb.run is not None and _active_config is not None:
                if bool(_active_config.get("WANDB_MEAN_STD_PLOTS", True)):
                    try:
                        from harvest_socialjax.legacy_wandb.wandb_mean_std_plots import (
                            finalize_training_mean_std_plots,
                        )

                        finalize_training_mean_std_plots(
                            wandb.run,
                            window=int(_active_config.get("WANDB_MEAN_STD_WINDOW", 10)),
                        )
                    except Exception as exc:
                        print(f"[wandb] mean±std plot finalize failed: {exc!r}")
            if wandb.run is not None:
                wandb.finish()
                print("[wandb] finished")
        except Exception as exc:
            print(f"[wandb] finish failed: {exc!r}")
    _wandb_active = False
    _active_config = None
    _episode_axis_defined = False
    _last_wandb_episode_step = None
