"""Hydra / trainer hooks for optional police agent runs."""
from __future__ import annotations

import jax.numpy as jnp

from harvest_socialjax.envs.police.mechanics import (
    normalize_police_agent_ids,
    police_agent_indices,
)


def n_harvesters_from_config(config: dict) -> int:
    if config.get("N_HARVESTERS") is not None and config.get("N_HARVESTERS") != "":
        return int(config["N_HARVESTERS"])
    if config.get("ENV_NUM_AGENTS") is not None and config.get("ENV_NUM_AGENTS") != "":
        return int(config["ENV_NUM_AGENTS"])
    env_kwargs = dict(config.get("ENV_KWARGS") or {})
    return int(env_kwargs.get("num_agents", 0))


def _resolve_police_ids(config: dict, *, n_harvesters: int) -> tuple[int, ...]:
    """Police indices appended after harvesters.

    Precedence:
    1. ``POLICE_AGENT_IDS`` — explicit list/tuple
    2. ``POLICE_AGENT_ID`` — single id (legacy)
    3. ``NUM_POLICE_AGENTS`` — append that many at ``N, N+1, …`` (default 1)
    """
    ids_cfg = config.get("POLICE_AGENT_IDS")
    if ids_cfg is not None and ids_cfg != "":
        if isinstance(ids_cfg, (list, tuple)):
            return tuple(int(i) for i in ids_cfg)
        # Hydra may pass a comma-string from CLI.
        return tuple(int(x.strip()) for x in str(ids_cfg).split(",") if x.strip())

    police_id = config.get("POLICE_AGENT_ID")
    if police_id is not None and police_id != "":
        return (int(police_id),)

    n_police = int(config.get("NUM_POLICE_AGENTS", 1) or 1)
    if n_police < 1:
        raise ValueError(f"NUM_POLICE_AGENTS must be >= 1, got {n_police}")
    return tuple(range(n_harvesters, n_harvesters + n_police))


def apply_police_agent_settings(config: dict) -> None:
    """Inject police env kwargs when ``ENABLE_POLICE_AGENT=true``.

    ``ENV_NUM_AGENTS`` / scale presets count **harvesters only**. Police are
    appended (default ids ``N_harvesters .. N+NUM_POLICE-1``). Requires
    reputation v2 so labels exist for the +1 Bad / −1 Good police reward.
    """
    n_harvesters = n_harvesters_from_config(config)
    if n_harvesters <= 0:
        raise ValueError(
            "ENV_NUM_AGENTS (harvesting agents) must be set before police setup"
        )
    config["N_HARVESTERS"] = n_harvesters

    if not bool(config.get("ENABLE_POLICE_AGENT", False)):
        return
    if not bool(config.get("ENABLE_REPUTATION_V2", False)):
        raise ValueError(
            "ENABLE_POLICE_AGENT requires ENABLE_REPUTATION_V2=true "
            "(binary Good/Bad labels for police reward)"
        )

    env_kwargs = dict(config.get("ENV_KWARGS") or {})
    police_ids = _resolve_police_ids(config, n_harvesters=n_harvesters)

    n_total = n_harvesters + len(police_ids)
    for i in police_ids:
        if i < n_harvesters:
            raise ValueError(
                f"police id={i} must be >= N harvesters ({n_harvesters}); "
                "police is added on top of harvesting agents, not replacing one"
            )
        if i >= n_total:
            raise ValueError(
                f"police id={i} out of range for num_agents={n_total} "
                f"(N_harvesters={n_harvesters}, n_police={len(police_ids)})"
            )

    normalize_police_agent_ids(police_ids, num_agents=n_total)
    config["NUM_POLICE_AGENTS"] = len(police_ids)
    # Keep orchard sized for harvesters (density was set with ENV_NUM_AGENTS).
    density = float(env_kwargs.get("apple_density", 1.0))
    env_kwargs["apple_density"] = density * (float(n_harvesters) / float(n_total))
    env_kwargs["num_agents"] = n_total
    env_kwargs["police_agent_ids"] = list(police_ids)

    police_block = dict(config.get("POLICE") or {})
    env_kwargs["police_focus_on_bad"] = bool(police_block.get("focus_on_bad", False))

    beam_length = int(env_kwargs.get("beam_length", 20))
    beam_width = int(env_kwargs.get("beam_width", 5))
    tag_timeout = int(env_kwargs.get("tag_timeout", 25))

    if "police_beam_length" in police_block and police_block["police_beam_length"] is not None:
        env_kwargs["police_beam_length"] = int(police_block["police_beam_length"])
    else:
        env_kwargs["police_beam_length"] = beam_length

    if "police_beam_width" in police_block and police_block["police_beam_width"] is not None:
        env_kwargs["police_beam_width"] = int(police_block["police_beam_width"])
    else:
        env_kwargs["police_beam_width"] = beam_width

    if (
        "police_timeout_duration" in police_block
        and police_block["police_timeout_duration"] is not None
    ):
        env_kwargs["police_timeout_duration"] = int(
            police_block["police_timeout_duration"]
        )
    else:
        env_kwargs["police_timeout_duration"] = tag_timeout

    config["ENV_KWARGS"] = env_kwargs
    print(
        f"[police] ENABLE_POLICE_AGENT=true "
        f"N_harvesters={n_harvesters} (agents 0..{n_harvesters - 1}) + "
        f"{len(police_ids)} police {list(police_ids)} -> num_agents={n_total} "
        f"focus_on_bad={env_kwargs['police_focus_on_bad']} "
        f"police_beam={env_kwargs['police_beam_length']}x"
        f"{env_kwargs['police_beam_width']} "
        f"police_timeout={env_kwargs['police_timeout_duration']} "
        f"(harvester_beam={beam_length}x{beam_width} tag_timeout={tag_timeout})"
    )


def resolve_police_mask_from_env(env, num_agents: int) -> jnp.ndarray:
    cur = env
    for _ in range(16):
        for attr in ("police_mask", "_police_mask"):
            mask = getattr(cur, attr, None)
            if mask is not None:
                mask = jnp.asarray(mask, dtype=jnp.bool_)
                if mask.shape == (num_agents,) and bool(jnp.any(mask)):
                    return mask
        nxt = getattr(cur, "env", None) or getattr(cur, "_env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return jnp.zeros((num_agents,), dtype=jnp.bool_)


__all__ = [
    "apply_police_agent_settings",
    "n_harvesters_from_config",
    "police_agent_indices",
    "resolve_police_mask_from_env",
]
