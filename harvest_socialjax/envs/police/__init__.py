"""Optional police agent for harvest IPPO + reputation v2.

Enable with ``ENABLE_POLICE_AGENT=true`` and ``ENABLE_REPUTATION_V2=true``.
Police are appended on top of ``ENV_NUM_AGENTS`` harvesters
(``NUM_POLICE_AGENTS``, default 1 → ids ``N, N+1, …``).
"""
from __future__ import annotations

from harvest_socialjax.envs.police.config import (
    PoliceConfig,
    police_config_from_dict,
)
from harvest_socialjax.envs.police.mechanics import (
    apply_police_focus_on_bad_zap_hits,
    harvester_metric_weights,
    mask_police_harvest,
    mask_zap_hits_on_police,
    normalize_police_agent_ids,
    police_agent_indices,
    police_agent_mask,
    police_binary_zap_reward,
    split_beam_victims,
)
from harvest_socialjax.envs.police.train_hooks import (
    apply_police_agent_settings,
    n_harvesters_from_config,
    resolve_police_mask_from_env,
)

__all__ = [
    "PoliceConfig",
    "apply_police_agent_settings",
    "apply_police_focus_on_bad_zap_hits",
    "harvester_metric_weights",
    "mask_police_harvest",
    "mask_zap_hits_on_police",
    "n_harvesters_from_config",
    "normalize_police_agent_ids",
    "police_agent_indices",
    "police_agent_mask",
    "police_binary_zap_reward",
    "police_config_from_dict",
    "resolve_police_mask_from_env",
    "split_beam_victims",
]
