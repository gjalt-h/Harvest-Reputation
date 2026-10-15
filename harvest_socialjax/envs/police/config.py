"""Police agent configuration for harvest + reputation v2."""
from __future__ import annotations

from flax import struct


@struct.dataclass
class PoliceConfig:
    """Police beam / timeout overrides and optional focus-on-Bad tagging."""

    # None → use the same beam as harvesters (ENV_KWARGS beam_length/width).
    police_beam_length: int | None = None
    police_beam_width: int | None = None
    # None → use ENV_KWARGS.tag_timeout (default 25).
    police_timeout_duration: int | None = None
    # If True, police zap only freezes Bad (label < 0) targets.
    focus_on_bad: bool = False


def police_config_from_dict(config: dict) -> PoliceConfig:
    """Build ``PoliceConfig`` from Hydra ``POLICE`` block."""
    block = dict(config.get("POLICE") or {})
    pbl = block.get("police_beam_length")
    pbw = block.get("police_beam_width")
    ptd = block.get("police_timeout_duration")
    return PoliceConfig(
        police_beam_length=None if pbl is None else int(pbl),
        police_beam_width=None if pbw is None else int(pbw),
        police_timeout_duration=None if ptd is None else int(ptd),
        focus_on_bad=bool(block.get("focus_on_bad", False)),
    )
