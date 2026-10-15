"""Harvest Open extensions for SocialJax (IPPO training, reputation, etc.)."""
from __future__ import annotations

from typing import Any

import harvest_socialjax.jax_compat  # noqa: F401
from harvest_socialjax.socialjax_path import ensure_socialjax_on_path

ensure_socialjax_on_path()

from harvest_socialjax.envs.harvest_open import (  # noqa: E402
    Actions,
    Harvest_open,
)


def make_harvest_open(**kwargs: Any) -> Harvest_open:
    """Construct the Harvest Open env."""
    return Harvest_open(**kwargs)


def register_harvest_env() -> None:
    """Patch ``socialjax.make`` so ``harvest_common_open`` uses ``Harvest_open``."""
    import socialjax
    import socialjax.registration as registration

    _original_make = registration.make

    def patched_make(env_id: str, **env_kwargs):
        if env_id == "harvest_common_open":
            return Harvest_open(**env_kwargs)
        return _original_make(env_id, **env_kwargs)

    registration.make = patched_make
    socialjax.make = patched_make
