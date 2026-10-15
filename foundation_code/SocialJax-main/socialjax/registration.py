"""Minimal SocialJax env registry (trimmed vendored subset).

Upstream SocialJax registers many substrates. This thesis release only needs
``harvest_common_open``, which ``harvest_socialjax.register_harvest_env()``
patches to the local Harvest Open implementation before training.
"""

REGISTERED_ENVS = [
    "harvest_common_open",
]


def make(env_id: str, **env_kwargs):
    """JAX-style ``env.make(env_name)`` for the trimmed registry."""
    if env_id not in REGISTERED_ENVS:
        raise ValueError(
            f"{env_id} is not in registered SocialJax environments "
            f"(trimmed subset: {REGISTERED_ENVS})"
        )
    if env_id == "harvest_common_open":
        from socialjax.environments.common_harvest import Harvest_open

        return Harvest_open(**env_kwargs)
    raise ValueError(f"Unhandled env_id={env_id!r}")
