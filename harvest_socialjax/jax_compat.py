"""JAX API compatibility for environments pinned to JAX < 0.4.25."""
from __future__ import annotations

import types

import jax


def ensure_jax_tree_namespace() -> None:
    """Backfill ``jax.tree.*`` helpers used by newer SocialJax code."""
    if hasattr(jax, "tree"):
        return
    jax.tree = types.SimpleNamespace()
    jax.tree.map = jax.tree_util.tree_map
    jax.tree.leaves = jax.tree_util.tree_leaves
    jax.tree.structure = jax.tree_util.tree_structure
    jax.tree.flatten = jax.tree_util.tree_flatten
    jax.tree.unflatten = jax.tree_util.tree_unflatten


ensure_jax_tree_namespace()
