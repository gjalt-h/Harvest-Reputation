"""Lightweight DEBUG_PERF helpers shared by IPPO / DQN training scripts.

Does not affect training math — host-side timing, shape-signature prints, and
device/backend reporting only.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Hashable

import jax


def print_jax_device_sanity(*, expect_gpu: bool | None = None) -> None:
    """Print devices/backend once; warn if GPU expected but backend is cpu."""
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"[perf] jax.default_backend() = {backend}")
    print(f"[perf] jax.local_device_count() = {jax.local_device_count()}")
    print(f"[perf] jax.devices() = {devices}")
    platforms_env = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if expect_gpu is None:
        expect_gpu = platforms_env == ""
    gpu_present = any(getattr(d, "platform", "") == "gpu" for d in devices) or any(
        "cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devices
    )
    if platforms_env == "cpu":
        print(
            "[perf] NOTE: JAX_PLATFORMS=cpu — training will use CPU "
            "(expected on login nodes; unset JAX_PLATFORMS on GPU/SLURM nodes)."
        )
    elif expect_gpu and backend == "cpu" and not gpu_present:
        print(
            "[perf] WARNING: backend is 'cpu' and no GPU device visible. "
            "If this is a GPU node, check CUDA / JAX install / JAX_PLATFORMS."
        )
    elif expect_gpu and backend == "cpu" and gpu_present:
        print(
            "[perf] WARNING: GPU device(s) exist but default_backend is 'cpu'. "
            "Check JAX_PLATFORMS / device selection."
        )


@contextmanager
def log_compiles_first_update(enabled: bool):
    """Enable JAX compile logging for the first update only."""
    if not enabled:
        yield
        return
    prev = False
    try:
        prev = bool(getattr(jax.config, "jax_log_compiles", False))
    except Exception:
        prev = False
    try:
        jax.config.update("jax_log_compiles", True)
        print("[perf] jax_log_compiles=True for first update only")
        yield
    finally:
        try:
            jax.config.update("jax_log_compiles", prev)
        except Exception:
            pass
        print("[perf] jax_log_compiles restored after first update")


class ShapeSignatureTracker:
    """Print static shape signatures once per unique signature (retrace guard)."""

    def __init__(self):
        self._seen: set[Hashable] = set()

    def note(self, label: str, signature: Hashable) -> None:
        if signature in self._seen:
            return
        self._seen.add(signature)
        print(f"[perf] NEW shape signature ({label}): {signature}")


def make_trace_counter(label: str = "_update_body"):
    """Return (counter_list, on_trace) for detecting JIT retraces.

    Plain Python side effects inside a ``jax.jit``'d function body execute during
    *tracing* (compile / retrace), not on every steady-state XLA execution.
    """
    counter: list[int] = [0]

    def on_trace() -> None:
        counter[0] += 1
        n = counter[0]
        if n == 1:
            print(f"[perf] JIT TRACING {label} (count={n}) — compile/retrace path")
        else:
            print(
                f"[perf] WARNING: JIT RETRACE of {label} (count={n}) — "
                f"shapes/statics likely changed; throughput will drop"
            )

    return counter, on_trace
