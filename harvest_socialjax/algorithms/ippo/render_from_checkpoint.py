"""Greedy-eval GIF + episode diagnostic from a saved IPPO checkpoint.

Training runs should keep ``RENDER_GIF=false`` and ``CHECKPOINT_EVERY>0``;
call :func:`render_ippo_checkpoint` (or the CLI wrapper) on demand.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import yaml

from harvest_socialjax import register_harvest_env
from harvest_socialjax.algorithms.ippo.train_ippo_harvest import (
    ActorCritic,
    load_checkpoint,
    reputation_config_from_dict,
    reputation_v2_config_from_dict,
    reputation_v2_enabled,
    restore_stacked_train_state,
    spatial_reputation_enabled,
)
from harvest_socialjax.algorithms.render_gif import GifRenderer
from harvest_socialjax.envs.env_scale_presets import apply_env_scale
from harvest_socialjax.socialjax_path import ensure_socialjax_on_path
from harvest_socialjax.wrappers import ReputationV2Wrapper, ReputationWrapper

ensure_socialjax_on_path()
import socialjax  # noqa: E402

_CKPT_UPDATE_RE = re.compile(r"ckpt_update_(\d+)\.pkl$")


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    """Load resolved Hydra config from a training output directory."""
    run_dir = Path(run_dir)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Missing Hydra config at {cfg_path}. "
            "Pass --run-dir to the training hydra.run.dir output."
        )
    with cfg_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise TypeError(f"Expected dict config in {cfg_path}, got {type(config)}")
    apply_env_scale(config)
    return config


def resolve_checkpoint_path(
    checkpoint: str | Path,
    *,
    run_dir: str | Path | None = None,
) -> Path:
    """Resolve ``checkpoint`` to an existing ``.pkl`` file."""
    ckpt = Path(checkpoint)
    if ckpt.is_file():
        return ckpt
    if run_dir is not None:
        run_dir = Path(run_dir)
        candidate = run_dir / checkpoint
        if candidate.is_file():
            return candidate
        candidate = run_dir / "checkpoints" / Path(checkpoint).name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def checkpoint_update_index(ckpt_path: str | Path) -> int:
    """Read PPO update index from ``ckpt_update_XXXXXX.pkl`` filename."""
    match = _CKPT_UPDATE_RE.search(Path(ckpt_path).name)
    if match:
        return int(match.group(1))
    payload = load_checkpoint(ckpt_path)
    return int(payload["update"])


def _apply_render_defaults(
    config: dict[str, Any],
    *,
    diag_only: bool = False,
    overlay_reputation: bool = False,
    gif_frames: int | None = None,
    clean_good_bad: bool = False,
    gif_only: bool = False,
    occupancy_heatmap: bool = False,
    occupancy_mode: str = "auto",
) -> dict[str, Any]:
    """Enable post-hoc visualisation without mutating the training config on disk.

    ``diag_only``
        Skip GIF encode; still run the greedy episode and write episode-diag.
    ``overlay_reputation``
        Wrap eval with ``ReputationWrapper`` even if training had
        ``ENABLE_REPUTATION=false``. Rep updates are recorded for episode-diag
        plots; the policy is always spatial-only (ch15/ch16 when wrapper on).
    ``gif_frames``
        If set (and not ``diag_only``), write one short GIF with exactly this
        many frames (first frames of the episode) and skip the end-tail GIF.
        Full episode still runs for episode-diag / occupancy unless ``gif_only``.
    ``clean_good_bad``
        Grid-only GIF: colour agents Good=green / Bad=red, compact legend,
        no title bar / numeric reputation panel / on-map labels.
        Implies GIF-only (no episode-diag) unless ``gif_only`` is explicitly
        False via caller — pass ``gif_only=False`` only if you also want diag.
    ``gif_only``
        Skip episode-diag PNG/CSV; with ``gif_frames``, stop after enough frames
        unless ``occupancy_heatmap`` (full episode required for the visit map).
    ``occupancy_heatmap``
        Accumulate per-cell visit counts over the full episode and write a
        thesis-style heatmap PNG (+ ``.npz``). Forces a full-length rollout.
    """
    out = dict(config)
    trained_rep = bool(out.get("ENABLE_REPUTATION", False))
    if overlay_reputation:
        out["OVERLAY_REPUTATION"] = True
        out["ENABLE_REPUTATION"] = True
        if not out.get("REPUTATION"):
            out["REPUTATION"] = {"r_star": 0.2}
        else:
            out["REPUTATION"] = dict(out["REPUTATION"])
            out["REPUTATION"].setdefault("r_star", 0.2)
    else:
        out["OVERLAY_REPUTATION"] = False

    enable_rep = bool(out.get("ENABLE_REPUTATION", False)) or bool(
        out.get("ENABLE_REPUTATION_V2", False)
    )
    # Clean Good/Bad previews default to GIF-only (no matplotlib diag).
    # Occupancy heatmap needs the full episode — do not treat as gif-only stop.
    effective_gif_only = (
        bool(gif_only or clean_good_bad)
        and (not diag_only)
        and (not occupancy_heatmap)
    )
    heatmap_only = bool(occupancy_heatmap) and (not clean_good_bad) and (
        diag_only or gif_frames is None
    )
    want_gif = (
        (not diag_only)
        and (not heatmap_only)
        and (gif_frames is None or int(gif_frames) > 0)
    )
    short_gif = (
        want_gif and gif_frames is not None and int(gif_frames) > 0
    )
    n_gif = int(gif_frames) if short_gif else int(out.get("RENDER_GIF_START_STEPS", 250))

    out.update(
        {
            "USE_WANDB": False,
            "RENDER_GIF": want_gif,
            "RENDER_GIF_START_STEPS": n_gif,
            "RENDER_GIF_MAX_FRAMES": n_gif if short_gif else 0,
            "RENDER_GIF_TAIL_STEPS": int(
                out.get(
                    "RENDER_GIF_TAIL_STEPS",
                    out.get("RENDER_GIF_STEPS", 100),
                )
            ),
            # Short preview GIFs: no end-tail. Default training-style render keeps
            # whatever was in the run config unless short_gif forces it off.
            "RENDER_GIF_TAIL": (
                False
                if short_gif or diag_only or heatmap_only
                else bool(out.get("RENDER_GIF_TAIL", False))
            ),
            "RENDER_GIF_CANONICAL_ORIENT": bool(
                out.get("RENDER_GIF_CANONICAL_ORIENT", True)
            ),
            "RENDER_GIF_REP_PANEL": (
                False
                if clean_good_bad
                else bool(out.get("RENDER_GIF_REP_PANEL", enable_rep and want_gif))
            ),
            # Paint i:±rep on the short GIF when overlaying reputation.
            "RENDER_GIF_OVERLAY_REP": (
                False
                if clean_good_bad
                else bool(
                    out.get(
                        "RENDER_GIF_OVERLAY_REP",
                        bool(overlay_reputation and want_gif),
                    )
                )
            ),
            "RENDER_GIF_CLEAN_GOOD_BAD": bool(clean_good_bad),
            "RENDER_OCCUPANCY_HEATMAP": bool(occupancy_heatmap),
            "RENDER_OCCUPANCY_MODE": str(occupancy_mode or "auto"),
        }
    )
    if diag_only:
        out["RENDER_GIF"] = False
        out["RENDER_EPISODE_DIAG"] = True
    elif heatmap_only:
        out["RENDER_GIF"] = False
        out["RENDER_EPISODE_DIAG"] = False
    elif effective_gif_only or (clean_good_bad and occupancy_heatmap):
        # Good/Bad GIF ± heatmap: skip matplotlib episode-diag.
        out["RENDER_EPISODE_DIAG"] = False
    else:
        out["RENDER_EPISODE_DIAG"] = True
    return out


def _build_eval_env(config: dict[str, Any]):
    register_harvest_env()
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if reputation_v2_enabled(config):
        env = ReputationV2Wrapper(env, reputation_v2_config_from_dict(config))
    elif spatial_reputation_enabled(config):
        env = ReputationWrapper(env, reputation_config_from_dict(config))
    return env


def _hydra_config_name(run_dir: Path) -> str | None:
    hydra_path = run_dir / ".hydra" / "hydra.yaml"
    if not hydra_path.is_file():
        return None
    with hydra_path.open(encoding="utf-8") as fh:
        hydra_cfg = yaml.safe_load(fh) or {}
    job = hydra_cfg.get("hydra", {}).get("job", {})
    return job.get("config_name")


def render_ippo_checkpoint(
    *,
    run_dir: str | Path,
    checkpoint: str | Path,
    out_dir: str | Path | None = None,
    update: int | None = None,
    seed: int | None = None,
    label: str = "ippo",
    diag_only: bool = False,
    overlay_reputation: bool = False,
    gif_frames: int | None = None,
    clean_good_bad: bool = False,
    gif_only: bool = False,
    occupancy_heatmap: bool = False,
    occupancy_mode: str = "auto",
) -> list[Path]:
    """Run greedy rollout GIF and/or episode diag for one checkpoint.

    Returns paths written under ``out_dir`` (default: ``<run_dir>/gifs/render_uXXXXXX``).

    ``overlay_reputation``
        Force ``ReputationWrapper`` on eval even when the checkpoint was trained
        without reputation. Behaviour is unchanged when the wrapper is off;
        rep updates are recorded for episode-diag plots.
    ``diag_only``
        Skip GIF files; write episode-diag PNG/CSV only.
    ``gif_frames``
        If set, save one short GIF with this many frames (first of episode) and
        still write the full-episode diag unless ``gif_only`` / ``clean_good_bad``.
    ``clean_good_bad``
        Grid-only GIF with Good=green / Bad=red agents and a compact legend.
        Implies GIF-only (no episode-diag).
    ``gif_only``
        Skip episode-diag; with ``gif_frames``, stop after enough frames.
    ``occupancy_heatmap``
        Full-episode cell-visit heatmap (same RNG as GIF when seed/update match).
    """
    run_dir = Path(run_dir)
    ckpt_path = resolve_checkpoint_path(checkpoint, run_dir=run_dir)
    run_config = load_run_config(run_dir)
    trained_from_run = spatial_reputation_enabled(run_config)
    trained_v2_from_run = reputation_v2_enabled(run_config)
    payload = load_checkpoint(ckpt_path)
    ckpt_cfg = payload.get("config") or {}
    if "ENABLE_REPUTATION" in ckpt_cfg:
        trained_with_rep = bool(ckpt_cfg["ENABLE_REPUTATION"])
    else:
        trained_with_rep = trained_from_run
    if "ENABLE_REPUTATION_V2" in ckpt_cfg:
        trained_with_v2 = bool(ckpt_cfg["ENABLE_REPUTATION_V2"])
    else:
        trained_with_v2 = trained_v2_from_run

    if (
        not overlay_reputation
        and "ENABLE_REPUTATION" in ckpt_cfg
        and trained_with_rep != trained_from_run
    ):
        raise ValueError(
            f"Checkpoint ENABLE_REPUTATION={trained_with_rep} but run config has "
            f"ENABLE_REPUTATION={trained_from_run}. Use the matching training run "
            "dir, or pass overlay_reputation=True to add a post-hoc rep side-channel."
        )
    if (
        "ENABLE_REPUTATION_V2" in ckpt_cfg
        and trained_with_v2 != trained_v2_from_run
    ):
        raise ValueError(
            f"Checkpoint ENABLE_REPUTATION_V2={trained_with_v2} but run config has "
            f"ENABLE_REPUTATION_V2={trained_v2_from_run}. Use the matching "
            "training hydra.run.dir."
        )

    config = _apply_render_defaults(
        run_config,
        diag_only=diag_only,
        overlay_reputation=overlay_reputation,
        gif_frames=gif_frames,
        clean_good_bad=clean_good_bad,
        gif_only=gif_only,
        occupancy_heatmap=occupancy_heatmap,
        occupancy_mode=occupancy_mode,
    )
    if overlay_reputation and not trained_with_rep:
        config["ENABLE_REPUTATION"] = True
        config["OVERLAY_REPUTATION"] = True
        if not config.get("REPUTATION"):
            config["REPUTATION"] = {"r_star": 0.2}
        print(
            "[render] OVERLAY_REPUTATION: env wrapper ON for diag plots"
        )
    if gif_frames is not None and not diag_only:
        print(
            f"[render] short GIF: first {int(gif_frames)} frames"
            + (
                " (GIF-only; no diag / early stop)"
                if not config.get("RENDER_EPISODE_DIAG", True)
                else " (full episode still used for diag)"
            )
        )
    if clean_good_bad and not diag_only:
        print("[render] clean Good/Bad GIF (grid + legend only)")
    if occupancy_heatmap:
        print("[render] full-episode occupancy heatmap (same RNG as GIF)")

    update_i = int(
        update
        if update is not None
        else payload.get("update", checkpoint_update_index(ckpt_path))
    )

    env = _build_eval_env(config)
    agent_keys = [str(a) for a in env.agents]
    params, _opt_state, _steps = restore_stacked_train_state(
        payload, agent_keys=agent_keys
    )

    network = ActorCritic(
        action_dim=env.action_space().n,
        activation=str(config.get("ACTIVATION", "relu")),
    )

    if out_dir is None:
        out_dir = run_dir / "gifs" / f"render_u{update_i:06d}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_name = _hydra_config_name(run_dir) or config.get("_CFG_NAME")
    renderer = GifRenderer(config, env, out_dir, cfg_name=cfg_name)

    rollout_seed = int(seed if seed is not None else config.get("SEED", 0))
    rng = jax.random.PRNGKey(rollout_seed + update_i + 9999)
    paths = renderer.render_ippo(
        params,
        update_i,
        network.apply,
        rng=rng,
        seed_i=0,
        label=label,
    )
    if not paths:
        raise RuntimeError(f"Render produced no outputs for {ckpt_path}")
    return paths
