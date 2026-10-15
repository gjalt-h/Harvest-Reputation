#!/usr/bin/env python3
"""Render GIF + episode diagnostic from a saved IPPO checkpoint.

Training should use ``RENDER_GIF=false`` and ``CHECKPOINT_EVERY>0``. After a run
finishes, render selected checkpoints on demand (GPU or CPU).

Examples::

  # Short 100-frame GIF + full-episode reputation diag (post-hoc overlay)
  python harvest_socialjax/scripts/main/render_ippo_from_checkpoint.py \\
      --run-dir outputs/ippo_test_n_shared_N4_25963068 \\
      --checkpoint checkpoints/ckpt_latest.pkl \\
      --updates 0,50,250 \\
      --overlay-reputation --gif-frames 100

  # Episode-diag only (no GIF)
  python harvest_socialjax/scripts/main/render_ippo_from_checkpoint.py \\
      --run-dir outputs/ippo_test_n_shared_N4_25963068 \\
      --checkpoint checkpoints/ckpt_latest.pkl \\
      --updates 0,50,250 \\
      --diag-only --overlay-reputation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harvest_socialjax.algorithms.ippo.render_from_checkpoint import (  # noqa: E402
    checkpoint_update_index,
    render_ippo_checkpoint,
    resolve_checkpoint_path,
)


def _parse_updates(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Greedy-eval GIF + reputation episode diag from IPPO checkpoint"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training hydra output dir (contains .hydra/config.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to ckpt .pkl (absolute, or relative to run-dir/checkpoints/)",
    )
    parser.add_argument(
        "--updates",
        default=None,
        help="Comma-separated PPO update indices to render (default: checkpoint only)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <run-dir>/gifs/render_uXXXXXX per update)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Rollout PRNG seed base (default: training SEED)",
    )
    parser.add_argument(
        "--diag-only",
        "--no-gif",
        dest="diag_only",
        action="store_true",
        help="Skip GIF encode; write episode-diag PNG/CSV only",
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=None,
        help=(
            "Write one short GIF with this many frames (first of the episode) "
            "and still run the full episode for episode-diag. Default: full "
            "start/tail GIF windows from the run config. Ignored with --diag-only."
        ),
    )
    parser.add_argument(
        "--overlay-reputation",
        action="store_true",
        help=(
            "Wrap eval with ReputationWrapper even if the checkpoint was trained "
            "without reputation. Policy stays spatial-only; rep is a side-channel "
            "for episode-diag plots (and on-GIF labels when --gif-frames is set)."
        ),
    )
    parser.add_argument(
        "--clean-good-bad",
        action="store_true",
        help=(
            "Grid-only GIF: colour Good agents green / Bad agents red, add a "
            "compact legend, and omit title bar / numeric reputation panel. "
            "Implies --gif-only (no episode-diag)."
        ),
    )
    parser.add_argument(
        "--gif-only",
        action="store_true",
        help=(
            "Skip episode-diag PNG/CSV. With --gif-frames, stop the rollout "
            "once enough frames are captured (much faster)."
        ),
    )
    parser.add_argument(
        "--occupancy-heatmap",
        action="store_true",
        help=(
            "Accumulate agent visit counts over the full episode and write an "
            "occupancy heatmap PNG (+ .npz). Uses the same rollout RNG as the GIF "
            "(SEED + update + 9999). Alone (no --clean-good-bad / --gif-frames) "
            "this is heatmap-only; combine with GIF flags to also encode a GIF."
        ),
    )
    parser.add_argument(
        "--occupancy-mode",
        default="auto",
        choices=("auto", "police", "all", "agents"),
        help=(
            "Heatmap layout: auto (police if police run else per-agent panels), "
            "agents (2×2 per agent), all (sum), police (police only)."
        ),
    )
    args = parser.parse_args()

    if args.diag_only and args.gif_frames is not None:
        print("[render] WARNING: --diag-only ignores --gif-frames")

    run_dir = Path(args.run_dir)
    ckpt_path = resolve_checkpoint_path(args.checkpoint, run_dir=run_dir)
    updates = _parse_updates(args.updates)

    if updates is None:
        updates = [checkpoint_update_index(ckpt_path)]

    all_paths: list[Path] = []
    for update_i in updates:
        candidate = run_dir / "checkpoints" / f"ckpt_update_{update_i:06d}.pkl"
        if candidate.is_file():
            ckpt_for_update = candidate
        elif len(updates) == 1:
            ckpt_for_update = ckpt_path
        else:
            raise FileNotFoundError(
                f"Missing checkpoint for update {update_i}: {candidate}. "
                "Train with CHECKPOINT_EVERY>0 or pass a single --updates value."
            )
        out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else run_dir / "gifs" / f"render_u{update_i:06d}"
        )
        print(
            f"[render] run_dir={run_dir} ckpt={ckpt_for_update} out={out_dir} "
            f"diag_only={args.diag_only} overlay_reputation={args.overlay_reputation} "
            f"clean_good_bad={args.clean_good_bad} gif_only={args.gif_only} "
            f"gif_frames={args.gif_frames} occupancy_heatmap={args.occupancy_heatmap} "
            f"occupancy_mode={args.occupancy_mode}"
        )
        paths = render_ippo_checkpoint(
            run_dir=run_dir,
            checkpoint=ckpt_for_update,
            out_dir=out_dir,
            update=update_i,
            seed=args.seed,
            diag_only=bool(args.diag_only),
            overlay_reputation=bool(args.overlay_reputation),
            gif_frames=args.gif_frames,
            clean_good_bad=bool(args.clean_good_bad),
            gif_only=bool(args.gif_only),
            occupancy_heatmap=bool(args.occupancy_heatmap),
            occupancy_mode=str(args.occupancy_mode),
        )
        all_paths.extend(paths)
        for path in paths:
            print(f"[render] saved {path}")

    print(f"[render] done ({len(all_paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
