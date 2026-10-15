#!/usr/bin/env python3
"""Fork W&B history into a *new* run (source run left intact).

Copies logged rows with ``episode_count <= --max-ep`` into a fresh online run,
then prints ``FORKED_WANDB_ID=<id>`` for continue jobs to resume/append.

  WANDB_ENTITY=YOUR_ENTITY WANDB_PROJECT=YOUR_PROJECT \\
  python harvest_socialjax/scripts/fork_wandb_history.py \\
      --source-id SOURCE_RUN_ID --max-ep 24960 \\
      --name ippo_v2_..._drop_enf_from25k_... \\
      --group ippo_v2_drop_enf_continue_50M \\
      --tags ippo,drop_enf,forked_history
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

# Prefer env; do not hardcode a personal W&B entity for public release.
os.environ.setdefault("WANDB_PROJECT", os.environ.get("WANDB_PROJECT", "harvest_ippo_rep"))
os.environ.setdefault("WANDB__SERVICE_WAIT", "180")
_tmp = os.environ.get("TMPDIR", "")
if (not _tmp) or _tmp.startswith("/scratch-local") or _tmp.startswith("/scratch-node"):
    _home_tmp = os.path.join(os.path.expanduser("~"), ".cache", "wandb_tmp")
    os.makedirs(_home_tmp, exist_ok=True)
    os.environ["TMPDIR"] = _home_tmp


_SKIP_PREFIXES = ("_",)
_SKIP_KEYS = {
    "_step",
    "_runtime",
    "_timestamp",
    "_wandb",
}


def _is_loggable(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, (bool, int, float)):
        return True
    if isinstance(v, str):
        return False
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return True
        if isinstance(v, np.ndarray) and v.ndim == 0:
            return True
    except ImportError:
        pass
    return False


def _row_payload(row: dict, *, max_ep: int) -> dict | None:
    ep = row.get("episode_count")
    if ep is None:
        return None
    try:
        ep_i = int(float(ep))
    except (TypeError, ValueError):
        return None
    if ep_i > int(max_ep):
        return None
    out: dict[str, Any] = {"episode_count": ep_i}
    for k, v in row.items():
        if k in _SKIP_KEYS or k == "episode_count":
            continue
        if any(k.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if not _is_loggable(v):
            continue
        out[k] = v
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-id", required=True)
    p.add_argument("--max-ep", type=int, required=True)
    p.add_argument("--name", required=True)
    p.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY") or None,
        help="W&B entity (or set WANDB_ENTITY)",
    )
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "harvest_ippo_rep"))
    p.add_argument("--group", default=None)
    p.add_argument("--tags", default="ippo,drop_enf,forked_history,continue")
    p.add_argument("--notes", default=None)
    p.add_argument("--dir", default=None, help="local wandb dir (optional)")
    args = p.parse_args()
    if not args.entity:
        print(
            "FATAL: set WANDB_ENTITY or pass --entity (no personal default in public release)",
            file=sys.stderr,
        )
        return 2

    import wandb

    api = wandb.Api(timeout=120)
    src_path = f"{args.entity}/{args.project}/{args.source_id}"
    src = api.run(src_path)
    print(
        f"[fork] source={src_path} name={src.name!r} state={src.state} "
        f"max_ep={args.max_ep}",
        flush=True,
    )

    tags = [t.strip() for t in str(args.tags).split(",") if t.strip()]
    init_kwargs: dict[str, Any] = {
        "project": args.project,
        "entity": args.entity,
        "name": args.name,
        "tags": tags or None,
        "notes": args.notes
        or (
            f"Forked history from {args.source_id} up to episode_count≤{args.max_ep}; "
            "source run left intact. Second half continues with enf/police/peer OFF."
        ),
        "reinit": True,
        "resume": "allow",
        # Must be online so the continue job can resume=must this id.
        "mode": "online",
    }
    if args.group:
        init_kwargs["group"] = args.group
    if args.dir:
        init_kwargs["dir"] = args.dir
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    run = wandb.init(**init_kwargs)
    new_id = run.id
    print(f"[fork] new run id={new_id} name={run.name}", flush=True)
    wandb.define_metric("episode_count")
    wandb.define_metric("*", step_metric="episode_count")

    n_in = 0
    n_out = 0
    last_ep = -1
    try:
        for row in src.scan_history(page_size=1000):
            n_in += 1
            payload = _row_payload(row, max_ep=args.max_ep)
            if payload is None:
                continue
            ep_i = int(payload["episode_count"])
            wandb.log(payload, step=ep_i)
            n_out += 1
            last_ep = ep_i
            if n_out % 200 == 0:
                print(f"[fork] logged {n_out} rows (last ep={last_ep})…", flush=True)
    except Exception as exc:
        print(f"[fork] FATAL while copying history: {type(exc).__name__}: {exc}", flush=True)
        wandb.finish(exit_code=1)
        return 1

    print(
        f"[fork] done scanned={n_in} logged={n_out} last_ep={last_ep} "
        f"new_id={new_id}",
        flush=True,
    )
    # Leave run open? finish so training can resume=must cleanly.
    wandb.finish()
    # Machine-readable for sbatch capture:
    print(f"FORKED_WANDB_ID={new_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
