#!/bin/bash
#SBATCH --job-name=ippo_render
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
# Post-hoc GIF + reputation episode diag from a training checkpoint.
# Submit with RUN_DIR and CHECKPOINT set (see examples below).
# Expected ~15–45 min per checkpoint (full 1000-step greedy eval).
#SBATCH --time=01:30:00
#SBATCH --output=logs/ippo_render_ckpt_%j.out
#SBATCH --error=logs/ippo_render_ckpt_%j.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

# Required:
#   export RUN_DIR=outputs/ippo_test_reputation_5m_<jobid>
#   export CHECKPOINT=checkpoints/ckpt_update_000046.pkl
# Optional:
#   export UPDATES=0,50,250
#   export OUT_DIR=...
#   export DIAG_ONLY=1              # episode-diag only (no GIF)
#   export OVERLAY_REPUTATION=1     # post-hoc ReputationWrapper on non-rep ckpt
#   export GIF_FRAMES=100           # short preview GIF + full-episode diag

RUN_DIR="${RUN_DIR:-}"
CHECKPOINT="${CHECKPOINT:-}"
UPDATES="${UPDATES:-}"
OUT_DIR="${OUT_DIR:-}"
DIAG_ONLY="${DIAG_ONLY:-}"
OVERLAY_REPUTATION="${OVERLAY_REPUTATION:-}"
GIF_FRAMES="${GIF_FRAMES:-}"

if [ -z "$RUN_DIR" ] || [ -z "$CHECKPOINT" ]; then
    echo "FATAL: set RUN_DIR and CHECKPOINT before sbatch, e.g.:"
    echo "  export RUN_DIR=outputs/ippo_test_n_shared_N4_25963068"
    echo "  export CHECKPOINT=checkpoints/ckpt_latest.pkl"
    echo "  export UPDATES=0,50,250"
    echo "  export OVERLAY_REPUTATION=1"
    echo "  export GIF_FRAMES=100"
    echo "  sbatch harvest_socialjax/scripts/main/run_ippo_render_from_checkpoint.sbatch"
    exit 1
fi

echo "=== ippo render from checkpoint started at $(date) on $(hostname) ==="
echo "RUN_DIR=$RUN_DIR  CHECKPOINT=$CHECKPOINT  UPDATES=${UPDATES:-<single>}"
echo "DIAG_ONLY=${DIAG_ONLY:-0}  OVERLAY_REPUTATION=${OVERLAY_REPUTATION:-0}  GIF_FRAMES=${GIF_FRAMES:-default}"

module purge
module load 2025
module load Miniconda3/25.5.1-1
eval "$(${CONDA_EXE} shell.bash hook)"
source activate SocialJax
set +u; source ~/.bashrc 2>/dev/null || true; set -u
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

_CMD=(python harvest_socialjax/scripts/render_ippo_from_checkpoint.py
    --run-dir "$RUN_DIR"
    --checkpoint "$CHECKPOINT")
if [ -n "$UPDATES" ]; then
    _CMD+=(--updates "$UPDATES")
fi
if [ -n "$OUT_DIR" ]; then
    _CMD+=(--out-dir "$OUT_DIR")
fi
if [ -n "$DIAG_ONLY" ] && [ "$DIAG_ONLY" != "0" ]; then
    _CMD+=(--diag-only)
fi
if [ -n "$OVERLAY_REPUTATION" ] && [ "$OVERLAY_REPUTATION" != "0" ]; then
    _CMD+=(--overlay-reputation)
fi
if [ -n "$GIF_FRAMES" ]; then
    _CMD+=(--gif-frames "$GIF_FRAMES")
fi

"${_CMD[@]}"
EXIT=$?

echo "=== Render finished at $(date), exit=$EXIT ==="
exit $EXIT
