#!/bin/bash
#SBATCH --job-name=ippo_n4_tagbeam
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --array=0-4
# N=4 standard IPPO — five one-off ablations (not a factorial):
#   0: independent · tag_timeout=10s   (default beam 12×3, tag=25)
#   1: independent · tag_timeout=75s
#   2: independent · beam 6×1         (default tag=25)
#   3: independent · beam 24×7        (default tag=25)
#   4: shared-opt   · beam 24×7        (OPTIMIZE_SHARED_REWARDS)
# All other knobs: ippo_test baseline (density=1.0, regrowth=1.0, 72 apples).
#SBATCH --time=08:00:00
#SBATCH --output=logs/ippo_n4_tagbeam_%a_%A.out
#SBATCH --error=logs/ippo_n4_tagbeam_%a_%A.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

_N=4
IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
if [ "$IDX" -ge 5 ]; then
    echo "FATAL: array task $IDX out of range (0-4)"
    exit 1
fi

# Defaults = test preset (tag=25, beam=12×3, density=1.0, regrowth=1.0)
TAG_TIMEOUT=25
BEAM_L=12
BEAM_W=3
VARIANT=""
CONFIG_NAME="ippo_test"
MODE="independent"

case "$IDX" in
    0)
        VARIANT="tag10"
        TAG_TIMEOUT=10
        ;;
    1)
        VARIANT="tag75"
        TAG_TIMEOUT=75
        ;;
    2)
        VARIANT="beam6x1"
        BEAM_L=6
        BEAM_W=1
        ;;
    3)
        VARIANT="beam24x7"
        BEAM_L=24
        BEAM_W=7
        ;;
    4)
        VARIANT="beam24x7_shared"
        BEAM_L=24
        BEAM_W=7
        CONFIG_NAME="ippo_test_shared"
        MODE="shared"
        ;;
esac

SBATCH_FILE="run_ippo_test_n4_tag_beam_sweep.sbatch"
OUT_DIR="${PROJECT_DIR}/outputs/ippo_${MODE}_n4_${VARIANT}_${SLURM_JOB_ID}"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${SLURM_ARRAY_JOB_ID}_${IDX}_n4_tagbeam"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="ippo_${MODE}_n4_${VARIANT}_${SLURM_JOB_ID}"
export RUN_TWEAKS="ENV_NUM_AGENTS=${_N},+ENV_KWARGS.tag_timeout=${TAG_TIMEOUT},+ENV_KWARGS.beam_length=${BEAM_L},+ENV_KWARGS.beam_width=${BEAM_W},variant=${VARIANT},mode=${MODE}"
export RUN_NOTES="N=4 tag/beam ablation idx=${IDX} variant=${VARIANT} mode=${MODE}"

if [ "$MODE" = "shared" ]; then
    export RUN_TWEAKS="${RUN_TWEAKS},OPTIMIZE_SHARED_REWARDS=true"
    WANDB_GROUP="ippo_shared_n4_tag_beam"
else
    WANDB_GROUP="ippo_test_n4_tag_beam"
fi

echo "=== ippo N=4 tag/beam ablation started at $(date) on $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID  array=${IDX}  variant=${VARIANT}  mode=${MODE}"
echo "  config=${CONFIG_NAME}  N=${_N}  tag_timeout=${TAG_TIMEOUT}  beam=${BEAM_L}x${BEAM_W}"
echo "  out=$OUT_DIR"

module purge
module load 2025
_MOD_ERR=$(mktemp)
module load Miniconda3/25.5.1-1 2>"$_MOD_ERR"
if [ -s "$_MOD_ERR" ]; then
    grep -vE 'Mixing Conda and module environments|servicedesk\.surfsara\.nl' "$_MOD_ERR" >&2 || true
fi
rm -f "$_MOD_ERR"
eval "$(${CONDA_EXE} shell.bash hook)"
_ACT_ERR=$(mktemp)
source activate SocialJax 2>"$_ACT_ERR"
if [ -s "$_ACT_ERR" ]; then
    grep -vE 'Mixing Conda and module environments|servicedesk\.surfsara\.nl' "$_ACT_ERR" >&2 || true
fi
rm -f "$_ACT_ERR"

set +u; source ~/.bashrc 2>/dev/null || true; set -u
export WANDB_MODE=offline
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_RUN_GROUP="${WANDB_GROUP}"
export WANDB_NAME="${RUN_LABEL}"
export WANDB_TAGS="ippo,test,n4,${MODE},${VARIANT},tag_beam_ablation,supervisor_batch"
mkdir -p "$WANDB_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

bash "$_REGISTER_SH" submitted || true

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name "$CONFIG_NAME" \
    ENV_NUM_AGENTS="${_N}" \
    ENV_KWARGS.apple_layout_mode=auto \
    +ENV_KWARGS.tag_timeout="${TAG_TIMEOUT}" \
    +ENV_KWARGS.beam_length="${BEAM_L}" \
    +ENV_KWARGS.beam_width="${BEAM_W}" \
    RENDER_GIF=true \
    RENDER_GIF_COUNT=5 \
    RENDER_GIF_START_STEPS=250 \
    RENDER_GIF_TAIL_STEPS=100 \
    RENDER_GIF_TAIL=true \
    hydra.run.dir="${OUT_DIR}" \
    || TRAIN_EXIT=$?
export TRAIN_EXIT

echo "=== Training finished at $(date), exit=$TRAIN_EXIT ==="
ls -la "${OUT_DIR}/gifs/" 2>/dev/null || echo "WARNING: no gifs/ directory"

WANDB_RUN_ID=""
WANDB_RUNS=$(find "$WANDB_DIR" -type d -name "offline-run-*" 2>/dev/null)
if [ -n "$WANDB_RUNS" ]; then
    for run_dir in $WANDB_RUNS; do
        wandb sync "$run_dir" || echo "WARNING: sync failed for $run_dir"
        WANDB_RUN_ID=$(basename "$run_dir" | sed 's/.*-//')
    done
fi
export WANDB_RUN_ID

bash "$_REGISTER_SH" finished || true
exit $TRAIN_EXIT
