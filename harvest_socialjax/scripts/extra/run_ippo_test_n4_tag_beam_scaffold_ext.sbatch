#!/bin/bash
#SBATCH --job-name=ippo_n4_tb_ext
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --array=0-3
# N=4 scaffold orchard — extra tag/beam ablations (independent only):
#   0: beam 8×3   (tag=25 default)
#   1: beam 18×5  (tag=25)
#   2: tag_timeout=50  (beam 12×3 default)
#   3: tag_timeout=100 (beam 12×3)
# apple_layout_mode=scaffold (N=4 × 18 apa).
# Inherits ippo_test defaults: 50M steps, ANNEAL_LR horizon 100M (end LR = ½).
# Same W&B group as prior tag/beam sweep → plots together.
#SBATCH --time=05:00:00
#SBATCH --output=logs/ippo_n4_tagbeam_ext_%a_%A.out
#SBATCH --error=logs/ippo_n4_tagbeam_ext_%a_%A.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

_N=4
LAYOUT_MODE="scaffold"
IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
if [ "$IDX" -ge 4 ]; then
    echo "FATAL: array task $IDX out of range (0-3)"
    exit 1
fi

TAG_TIMEOUT=25
BEAM_L=12
BEAM_W=3
VARIANT=""
CONFIG_NAME="ippo_test"
MODE="independent"

case "$IDX" in
    0)
        VARIANT="beam8x3"
        BEAM_L=8
        BEAM_W=3
        ;;
    1)
        VARIANT="beam18x5"
        BEAM_L=18
        BEAM_W=5
        ;;
    2)
        VARIANT="tag50"
        TAG_TIMEOUT=50
        ;;
    3)
        VARIANT="tag100"
        TAG_TIMEOUT=100
        ;;
esac

SBATCH_FILE="run_ippo_test_n4_tag_beam_scaffold_ext.sbatch"
OUT_DIR="${PROJECT_DIR}/outputs/ippo_${MODE}_n4_${VARIANT}_${SLURM_JOB_ID}"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${SLURM_ARRAY_JOB_ID}_${IDX}_n4_tagbeam_ext"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="ippo_${MODE}_n4_${VARIANT}_${SLURM_JOB_ID}"
export RUN_TWEAKS="ENV_NUM_AGENTS=${_N},ENV_KWARGS.apple_layout_mode=${LAYOUT_MODE},+ENV_KWARGS.tag_timeout=${TAG_TIMEOUT},+ENV_KWARGS.beam_length=${BEAM_L},+ENV_KWARGS.beam_width=${BEAM_W},variant=${VARIANT},mode=${MODE},TOTAL_TIMESTEPS=50M,ANNEAL_LR_TOTAL_TIMESTEPS=100M"
export RUN_NOTES="N=4 scaffold tag/beam ext idx=${IDX} variant=${VARIANT} 50M/anneal100M"

export WANDB_RUN_GROUP="ippo_test_n4_tag_beam"
export WANDB_NAME="${RUN_LABEL}"
export WANDB_TAGS="ippo,test,n4,${MODE},${VARIANT},tag_beam_ablation,scaffold,50m,supervisor_batch"

echo "=== ippo N=4 scaffold tag/beam ext started at $(date) on $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID  array=${IDX}  variant=${VARIANT}  layout=${LAYOUT_MODE}"
echo "  tag_timeout=${TAG_TIMEOUT}  beam=${BEAM_L}x${BEAM_W}"
echo "  config=${CONFIG_NAME} (50M steps, ANNEAL_LR horizon 100M)  out=$OUT_DIR"

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
mkdir -p "$WANDB_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

bash "$_REGISTER_SH" submitted || true

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name "$CONFIG_NAME" \
    ENV_NUM_AGENTS="${_N}" \
    ENV_KWARGS.apple_layout_mode="${LAYOUT_MODE}" \
    +ENV_KWARGS.tag_timeout="${TAG_TIMEOUT}" \
    +ENV_KWARGS.beam_length="${BEAM_L}" \
    +ENV_KWARGS.beam_width="${BEAM_W}" \
    +variant="${VARIANT}" \
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
