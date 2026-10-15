#!/bin/bash
#SBATCH --job-name=ippo_rep50_cmp
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
# 50M two-way compare (781 PPO updates each).
#   idx 0 → ippo_test_baseline_50m.yaml        N=4 harvesters, rep OFF (control)
#   idx 1 → ippo_test_reputation_50m.yaml      N=4 harvesters, rep ON
# Metrics-only (RENDER_GIF=false); W&B offline + sync every WANDB_SYNC_INTERVAL sec.
#SBATCH --array=0-1
#SBATCH --time=10:00:00
#SBATCH --output=logs/ippo_rep50_compare_%A_%a.out
#SBATCH --error=logs/ippo_rep50_compare_%A_%a.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

IDX="${SLURM_ARRAY_TASK_ID:-0}"
case "$IDX" in
  0)
    CONFIG_NAME="ippo_test_baseline_50m"
    RUN_TAG="n4_no_rep"
    RUN_NOTES="N=4 harvesters, rep OFF, 50M steps (control — same env as rep runs)"
    WANDB_EXTRA_TAGS="no_reputation,n4"
    WANDB_PROJECT="harvest_ippo"
    ;;
  1)
    CONFIG_NAME="ippo_test_reputation_50m"
    RUN_TAG="n4_rep"
    RUN_NOTES="N=4 harvesters, rep ON, 50M steps"
    WANDB_EXTRA_TAGS="reputation,n4"
    WANDB_PROJECT="harvest_ippo_rep"
    ;;
  *)
    echo "FATAL: array task $IDX out of range (0-1)"
    exit 1
    ;;
esac

SBATCH_FILE="${SBATCH_FILE:-run_ippo_test_reputation_50m_compare.sbatch}"
OUT_DIR="${PROJECT_DIR}/outputs/ippo_${RUN_TAG}_${SLURM_ARRAY_JOB_ID}_${IDX}"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${SLURM_ARRAY_JOB_ID}_${IDX}_${RUN_TAG}"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"
_WANDB_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/wandb_offline.sh"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="ippo_${RUN_TAG}_${SLURM_ARRAY_JOB_ID}_${IDX}"
export RUN_NOTES

echo "=== ippo 50M compare started at $(date) on $(hostname) ==="
echo "Array ${SLURM_ARRAY_JOB_ID}_${IDX}  config=$CONFIG_NAME  tag=$RUN_TAG"
echo "  TOTAL_TIMESTEPS=50M  RENDER_GIF=false  CHECKPOINT_EVERY=50"
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
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_GROUP="ippo_reputation_compare_50M"
export WANDB_NAME="${RUN_LABEL}"
export WANDB_TAGS="ippo,test,50m,${WANDB_EXTRA_TAGS},compare,supervisor_batch"
export WANDB_SYNC_INTERVAL="${WANDB_SYNC_INTERVAL:-300}"
mkdir -p "$WANDB_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

bash "$_REGISTER_SH" submitted || true

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1
echo "W&B project: $WANDB_PROJECT"

# shellcheck source=/dev/null
source "$_WANDB_SH"
wandb_offline_job_start "$WANDB_DIR" "$WANDB_SYNC_INTERVAL"

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name "$CONFIG_NAME" \
    hydra.run.dir="${OUT_DIR}" \
    || TRAIN_EXIT=$?
export TRAIN_EXIT

wandb_offline_job_finish "$WANDB_DIR"

echo "=== Training finished at $(date), exit=$TRAIN_EXIT ==="
echo "Checkpoints: ${OUT_DIR}/checkpoints/"
ls -la "${OUT_DIR}/checkpoints/" 2>/dev/null | tail -5 || echo "WARNING: no checkpoints"

bash "$_REGISTER_SH" finished || true
exit $TRAIN_EXIT
