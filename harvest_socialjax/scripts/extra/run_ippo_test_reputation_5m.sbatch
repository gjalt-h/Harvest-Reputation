#!/bin/bash
#SBATCH --job-name=ippo_rep_5m
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
# Reputation ippo_test: 5M steps, checkpoints every 10 updates (no mid-run GIFs).
# Post-hoc render: scripts/main/render_ippo_from_checkpoint.py or run_ippo_render_from_checkpoint.sbatch
# Expected ~1.5 h training; walltime 2× for safety.
#SBATCH --time=03:00:00
#SBATCH --output=logs/ippo_test_reputation_5m_%j.out
#SBATCH --error=logs/ippo_test_reputation_5m_%j.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

CONFIG_NAME="${CONFIG_NAME:-ippo_test_reputation_5m}"
SBATCH_FILE="${SBATCH_FILE:-run_ippo_test_reputation_5m.sbatch}"
OUT_DIR="${PROJECT_DIR}/outputs/ippo_test_reputation_5m_${SLURM_JOB_ID}"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${SLURM_JOB_ID}_rep_5m"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"
_WANDB_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/wandb_offline.sh"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="${RUN_LABEL:-ippo_test_reputation_5m_${SLURM_JOB_ID}}"
export RUN_NOTES="${RUN_NOTES:-Rep ON, 5M steps, checkpoints only (render GIFs post-hoc)}"

echo "=== ippo_test_reputation_5m started at $(date) on $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID  config=$CONFIG_NAME  out=$OUT_DIR"
echo "CHECKPOINT_EVERY=10  RENDER_GIF=false  (render later with render_ippo_from_checkpoint.py)"

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
    grep -vE 'Mixing Conda and module environments|servicedesk\.surfsara\.nl' "$_MOD_ERR" >&2 || true
fi
rm -f "$_MOD_ERR"

set +u; source ~/.bashrc 2>/dev/null || true; set -u
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-harvest_ippo_rep}"
export WANDB_TAGS="ippo,test,reputation,5m,checkpoints,num_seeds_1,seed_0"
export WANDB_NAME="${RUN_LABEL}"
mkdir -p "$WANDB_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

bash "$_REGISTER_SH" submitted || true

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1
echo "W&B project: $WANDB_PROJECT"

if [ "$WANDB_MODE" = "offline" ]; then
    # shellcheck source=/dev/null
    source "$_WANDB_SH"
    wandb_offline_job_start "$WANDB_DIR"
fi

HYDRA_EXTRA=()
if [ -n "${RUN_TWEAKS:-}" ]; then
    echo "RUN_TWEAKS=$RUN_TWEAKS"
    IFS=',' read -ra _pairs <<< "$RUN_TWEAKS"
    for pair in "${_pairs[@]}"; do
        pair="$(echo "$pair" | xargs)"
        [ -n "$pair" ] && HYDRA_EXTRA+=("$pair")
    done
fi

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name "$CONFIG_NAME" \
    hydra.run.dir="${OUT_DIR}" \
    "${HYDRA_EXTRA[@]}" \
    || TRAIN_EXIT=$?
export TRAIN_EXIT

echo "=== Training finished at $(date), exit=$TRAIN_EXIT ==="
echo "Checkpoints: ${OUT_DIR}/checkpoints/"
ls -la "${OUT_DIR}/checkpoints/" 2>/dev/null | tail -8 || echo "WARNING: no checkpoints"
echo "Post-hoc render example:"
echo "  export RUN_DIR=${OUT_DIR}"
echo "  export CHECKPOINT=checkpoints/ckpt_update_000046.pkl"
echo "  sbatch harvest_socialjax/scripts/main/run_ippo_render_from_checkpoint.sbatch"

if [ "$WANDB_MODE" = "offline" ]; then
    wandb_offline_job_finish "$WANDB_DIR"
fi

bash "$_REGISTER_SH" finished || true
exit $TRAIN_EXIT
