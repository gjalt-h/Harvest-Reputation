#!/bin/bash
#SBATCH --job-name=ippo_shared_n4
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=06:00:00
#SBATCH --output=logs/ippo_shared_n4_%j.out
#SBATCH --error=logs/ippo_shared_n4_%j.err
#
# IPPO with SHARED rewards, N=4 agents, test-scale (12×24), NUM_SEEDS=3.

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

WANDB_NAME="ippo_shared_n4"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${SLURM_JOB_ID}_shared_n4"
HYDRA_DIR="${PROJECT_DIR}/outputs/ippo_shared/${WANDB_NAME}"

echo "=== ${WANDB_NAME} started at $(date) on $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID  partition=${SLURM_JOB_PARTITION:-unset}"

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
export WANDB_DIR
export WANDB_RUN_GROUP="ippo_shared_rewards"
export WANDB_NAME
export WANDB_TAGS="ippo,shared_rewards,n4,num_seeds_3"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0
mkdir -p "$WANDB_DIR" "$HYDRA_DIR"

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name ippo_test_shared \
    ENV_NUM_AGENTS=4 \
    ENV_KWARGS.apple_layout_mode=auto \
    SEED=0 \
    NUM_SEEDS=3 \
    NUM_ENVS=32 \
    VERBOSE_LOGGING=true \
    LOG_PER_AGENT=false \
    LOG_GIFT_METRICS=false \
    USE_LOCAL_DASHBOARD=false \
    hydra.run.dir="${HYDRA_DIR}" \
    > "${PROJECT_DIR}/logs/${WANDB_NAME}.out" \
    2> "${PROJECT_DIR}/logs/${WANDB_NAME}.err" || TRAIN_EXIT=$?

echo "=== ${WANDB_NAME} finished at $(date), exit=$TRAIN_EXIT ==="
WANDB_RUNS=$(find "$WANDB_DIR" -type d -name "offline-run-*" 2>/dev/null || true)
if [ -n "${WANDB_RUNS}" ]; then
    for run_dir in $WANDB_RUNS; do
        wandb sync "$run_dir" || echo "WARNING: sync failed for $run_dir"
    done
fi
exit $TRAIN_EXIT
