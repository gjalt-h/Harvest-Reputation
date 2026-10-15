#!/bin/bash
#SBATCH --job-name=ippo_n5_regrow15
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=06:00:00
#SBATCH --output=logs/ippo_n5_regrow15_%j.out
#SBATCH --error=logs/ippo_n5_regrow15_%j.err
#
# Single run: N=5 agents, gamma=0.99, ent_coef=0.01, regrowth_scale=1.5,
# NUM_SEEDS=3, NUM_ENVS=32.
# Partition MUST be gpu_mig. Never --partition=a100.

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

WANDB_NAME="n5_gamma0.99_ent0.01_regrow1.5"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${SLURM_JOB_ID}_n5regrow"
HYDRA_DIR="${PROJECT_DIR}/outputs/ippo_toc_adhoc/${WANDB_NAME}"

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
export WANDB_RUN_GROUP="ippo_toc_harvest_sweep"
export WANDB_NAME
export WANDB_TAGS="ippo,toc,n5,regrowth_1.5,num_seeds_3"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0
mkdir -p "$WANDB_DIR" "$HYDRA_DIR"

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name ippo_test \
    ENV_NUM_AGENTS=5 \
    GAMMA=0.99 \
    ENT_COEF=0.01 \
    SEED=0 \
    NUM_ENVS=32 \
    NUM_SEEDS=3 \
    ENV_KWARGS.regrowth_scale=1.5 \
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
