#!/bin/bash
#SBATCH --job-name=ippo_n4_regrow
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/ippo_n4_regrow_%a_%A.out
#SBATCH --error=logs/ippo_n4_regrow_%a_%A.err
#
# N=4 agents, gamma=0.99, ent_coef=0.01, sweep regrowth_scale 0.5..1.0 step 0.1.
# Array: 0→0.5  1→0.6  2→0.7  3→0.8  4→0.9  5→1.0
# NUM_SEEDS=3 vmapped, NUM_ENVS=32.
# Partition MUST be gpu_mig. Never --partition=a100.

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

REGROWTH_VALUES=(0.5 0.6 0.7 0.8 0.9 1.0)
IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
REGROWTH="${REGROWTH_VALUES[$IDX]}"

WANDB_NAME="n4_gamma0.99_ent0.01_regrow${REGROWTH}"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${SLURM_JOB_ID}_n4r${IDX}"
HYDRA_DIR="${PROJECT_DIR}/outputs/ippo_toc_adhoc/${WANDB_NAME}"
PER_CFG_OUT="${PROJECT_DIR}/logs/${WANDB_NAME}.out"
PER_CFG_ERR="${PROJECT_DIR}/logs/${WANDB_NAME}.err"

echo "=== ${WANDB_NAME} started at $(date) on $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID  array=${IDX}  partition=${SLURM_JOB_PARTITION:-unset}"
echo "REGROWTH=${REGROWTH}"

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
export WANDB_TAGS="ippo,toc,n4,regrowth_sweep,regrowth_${REGROWTH},num_seeds_3"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0
mkdir -p "$WANDB_DIR" "$HYDRA_DIR"

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name ippo_test \
    ENV_NUM_AGENTS=4 \
    ENV_KWARGS.apple_layout_mode=auto \
    GAMMA=0.99 \
    ENT_COEF=0.01 \
    SEED=0 \
    NUM_ENVS=32 \
    NUM_SEEDS=3 \
    ENV_KWARGS.regrowth_scale="${REGROWTH}" \
    VERBOSE_LOGGING=true \
    LOG_PER_AGENT=false \
    LOG_GIFT_METRICS=false \
    USE_LOCAL_DASHBOARD=false \
    hydra.run.dir="${HYDRA_DIR}" \
    > "${PER_CFG_OUT}" \
    2> "${PER_CFG_ERR}" || TRAIN_EXIT=$?

echo "=== ${WANDB_NAME} finished at $(date), exit=$TRAIN_EXIT ==="
WANDB_RUNS=$(find "$WANDB_DIR" -type d -name "offline-run-*" 2>/dev/null || true)
if [ -n "${WANDB_RUNS}" ]; then
    for run_dir in $WANDB_RUNS; do
        wandb sync "$run_dir" || echo "WARNING: sync failed for $run_dir"
    done
fi
exit $TRAIN_EXIT
