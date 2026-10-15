#!/bin/bash
#SBATCH --job-name=ippo_rep_25m
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --array=0-2
# Reputation IPPO 25M: N=4, plus_patch, 98 apples (~391 PPO updates).
# alpha_h sweep (alpha_z=0, enforcement strictness=0.05):
#   0 → alpha_h=0.1
#   1 → alpha_h=0.5
#   2 → alpha_h=0.8
# Metrics + checkpoints only (RENDER_GIF=false). Expected ~3–5 h/task on MIG.
#SBATCH --time=06:00:00
#SBATCH --output=logs/ippo_test_reputation_25m_%A_%a.out
#SBATCH --error=logs/ippo_test_reputation_25m_%A_%a.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

CONFIG_NAME="${CONFIG_NAME:-ippo_test_reputation_25m}"
SBATCH_FILE="${SBATCH_FILE:-run_ippo_test_reputation_25m.sbatch}"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"
_WANDB_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/wandb_offline.sh"

# Fixed for this sweep.
ENABLE_ENFORCEMENT=true
STRICTNESS=0.05
ALPHA_Z=0.0

IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
case "$IDX" in
  0)
    ALPHA_H=0.1
    ARM_TAG="ah0p1_az0_s0p05"
    RUN_NOTES="Rep ON, alpha_h=0.1 alpha_z=0 enforcement s=0.05, N=4 plus_patch 98 apples, 25M"
    ;;
  1)
    ALPHA_H=0.5
    ARM_TAG="ah0p5_az0_s0p05"
    RUN_NOTES="Rep ON, alpha_h=0.5 alpha_z=0 enforcement s=0.05, N=4 plus_patch 98 apples, 25M"
    ;;
  2)
    ALPHA_H=0.8
    ARM_TAG="ah0p8_az0_s0p05"
    RUN_NOTES="Rep ON, alpha_h=0.8 alpha_z=0 enforcement s=0.05, N=4 plus_patch 98 apples, 25M"
    ;;
  *)
    echo "FATAL: array task $IDX out of range (0-2)"
    exit 1
    ;;
esac

NAME_STEM="ippo_test_reputation_25m_${ARM_TAG}"
OUT_DIR="${PROJECT_DIR}/outputs/${NAME_STEM}_${SLURM_ARRAY_JOB_ID}_${IDX}"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${SLURM_ARRAY_JOB_ID}_${IDX}_rep25m_${ARM_TAG}"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="${NAME_STEM}_${SLURM_ARRAY_JOB_ID}_${IDX}"
export RUN_NOTES
export RUN_TWEAKS="REPUTATION.enable_enforcement=${ENABLE_ENFORCEMENT},REPUTATION.enforcement_strictness=${STRICTNESS},REPUTATION.alpha_h=${ALPHA_H},REPUTATION.alpha_z=${ALPHA_Z}"
export WANDB_TAGS="ippo,test,reputation,25m,n4,plus_patch,apples98,${ARM_TAG},num_seeds_1,seed_0"

echo "=== ippo_test_reputation_25m alpha_h sweep started at $(date) on $(hostname) ==="
echo "Array ${SLURM_ARRAY_JOB_ID}_${IDX}  config=$CONFIG_NAME  tag=$ARM_TAG"
echo "  TOTAL_TIMESTEPS=25M  ANNEAL_LR_TOTAL_TIMESTEPS=100M"
echo "  N=4  plus_patch  apples=98  ENABLE_REPUTATION=true"
echo "  alpha_h=${ALPHA_H}  alpha_z=${ALPHA_Z}  enable_enforcement=${ENABLE_ENFORCEMENT}  strictness=${STRICTNESS}"
echo "  RUN_TWEAKS=$RUN_TWEAKS"
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
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-harvest_ippo_rep}"
export WANDB_RUN_GROUP="ippo_test_reputation_25M_alpha_h"
export WANDB_NAME="${RUN_LABEL}"
export WANDB_SYNC_INTERVAL="${WANDB_SYNC_INTERVAL:-300}"
mkdir -p "$WANDB_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

bash "$_REGISTER_SH" submitted || true

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1
echo "W&B project: $WANDB_PROJECT  group: $WANDB_RUN_GROUP  arm: $ARM_TAG"

if [ "$WANDB_MODE" = "offline" ]; then
    # shellcheck source=/dev/null
    source "$_WANDB_SH"
    wandb_offline_job_start "$WANDB_DIR" "$WANDB_SYNC_INTERVAL"
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

if [ "$WANDB_MODE" = "offline" ]; then
    wandb_offline_job_finish "$WANDB_DIR"
fi

bash "$_REGISTER_SH" finished || true
exit $TRAIN_EXIT
