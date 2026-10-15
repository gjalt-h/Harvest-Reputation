#!/bin/bash
#SBATCH --job-name=ippo_blob98
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
# Independent IPPO 50M — N=4, 98 apples, scaffold densify ("one large blob").
# Contrast with plus_patch 98 (canonical ippo_test / apa 24.5).
# SEED=0; OPTIMIZE_SHARED_REWARDS=false; RENDER_GIF=false; continuable.
#SBATCH --time=08:00:00
#SBATCH --output=logs/ippo_n4_blob98_%j.out
#SBATCH --error=logs/ippo_n4_blob98_%j.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

SCRIPT_DIR="${PROJECT_DIR}/harvest_socialjax/scripts"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sbatch/plus98_verified_pins.sh"

_N=4
TARGET_APPLES=98
_BASE_APA=18
APA=24.5
LAYOUT_MODE="scaffold"
TOTAL_TS="${TOTAL_TIMESTEPS:-50000000}"
ANNEAL_HORIZON="${ANNEAL_LR_TOTAL_TIMESTEPS:-100000000}"
TRAIN_SEED="${SEED:-0}"
CONFIG_NAME="ippo_test"
SBATCH_FILE="run_ippo_n4_blob98_scaffold_50m.sbatch"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"

DENSITY=$(python -c "print(${TARGET_APPLES}/(${_BASE_APA}*${_N}))")
NAME_BASE="ippo_indep_n${_N}_blob98_scaffold_50m"
if [ "${TRAIN_SEED}" = "0" ]; then
    NAME_STEM="${NAME_BASE}"
else
    NAME_STEM="${NAME_BASE}_seed${TRAIN_SEED}"
fi
OUT_DIR="${PROJECT_DIR}/outputs/${NAME_STEM}"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${NAME_STEM}"

CKPT_LATEST="${OUT_DIR}/checkpoints/ckpt_latest.pkl"
RESUME_ARGS=()
if [ -f "$CKPT_LATEST" ]; then
    export RESUME_CHECKPOINT="$CKPT_LATEST"
    RESUME_ARGS+=("RESUME_CHECKPOINT=${CKPT_LATEST}")
    echo "=== RESUME: ${CKPT_LATEST} ==="
else
    unset RESUME_CHECKPOINT || true
    echo "=== FRESH: no checkpoint yet ==="
fi

TAGS="ippo,test,independent,scaffold,blob98,fixed98,50m,n${_N},apa24p5,supervisor_batch,continuable,seed${TRAIN_SEED}"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="${NAME_STEM}_${SLURM_JOB_ID}"
export RUN_TWEAKS="SEED=${TRAIN_SEED},ENV_NUM_AGENTS=${_N},ENV_KWARGS.apple_density=${DENSITY},ENV_KWARGS.apple_layout_mode=${LAYOUT_MODE},apples_per_agent=${APA},target_apples=${TARGET_APPLES},TOTAL_TIMESTEPS=${TOTAL_TS}"
export RUN_NOTES="Indep N=4 scaffold blob 98 apples dens=${DENSITY} seed=${TRAIN_SEED}"
export WANDB_TAGS="${TAGS}"

echo "=== N=4 blob98 scaffold started at $(date) on $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID  layout=${LAYOUT_MODE}  apples=${TARGET_APPLES}  dens=${DENSITY}"
echo "  TOTAL_TIMESTEPS=${TOTAL_TS}  out=$OUT_DIR"

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
export WANDB_RUN_GROUP="ippo_n4_blob98_50m"
export WANDB_NAME="${NAME_STEM}"
mkdir -p "$OUT_DIR" "$WANDB_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

bash "$_REGISTER_SH" submitted || true

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name "$CONFIG_NAME" \
    SEED="${TRAIN_SEED}" \
    ENV_NUM_AGENTS="${_N}" \
    ENV_KWARGS.apple_density="${DENSITY}" \
    ENV_KWARGS.apple_layout_mode="${LAYOUT_MODE}" \
    ENV_KWARGS.regrowth_scale=1.0 \
    +apples_per_agent="${APA}" \
    +target_apples="${TARGET_APPLES}" \
    "${PLUS98_IPPO_OVERRIDES[@]}" \
    hydra.run.dir="${OUT_DIR}" \
    "${RESUME_ARGS[@]}" \
    || TRAIN_EXIT=$?
export TRAIN_EXIT

echo "=== Training finished at $(date), exit=$TRAIN_EXIT ==="
ls -la "${OUT_DIR}/checkpoints/" 2>/dev/null | tail -8 || echo "WARNING: no checkpoints"

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
