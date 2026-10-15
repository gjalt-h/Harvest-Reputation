#!/bin/bash
#SBATCH --job-name=ippo_n5_apa
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --array=0-2
# Independent IPPO — test layout, N=5, plus_patch density sweep, 50M steps:
#   idx 0 → 18 apples/agent  (density=1.0  → 90 apples)
#   idx 1 → 21 apples/agent  (density=21/18 → 105 apples)
#   idx 2 → 24.5 apples/agent (density=24.5/18 → ~122–123 apples)
# LR anneal horizon 100M → end LR = ½ initial at 50M.
#
# Continuable from latest checkpoint:
#   Stable OUT_DIR per (N, apa) — re-submit the same array index and the job
#   auto-loads OUT_DIR/checkpoints/ckpt_latest.pkl if present.
#   To train further than 50M after a finished run:
#     sbatch --export=ALL,TOTAL_TIMESTEPS=100000000 \
#       harvest_socialjax/scripts/extra/run_ippo_test_n5_apa_18_21_24p5_50m.sbatch
#SBATCH --time=08:00:00
#SBATCH --output=logs/ippo_n5_apa_%a_%A.out
#SBATCH --error=logs/ippo_n5_apa_%a_%A.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

_BASE_APA=18
N=5
LAYOUT_MODE="plus_patch"
TOTAL_TS="${TOTAL_TIMESTEPS:-50000000}"
ANNEAL_HORIZON="${ANNEAL_LR_TOTAL_TIMESTEPS:-100000000}"

IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
case "$IDX" in
    0) APA=18;   APA_TAG="18" ;;
    1) APA=21;   APA_TAG="21" ;;
    2) APA=24.5; APA_TAG="24p5" ;;
    *)
        echo "FATAL: array task $IDX out of range (0-2)"
        exit 1
        ;;
esac

DENSITY=$(python -c "print(${APA}/${_BASE_APA})")
TARGET_APPLES=$(python -c "print(int(round(${APA} * ${N})))")
TRAIN_SEED="${SEED:-0}"

NAME_BASE="ippo_n${N}_apa_${APA_TAG}_50m"
if [ "${TRAIN_SEED}" = "0" ]; then
    NAME_STEM="${NAME_BASE}"
else
    NAME_STEM="${NAME_BASE}_seed${TRAIN_SEED}"
fi
# Stable dir so re-submits resume the same run (not per-jobid).
OUT_DIR="${PROJECT_DIR}/outputs/${NAME_STEM}"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${NAME_STEM}"
CONFIG_NAME="ippo_test"
SBATCH_FILE="run_ippo_test_n5_apa_18_21_24p5_50m.sbatch"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"

CKPT_LATEST="${OUT_DIR}/checkpoints/ckpt_latest.pkl"
RESUME_ARGS=()
if [ -f "$CKPT_LATEST" ]; then
    export RESUME_CHECKPOINT="$CKPT_LATEST"
    RESUME_ARGS+=("RESUME_CHECKPOINT=${CKPT_LATEST}")
    echo "=== RESUME: found ${CKPT_LATEST} ==="
else
    unset RESUME_CHECKPOINT || true
    echo "=== FRESH: no checkpoint at ${CKPT_LATEST} ==="
fi

TAGS="ippo,test,independent,plus_patch,50m,n${N},apa${APA_TAG},density_sweep,supervisor_batch,continuable,seed${TRAIN_SEED}"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="${NAME_STEM}_${SLURM_JOB_ID}"
export RUN_TWEAKS="SEED=${TRAIN_SEED},ENV_NUM_AGENTS=${N},ENV_KWARGS.apple_density=${DENSITY},ENV_KWARGS.apple_layout_mode=${LAYOUT_MODE},apples_per_agent=${APA},target_apples=${TARGET_APPLES},TOTAL_TIMESTEPS=${TOTAL_TS},ANNEAL_LR_TOTAL_TIMESTEPS=${ANNEAL_HORIZON}"
export RUN_NOTES="N=5 plus_patch apa=${APA} dens=${DENSITY} seed=${TRAIN_SEED} ts=${TOTAL_TS} idx=${IDX}"

echo "=== ippo N=5 apa sweep started at $(date) on $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID  array=${IDX}  N=${N}  apa=${APA}  seed=${TRAIN_SEED}"
echo "  density=${DENSITY}  target_apples≈${TARGET_APPLES}  layout=${LAYOUT_MODE}"
echo "  TOTAL_TIMESTEPS=${TOTAL_TS}  ANNEAL_LR_TOTAL_TIMESTEPS=${ANNEAL_HORIZON}"
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
# Sweep-family label only; train script sets seed-agnostic W&B group per (N, dens) cell.
export WANDB_RUN_GROUP="ippo_test_n5_apa_density_50m"
export WANDB_NAME="${NAME_STEM}"
export WANDB_TAGS="${TAGS}"
mkdir -p "$OUT_DIR" "$WANDB_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

bash "$_REGISTER_SH" submitted || true

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name "$CONFIG_NAME" \
    SEED="${TRAIN_SEED}" \
    NUM_SEEDS=1 \
    ENV_NUM_AGENTS="${N}" \
    ENV_KWARGS.apple_density="${DENSITY}" \
    ENV_KWARGS.apple_layout_mode="${LAYOUT_MODE}" \
    TOTAL_TIMESTEPS="${TOTAL_TS}" \
    ANNEAL_LR=true \
    ANNEAL_LR_TOTAL_TIMESTEPS="${ANNEAL_HORIZON}" \
    RENDER_GIF=false \
    hydra.run.dir="${OUT_DIR}" \
    "${RESUME_ARGS[@]}" \
    || TRAIN_EXIT=$?
export TRAIN_EXIT

echo "=== Training finished at $(date), exit=$TRAIN_EXIT ==="
ls -la "${OUT_DIR}/checkpoints/" 2>/dev/null || echo "WARNING: no checkpoints/ directory"
echo "GIFs off during train; later e.g.:"
echo "  python harvest_socialjax/scripts/main/render_ippo_from_checkpoint.py \\"
echo "    --run-dir ${OUT_DIR} --checkpoint checkpoints/ckpt_latest.pkl"

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
