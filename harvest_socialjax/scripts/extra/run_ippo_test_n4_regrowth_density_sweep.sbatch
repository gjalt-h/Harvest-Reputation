#!/bin/bash
#SBATCH --job-name=ippo_rg_ad
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --array=0-7
# Independent IPPO N=4 — two 1D sweeps (8 jobs total, not a full factorial):
#   idx 0–3: regrowth_scale ∈ {0.6, 0.8, 1.25, 1.5}  (apple_density=1.0 → 18/agent)
#   idx 4–7: apples/agent ∈ {10, 14, 22, 26}         (regrowth_scale=1.0)
# Orchard layout: auto — scaffold only at 18 apa; density axis (10/14/22/26) → plus_patch.
# Supervisor-standard PPO batch from ippo_test.yaml.
#SBATCH --time=08:00:00
#SBATCH --output=logs/ippo_test_n4_rg_ad_%a_%A.out
#SBATCH --error=logs/ippo_test_n4_rg_ad_%a_%A.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

_N=4
_BASE_APA=18
_REGROWTH_LIST=(0.6 0.8 1.25 1.5)
_APA_LIST=(10 14 22 26)

IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
if [ "$IDX" -ge 8 ]; then
    echo "FATAL: array task $IDX out of range (0-7)"
    exit 1
fi

# Defaults = ippo_test.yaml baseline for the axis not under sweep
REGROWTH="1.0"
APA="${_BASE_APA}"
SWEEP_AXIS=""

if [ "$IDX" -lt 4 ]; then
    SWEEP_AXIS="regrowth"
    REGROWTH="${_REGROWTH_LIST[$IDX]}"
else
    SWEEP_AXIS="density"
    AD_I=$((IDX - 4))
    APA="${_APA_LIST[$AD_I]}"
fi

DENSITY=$(python -c "print(${APA}/${_BASE_APA})")
TARGET_APPLES=$((APA * _N))
RG_TAG=$(echo "$REGROWTH" | tr '.' 'p')
AD_TAG="apa${APA}"

CONFIG_NAME="ippo_test"
SBATCH_FILE="run_ippo_test_n4_regrowth_density_sweep.sbatch"
OUT_DIR="${PROJECT_DIR}/outputs/ippo_test_n4_${SWEEP_AXIS}_rg${RG_TAG}_${AD_TAG}_${SLURM_JOB_ID}"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${SLURM_ARRAY_JOB_ID}_${IDX}_n4_rg_ad"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="ippo_n4_${SWEEP_AXIS}_rg${RG_TAG}_${AD_TAG}_${SLURM_JOB_ID}"
# auto: N=4 + apa=18 → scaffold; other apa on density axis → plus_patch.
LAYOUT_MODE="auto"
export RUN_TWEAKS="ENV_NUM_AGENTS=${_N},ENV_KWARGS.regrowth_scale=${REGROWTH},ENV_KWARGS.apple_density=${DENSITY},ENV_KWARGS.apple_layout_mode=${LAYOUT_MODE},apples_per_agent=${APA},target_apples=${TARGET_APPLES},sweep_axis=${SWEEP_AXIS}"
export RUN_NOTES="N=4 ${SWEEP_AXIS} sweep idx=${IDX} rg=${REGROWTH} apa=${APA} dens=${DENSITY} layout=${LAYOUT_MODE}"

echo "=== ippo_test N=4 ${SWEEP_AXIS} sweep started at $(date) on $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID  array=${IDX}  N=${_N}  axis=${SWEEP_AXIS}"
echo "  regrowth_scale=${REGROWTH}  apples_per_agent=${APA}  apple_density=${DENSITY}  target_apples=${TARGET_APPLES}"
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
export WANDB_RUN_GROUP="ippo_test_n4_regrowth_density"
export WANDB_NAME="${RUN_LABEL}"
export WANDB_TAGS="ippo,test,independent,n4,${SWEEP_AXIS}_sweep,rg_${RG_TAG},apa_${APA},supervisor_batch"
mkdir -p "$WANDB_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

bash "$_REGISTER_SH" submitted || true

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name "$CONFIG_NAME" \
    ENV_NUM_AGENTS="${_N}" \
    ENV_KWARGS.regrowth_scale="${REGROWTH}" \
    ENV_KWARGS.apple_density="${DENSITY}" \
    ENV_KWARGS.apple_layout_mode="${LAYOUT_MODE}" \
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
