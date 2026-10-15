#!/bin/bash
#SBATCH --job-name=ippo_plus50m
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --array=0-9
# Independent IPPO — forced plus_patch layouts, 50M steps (LR anneal as 100M):
#   idx 0–5: N × apa  —  N∈{4,5,6} × apa∈{18,24}
#   idx 6–9: N=4 density extras — apa∈{10,14,22,26}  (24 reused from idx 1)
# idx 1 (N=4, apa=24) is dual-tagged for BOTH n_sweep and density plots.
# TOTAL_TIMESTEPS=50M; ANNEAL_LR_TOTAL_TIMESTEPS=100M → end LR = ½ initial
# (matches first half of prior 100M anneal-to-zero runs).
#SBATCH --time=05:00:00
#SBATCH --output=logs/ippo_plus_patch_50m_%a_%A.out
#SBATCH --error=logs/ippo_plus_patch_50m_%a_%A.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

_BASE_APA=18
LAYOUT_MODE="plus_patch"
TOTAL_TS=50000000
ANNEAL_HORIZON=100000000  # LR schedule as if training 100M; stop at 50M → LR/2

IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
if [ "$IDX" -ge 10 ]; then
    echo "FATAL: array task $IDX out of range (0-9)"
    exit 1
fi

# Default: N×apa cells
N=4
APA=18
SWEEP_ROLE="n_apa"   # n_apa | density | both
case "$IDX" in
    0) N=4; APA=18; SWEEP_ROLE="both" ;;  # n_sweep_apa18 middle + density baseline
    1) N=4; APA=24; SWEEP_ROLE="both" ;;  # n_sweep_apa24 + density (no duplicate train)
    2) N=5; APA=18; SWEEP_ROLE="n_apa" ;;
    3) N=5; APA=24; SWEEP_ROLE="n_apa" ;;
    4) N=6; APA=18; SWEEP_ROLE="n_apa" ;;
    5) N=6; APA=24; SWEEP_ROLE="n_apa" ;;
    6) N=4; APA=10; SWEEP_ROLE="density" ;;
    7) N=4; APA=14; SWEEP_ROLE="density" ;;
    8) N=4; APA=22; SWEEP_ROLE="density" ;;
    9) N=4; APA=26; SWEEP_ROLE="density" ;;
esac

DENSITY=$(python -c "print(${APA}/${_BASE_APA})")
TARGET_APPLES=$((APA * N))

# Naming: apa_XX parses in wandb_sweep_viz; dual runs embed n_and_density.
if [ "$SWEEP_ROLE" = "both" ]; then
    NAME_STEM="ippo_plus_n${N}_apa_${APA}_n_and_density"
    AXIS_TAG="density"
elif [ "$SWEEP_ROLE" = "density" ]; then
    NAME_STEM="ippo_plus_n${N}_density_apa_${APA}"
    AXIS_TAG="density"
else
    NAME_STEM="ippo_plus_n${N}_apa_${APA}"
    AXIS_TAG="n_apa"
fi

CONFIG_NAME="ippo_test"
SBATCH_FILE="run_ippo_test_plus_patch_n_apa_50m.sbatch"
OUT_DIR="${PROJECT_DIR}/outputs/${NAME_STEM}_${SLURM_JOB_ID}"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${SLURM_ARRAY_JOB_ID}_${IDX}_plus50m"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"

# Tags drive plot membership: dual run carries both n_apa_sweep and density_sweep.
TAGS="ippo,test,independent,plus_patch,50m,n${N},apa${APA},supervisor_batch"
case "$SWEEP_ROLE" in
    n_apa)
        TAGS="${TAGS},n_apa_sweep,n_sweep"
        ;;
    density)
        TAGS="${TAGS},apa_density_sweep,density_sweep"
        ;;
    both)
        TAGS="${TAGS},n_apa_sweep,n_sweep,apa_density_sweep,density_sweep"
        ;;
esac

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="${NAME_STEM}_${SLURM_JOB_ID}"
export RUN_TWEAKS="ENV_NUM_AGENTS=${N},ENV_KWARGS.apple_density=${DENSITY},ENV_KWARGS.apple_layout_mode=${LAYOUT_MODE},apples_per_agent=${APA},target_apples=${TARGET_APPLES},sweep_axis=${AXIS_TAG},sweep_role=${SWEEP_ROLE},TOTAL_TIMESTEPS=${TOTAL_TS},ANNEAL_LR_TOTAL_TIMESTEPS=${ANNEAL_HORIZON}"
export RUN_NOTES="plus_patch 50M idx=${IDX} role=${SWEEP_ROLE} N=${N} apa=${APA} dens=${DENSITY}"

echo "=== ippo plus_patch 50M started at $(date) on $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID  array=${IDX}  role=${SWEEP_ROLE}  N=${N}  apa=${APA}"
echo "  density=${DENSITY}  apples=${TARGET_APPLES}  layout=${LAYOUT_MODE}"
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
export WANDB_RUN_GROUP="ippo_test_plus_patch_50m"
export WANDB_NAME="${RUN_LABEL}"
export WANDB_TAGS="${TAGS}"
mkdir -p "$WANDB_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

bash "$_REGISTER_SH" submitted || true

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name "$CONFIG_NAME" \
    ENV_NUM_AGENTS="${N}" \
    ENV_KWARGS.apple_density="${DENSITY}" \
    ENV_KWARGS.apple_layout_mode="${LAYOUT_MODE}" \
    TOTAL_TIMESTEPS="${TOTAL_TS}" \
    ANNEAL_LR=true \
    ANNEAL_LR_TOTAL_TIMESTEPS="${ANNEAL_HORIZON}" \
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
