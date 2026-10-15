#!/bin/bash
#SBATCH --job-name=ippo_n_sweep
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --array=0-2
# Independent IPPO: N=4,5,6 on test layout (ippo_test baseline, 1e8 steps).
# One MIG slice per array task (~8 GiB compile peak at N=5).
#SBATCH --time=08:00:00
#SBATCH --output=logs/ippo_test_n_indep_%a_%A.out
#SBATCH --error=logs/ippo_test_n_indep_%a_%A.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

_N_LIST=(4 5 6)
IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
if [ "$IDX" -ge "${#_N_LIST[@]}" ]; then
    echo "FATAL: array task $IDX out of range (0-$((${#_N_LIST[@]} - 1)))"
    exit 1
fi
N="${_N_LIST[$IDX]}"

# auto: scaffold only for N=4 × density=1.0 (18 apa); else plus_patch.
LAYOUT_MODE="auto"

CONFIG_NAME="ippo_test"
SBATCH_FILE="run_ippo_test_n_sweep.sbatch"
OUT_DIR="${PROJECT_DIR}/outputs/ippo_test_n_indep_N${N}_${SLURM_JOB_ID}"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${SLURM_ARRAY_JOB_ID}_${IDX}_n_indep"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="ippo_indep_n${N}_${SLURM_JOB_ID}"
export RUN_TWEAKS="ENV_NUM_AGENTS=${N},ENV_KWARGS.apple_layout_mode=${LAYOUT_MODE}"
export RUN_NOTES="Independent IPPO N-sweep task ${IDX} (N=${N}, layout=${LAYOUT_MODE})"

echo "=== ippo_test N-sweep (independent) started at $(date) on $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID  array=${IDX}  N=${N}  out=$OUT_DIR"

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
export WANDB_RUN_GROUP="ippo_test_n_indep"
export WANDB_NAME="${RUN_LABEL}"
export WANDB_TAGS="ippo,test,independent,n_sweep,n${N},supervisor_batch"
mkdir -p "$WANDB_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

bash "$_REGISTER_SH" submitted || true

nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name "$CONFIG_NAME" \
    ENV_NUM_AGENTS="${N}" \
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
