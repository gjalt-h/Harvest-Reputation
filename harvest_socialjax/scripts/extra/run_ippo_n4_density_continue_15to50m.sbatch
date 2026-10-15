#!/bin/bash
#SBATCH --job-name=ippo_d15to50
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --array=0-5
# Continue N=4 plus_patch density sweep 15M → 50M (indep + shared).
#   idx 0 → independent apa=18  (72 apples)
#   idx 1 → independent apa=21  (84 apples)
#   idx 2 → independent apa=24.5 (98 apples)  # fresh if no 15M ckpt
#   idx 3 → shared      apa=18  (72 apples)
#   idx 4 → shared      apa=21  (84 apples)
#   idx 5 → shared      apa=24.5 (98 apples)
# Weights: RESUME_CHECKPOINT via env only (not Hydra CLI).
# W&B: resume the *same* run id as the 15M source (not a new “seed”).
# Filesystem OUT_DIR is stable *_50m (≠ 15M jobid path) — naming only, SEED unchanged.
#SBATCH --time=08:00:00
#SBATCH --output=logs/ippo_density_15to50_%a_%A.out
#SBATCH --error=logs/ippo_density_15to50_%a_%A.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

SCRIPT_DIR="${PROJECT_DIR}/harvest_socialjax/scripts"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sbatch/plus98_verified_pins.sh"

_BASE_APA=18
_N=4
LAYOUT_MODE="plus_patch"
TOTAL_TS="${TOTAL_TIMESTEPS:-50000000}"
ANNEAL_HORIZON="${ANNEAL_LR_TOTAL_TIMESTEPS:-100000000}"

IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
case "$IDX" in
    0) MODE="indep";  APA=18;   APA_TAG="18";   TARGET_APPLES=72; CONFIG_NAME="ippo_test";        SHARED=false ;;
    1) MODE="indep";  APA=21;   APA_TAG="21";   TARGET_APPLES=84; CONFIG_NAME="ippo_test";        SHARED=false ;;
    2) MODE="indep";  APA=24.5; APA_TAG="24p5"; TARGET_APPLES=98; CONFIG_NAME="ippo_test";        SHARED=false ;;
    3) MODE="shared"; APA=18;   APA_TAG="18";   TARGET_APPLES=72; CONFIG_NAME="ippo_test_shared"; SHARED=true ;;
    4) MODE="shared"; APA=21;   APA_TAG="21";   TARGET_APPLES=84; CONFIG_NAME="ippo_test_shared"; SHARED=true ;;
    5) MODE="shared"; APA=24.5; APA_TAG="24p5"; TARGET_APPLES=98; CONFIG_NAME="ippo_test_shared"; SHARED=true ;;
    *)
        echo "FATAL: array task $IDX out of range (0-5)"
        exit 1
        ;;
esac

TRAIN_SEED="${SEED:-0}"
DENSITY=$(python -c "print(${TARGET_APPLES}/(${_BASE_APA}*${_N}))")
NAME_BASE="ippo_${MODE}_n${_N}_apa_${APA_TAG}_50m"
if [ "${TRAIN_SEED}" = "0" ]; then
    NAME_STEM="${NAME_BASE}"
else
    NAME_STEM="${NAME_BASE}_seed${TRAIN_SEED}"
fi
OUT_DIR="${PROJECT_DIR}/outputs/${NAME_STEM}"
# Default offline dir for *new* runs; overwritten below when we resume a 15M W&B id.
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${NAME_STEM}"
SBATCH_FILE="run_ippo_n4_density_continue_15to50m.sbatch"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"

# ---- checkpoint source (env-only resume) ----
CKPT_LOCAL="${OUT_DIR}/checkpoints/ckpt_latest.pkl"
SOURCE_STEM="ippo_${MODE}_15m_apa_${APA_TAG}"
RESUME_CKPT=""
SOURCE_RUN_DIR=""
if [ -f "$CKPT_LOCAL" ]; then
    RESUME_CKPT="$CKPT_LOCAL"
    SOURCE_RUN_DIR="$OUT_DIR"
else
    shopt -s nullglob
    _cands=( "${PROJECT_DIR}/outputs/${SOURCE_STEM}_"*/checkpoints/ckpt_latest.pkl )
    shopt -u nullglob
    if [ "${#_cands[@]}" -gt 0 ]; then
        SOURCE_RUN_DIR=$(ls -dt "${PROJECT_DIR}/outputs/${SOURCE_STEM}_"*/ 2>/dev/null | head -1)
        SOURCE_RUN_DIR="${SOURCE_RUN_DIR%/}"
        if [ -n "$SOURCE_RUN_DIR" ] && [ -f "${SOURCE_RUN_DIR}/checkpoints/ckpt_latest.pkl" ]; then
            RESUME_CKPT="${SOURCE_RUN_DIR}/checkpoints/ckpt_latest.pkl"
        fi
    fi
fi

ALLOW_FRESH=0
if [ "$IDX" = "2" ]; then
    ALLOW_FRESH=1
fi

unset RESUME_CHECKPOINT WANDB_RESUME_ID WANDB_RESUME_EPISODE_COUNT || true
if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
    export RESUME_CHECKPOINT="$RESUME_CKPT"
    echo "=== RESUME weights from ${RESUME_CHECKPOINT} ==="
    echo "    source_run_dir=${SOURCE_RUN_DIR:-?}"
    ls -la "$RESUME_CHECKPOINT" || true
else
    if [ "$ALLOW_FRESH" = "1" ]; then
        echo "=== FRESH start (allowed for idx=${IDX} ${MODE} apa=${APA}) ==="
        SOURCE_RUN_DIR=""
    else
        echo "FATAL: expected 15M checkpoint for ${SOURCE_STEM} (or ${CKPT_LOCAL})"
        ls -d "${PROJECT_DIR}/outputs/${SOURCE_STEM}_"* 2>/dev/null || echo "  (no matching dirs)"
        exit 1
    fi
fi

# ---- W&B: continue the *same* run id as the 15M source (paths ≠ seeds) ----
WANDB_NAME="${NAME_STEM}"
if [ -n "${SOURCE_RUN_DIR:-}" ]; then
    export _WB_LOOKUP_PROJECT="$PROJECT_DIR"
    export _WB_LOOKUP_SOURCE="$SOURCE_RUN_DIR"
    _WB_LOOKUP=$(python - <<'PY'
import os
import sys
from pathlib import Path

project = Path(os.environ["_WB_LOOKUP_PROJECT"])
source = Path(os.environ["_WB_LOOKUP_SOURCE"]).resolve()
stem = source.name
needles = {stem, str(source), source.as_posix()}

def run_id_from_dir(run_dir: Path):
    name = run_dir.name
    if name.startswith("offline-run-") or name.startswith("run-"):
        return name.rsplit("-", 1)[-1]
    return None

hits = []
for wand_root in sorted(project.glob("wandb_offline_*")):
    if not wand_root.is_dir():
        continue
    for run_dir in list(wand_root.glob("**/offline-run-*")) + list(
        wand_root.glob("**/run-*")
    ):
        if not run_dir.is_dir():
            continue
        files_dir = run_dir / "files"
        texts = []
        for rel in ("wandb-metadata.json", "config.yaml", "output.log"):
            p = files_dir / rel
            if p.is_file() and p.stat().st_size < 8_000_000:
                try:
                    texts.append(p.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
        blob = "\n".join(texts)
        if not blob:
            continue
        if not any(n in blob for n in needles):
            continue
        rid = run_id_from_dir(run_dir)
        if not rid:
            continue
        hits.append((run_dir.stat().st_mtime, rid, str(wand_root.resolve())))

if not hits:
    print("NONE")
    sys.exit(0)
hits.sort(reverse=True)
_mtime, rid, wroot = hits[0]
print(f"{rid}\t{wroot}")
PY
)
    unset _WB_LOOKUP_PROJECT _WB_LOOKUP_SOURCE
    if [ -n "$_WB_LOOKUP" ] && [ "$_WB_LOOKUP" != "NONE" ]; then
        export WANDB_RESUME_ID="$(printf '%s\n' "$_WB_LOOKUP" | awk -F'\t' '{print $1}')"
        _SRC_WANDB="$(printf '%s\n' "$_WB_LOOKUP" | awk -F'\t' '{print $2}')"
        if [ -n "$_SRC_WANDB" ] && [ -d "$_SRC_WANDB" ]; then
            WANDB_DIR="$_SRC_WANDB"
        fi
        WANDB_NAME="$(basename "$SOURCE_RUN_DIR")"
        echo "=== W&B resume id=${WANDB_RESUME_ID} dir=${WANDB_DIR} name=${WANDB_NAME} ==="
    else
        echo "WARNING: could not find W&B run id for ${SOURCE_RUN_DIR}; will open a NEW W&B run"
        echo "  (weights still resume from ckpt; curves will not append on the old link)"
    fi
fi

TAGS="ippo,test,plus_patch,50m,n${_N},apa${APA_TAG},density_sweep,supervisor_batch,continuable,seed${TRAIN_SEED},continue_15to50"
if [ "$SHARED" = true ]; then
    TAGS="${TAGS},shared_rewards"
    SHARED_ARGS=("OPTIMIZE_SHARED_REWARDS=true")
else
    TAGS="${TAGS},independent"
    SHARED_ARGS=()
fi

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="${NAME_STEM}_${SLURM_JOB_ID}"
export RUN_TWEAKS="SEED=${TRAIN_SEED},ENV_NUM_AGENTS=${_N},ENV_KWARGS.apple_density=${DENSITY},ENV_KWARGS.apple_layout_mode=${LAYOUT_MODE},TOTAL_TIMESTEPS=${TOTAL_TS},RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-},WANDB_RESUME_ID=${WANDB_RESUME_ID:-}"
export RUN_NOTES="Continue density ${MODE} N=4 apa=${APA} seed=${TRAIN_SEED} → ${TOTAL_TS} (W&B same run if id found)"
export WANDB_TAGS="${TAGS}"

echo "=== density 15→50M ${MODE} apa=${APA} seed=${TRAIN_SEED} started at $(date) on $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID  array=${IDX}  config=$CONFIG_NAME"
echo "  density=${DENSITY}  apples=${TARGET_APPLES}  TOTAL_TIMESTEPS=${TOTAL_TS}"
echo "  RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-NONE}"
echo "  WANDB_RESUME_ID=${WANDB_RESUME_ID:-NONE}  WANDB_NAME=${WANDB_NAME}"
echo "  out=$OUT_DIR  (stable 50m path; 15M sources keep jobid paths — not a new seed)"

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
export WANDB_RUN_GROUP="ippo_n4_density_50m"
export WANDB_NAME
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
    "${PLUS98_IPPO_OVERRIDES[@]}" \
    hydra.run.dir="${OUT_DIR}" \
    "${SHARED_ARGS[@]}" \
    || TRAIN_EXIT=$?
export TRAIN_EXIT

echo "=== Training finished at $(date), exit=$TRAIN_EXIT ==="
ls -la "${OUT_DIR}/checkpoints/" 2>/dev/null | tail -8 || echo "WARNING: no checkpoints"

WANDB_RUN_ID="${WANDB_RESUME_ID:-}"
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
