#!/bin/bash
#SBATCH --job-name=ippo_n4_apa
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --array=0-3
# Independent IPPO 50M — N=4 plus_patch density (seed 0):
#   idx 0 → apa=15   (60 apples)   fresh
#   idx 1 → apa=18   (72 apples)   REQUIRES 15M ckpt (reuse weights + W&B run)
#   idx 2 → apa=21   (84 apples)   REQUIRES 15M ckpt (reuse weights + W&B run)
#   idx 3 → apa=28   (112 apples)  fresh
# Canonical apa=24.5 / 98 apples already finished — not in this array.
# TOTAL_TIMESTEPS=50M; ANNEAL_LR_TOTAL_TIMESTEPS=100M; OPTIMIZE_SHARED_REWARDS=false.
#SBATCH --time=08:00:00
#SBATCH --output=logs/ippo_n4_apa_dens_%a_%A.out
#SBATCH --error=logs/ippo_n4_apa_dens_%a_%A.err

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
CONFIG_NAME="ippo_test"
SBATCH_FILE="run_ippo_n4_apa_15_18_21_28_50m.sbatch"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"

IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
case "$IDX" in
    0) APA=15; APA_TAG="15";   TARGET_APPLES=60;  REQUIRE_15M=0 ;;
    1) APA=18; APA_TAG="18";   TARGET_APPLES=72;  REQUIRE_15M=1 ;;
    2) APA=21; APA_TAG="21";   TARGET_APPLES=84;  REQUIRE_15M=1 ;;
    3) APA=28; APA_TAG="28";   TARGET_APPLES=112; REQUIRE_15M=0 ;;
    *)
        echo "FATAL: array task $IDX out of range (0-3)"
        exit 1
        ;;
esac

TRAIN_SEED="${SEED:-0}"
DENSITY=$(python -c "print(${TARGET_APPLES}/(${_BASE_APA}*${_N}))")
NAME_BASE="ippo_indep_n${_N}_apa_${APA_TAG}_50m"
if [ "${TRAIN_SEED}" = "0" ]; then
    NAME_STEM="${NAME_BASE}"
else
    NAME_STEM="${NAME_BASE}_seed${TRAIN_SEED}"
fi
OUT_DIR="${PROJECT_DIR}/outputs/${NAME_STEM}"
WANDB_DIR="${PROJECT_DIR}/wandb_offline_${NAME_STEM}"

# Prefer stable 50M ckpt; for apa 18/21 fall back to finished 15M run (reuse data).
CKPT_LOCAL="${OUT_DIR}/checkpoints/ckpt_latest.pkl"
SOURCE_STEM="ippo_indep_15m_apa_${APA_TAG}"
RESUME_CKPT=""
SOURCE_RUN_DIR=""
if [ -f "$CKPT_LOCAL" ]; then
    RESUME_CKPT="$CKPT_LOCAL"
    SOURCE_RUN_DIR="$OUT_DIR"
elif [ "$REQUIRE_15M" = "1" ]; then
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

# Weights via env only (not Hydra CLI) — RESUME_CHECKPOINT is not in the config struct.
unset RESUME_CHECKPOINT WANDB_RESUME_ID WANDB_RESUME_EPISODE_COUNT || true
if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
    export RESUME_CHECKPOINT="$RESUME_CKPT"
    echo "=== RESUME weights from ${RESUME_CHECKPOINT} ==="
    echo "    source_run_dir=${SOURCE_RUN_DIR:-?}"
else
    if [ "$REQUIRE_15M" = "1" ] && [ "${TRAIN_SEED}" = "0" ]; then
        echo "FATAL: expected 15M checkpoint for ${SOURCE_STEM} (or ${CKPT_LOCAL})"
        ls -d "${PROJECT_DIR}/outputs/${SOURCE_STEM}_"* 2>/dev/null || echo "  (no matching dirs)"
        exit 1
    fi
    if [ "$REQUIRE_15M" = "1" ] && [ "${TRAIN_SEED}" != "0" ]; then
        echo "=== FRESH start (apa=${APA}, seed=${TRAIN_SEED}; no seed-matched 15M continue) ==="
    else
        echo "=== FRESH start (apa=${APA}) ==="
    fi
    SOURCE_RUN_DIR=""
fi

# Continue the *same* W&B run id as the 15M source when available.
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
        if not blob or not any(n in blob for n in needles):
            continue
        rid = run_id_from_dir(run_dir)
        if rid:
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
        echo "WARNING: could not find W&B run id for ${SOURCE_RUN_DIR}; opening a NEW W&B run"
        echo "  (weights still resume from ckpt; curves will not append on the old link)"
    fi
fi

TAGS="ippo,test,independent,plus_patch,50m,n${_N},apa${APA_TAG},density_sweep,n4_apa_density,supervisor_batch,continuable,seed${TRAIN_SEED}"
if [ "$REQUIRE_15M" = "1" ]; then
    TAGS="${TAGS},continue_15to50"
fi

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="${NAME_STEM}_${SLURM_JOB_ID}"
export RUN_TWEAKS="SEED=${TRAIN_SEED},ENV_NUM_AGENTS=${_N},ENV_KWARGS.apple_density=${DENSITY},ENV_KWARGS.apple_layout_mode=${LAYOUT_MODE},apples_per_agent=${APA},target_apples=${TARGET_APPLES},TOTAL_TIMESTEPS=${TOTAL_TS},RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-},WANDB_RESUME_ID=${WANDB_RESUME_ID:-}"
export RUN_NOTES="Indep N=4 plus_patch apa=${APA} apples=${TARGET_APPLES} seed=${TRAIN_SEED} 50M (resume 15M=${REQUIRE_15M})"
export WANDB_TAGS="${TAGS}"

echo "=== N=4 apa density started at $(date) on $(hostname) ==="
echo "Job ID: $SLURM_JOB_ID  array=${IDX}  apa=${APA}  apples=${TARGET_APPLES}  dens=${DENSITY}"
echo "  TOTAL_TIMESTEPS=${TOTAL_TS}  RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-NONE}"
echo "  WANDB_RESUME_ID=${WANDB_RESUME_ID:-NONE}  WANDB_NAME=${WANDB_NAME}"
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
export WANDB_RUN_GROUP="ippo_n4_apa_density_50m"
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
    +apples_per_agent="${APA}" \
    +target_apples="${TARGET_APPLES}" \
    "${PLUS98_IPPO_OVERRIDES[@]}" \
    hydra.run.dir="${OUT_DIR}" \
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
