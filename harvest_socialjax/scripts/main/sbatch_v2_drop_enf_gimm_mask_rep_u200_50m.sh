#!/bin/bash
#SBATCH --job-name=ippo_v2_gimm_msk
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=logs/ippo_v2_drop_enf_gimm_mask_rep_u200_50m_%j.out
#SBATCH --error=logs/ippo_v2_drop_enf_gimm_mask_rep_u200_50m_%j.err
#
# Ablation of canonical MASKED br5_gimm (27186547 / 4vx3ryka):
#   continue from ckpt update 200 (≈19.2k eps) → 50M
#   ONLY change vs source: peer_punish_reward_enabled=false
#   keep: peer_focus_on_bad=true, mask_reputation_obs=true, IPPO pins, tdur25, …
# Labels still update and drive peer_focus / metrics; policy cannot see ch15/ch16.
#
# New W&B run: history ≤ resume ep forked from source (source intact).
#
#   sbatch --export=ALL,SWEEP_DATE=$(date +%Y%m%d_%H%M%S) \
#     harvest_socialjax/scripts/main/sbatch_v2_drop_enf_gimm_mask_rep_u200_50m.sh

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

WINDOW_LEN=2
MAX_H=1
NUM_ENVS="${NUM_ENVS:-192}"
NUM_AGENTS="${ENV_NUM_AGENTS:-4}"
TIMEOUT_DUR="${TIMEOUT_DURATION:-25}"
TOUT_TAG="tdur${TIMEOUT_DUR}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
TRAIN_SEED="${SEED:-0}"
TARGET_RESUME_UPDATE="${TARGET_RESUME_UPDATE:-200}"
TARGET_RESUME_EP="${TARGET_RESUME_EP:-$(( (TARGET_RESUME_UPDATE + 1) * 96 ))}"

# Canonical masked gimm (same settings as scratch; only drop peer reward below).
ARM_TAG="br5_gimm_mask_rep"
SOURCE_JOB="${WARMSTART_SOURCE_JOB:-27186547}"
SOURCE_WANDB_ID="${SOURCE_WANDB_ID:-}"
SOURCE_LOG="${SOURCE_LOG:-${PROJECT_DIR}/logs/ippo_v2_scratch_peer_br5_gimm_mask_rep_50m_${SOURCE_JOB}.out}"
PEER_TAG="drop_rew_keep_imm_from_u${TARGET_RESUME_UPDATE}_${ARM_TAG}"
PEER_FOCUS_ON_BAD=true
MASK_REP=true

ENT_COEF_VAL="${ENT_COEF:-0.01}"
ANNEAL_ENT_VAL="${ANNEAL_ENT:-false}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-25}"
GAMMA="${GAMMA:-0.995}"
TOTAL_TS="${TOTAL_TIMESTEPS:-50000000}"
ANNEAL_HORIZON="${ANNEAL_LR_TOTAL_TIMESTEPS:-100000000}"
CONFIG_NAME="${CONFIG_NAME:-ippo_test_reputation_v2}"
SBATCH_FILE="${SBATCH_FILE:-sbatch_v2_drop_enf_gimm_mask_rep_u200_50m.sh}"

SWEEP_DATE="${SWEEP_DATE:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${PROJECT_DIR}/runs/v2_drop_rew_keep_imm_gimm_mask_rep_u200_${SWEEP_DATE}"
OUT_DIR="${OUT_DIR:-${SWEEP_ROOT}/w${WINDOW_LEN}_m${MAX_H}_${PEER_TAG}_${TOUT_TAG}_s${TRAIN_SEED}}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/wandb_offline_${SLURM_JOB_ID}_v2_${PEER_TAG}_s${TRAIN_SEED}}"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"
_WANDB_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/wandb_offline.sh"
_FORK_PY="${PROJECT_DIR}/harvest_socialjax/scripts/fork_wandb_history.py"

# ---- source log (masked scratch) ----
if [ ! -f "$SOURCE_LOG" ]; then
    SOURCE_LOG=$(
        ls -1t "${PROJECT_DIR}"/logs/ippo_v2_scratch_peer_br5_gimm_mask_rep_50m_*.out 2>/dev/null | head -1 || true
    )
fi
if [ -z "${SOURCE_LOG:-}" ] || [ ! -f "${SOURCE_LOG:-}" ]; then
    echo "FATAL: source log missing for masked gimm ${SOURCE_JOB}"
    echo "  expected: logs/ippo_v2_scratch_peer_br5_gimm_mask_rep_50m_${SOURCE_JOB}.out"
    ls -lt "${PROJECT_DIR}"/logs/ippo_v2_scratch_peer_br5_gimm_mask_rep_50m_*.out 2>/dev/null | head -10 || echo "  (none)"
    exit 1
fi

# ---- checkpoint ----
_CKPT_DIR=""
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
if [ -n "$RESUME_CHECKPOINT" ] && [ -f "$RESUME_CHECKPOINT" ]; then
    _CKPT_DIR="$(dirname "$RESUME_CHECKPOINT")"
fi
if [ -z "$_CKPT_DIR" ]; then
    _OUT=$(grep -Eo 'out=/[^ ]+' "$SOURCE_LOG" | head -1 | sed 's/^out=//' || true)
    if [ -n "$_OUT" ] && [ -d "${_OUT}/checkpoints" ]; then
        _CKPT_DIR="${_OUT}/checkpoints"
    fi
fi
if [ -z "$_CKPT_DIR" ]; then
    shopt -s nullglob
    for _d in \
        "${PROJECT_DIR}"/runs/v2_scratch_peer_gimm_mask_rep_*/w${WINDOW_LEN}_m${MAX_H}_br5_gimm_mask_rep_${TOUT_TAG}_s${TRAIN_SEED}/checkpoints \
        "${PROJECT_DIR}"/runs/v2_scratch_peer_gimm_mask_rep_*/w${WINDOW_LEN}_m${MAX_H}_br5_gimm_mask_rep_*/checkpoints
    do
        if [ -d "$_d" ]; then _CKPT_DIR="$_d"; break; fi
    done
    shopt -u nullglob
fi
if [ -z "$_CKPT_DIR" ] || [ ! -d "$_CKPT_DIR" ]; then
    echo "FATAL: checkpoint dir missing for masked gimm ${SOURCE_JOB}"
    echo "  set RESUME_CHECKPOINT=/path/to/ckpt_update_000200.pkl"
    exit 1
fi

if [ -z "${RESUME_CHECKPOINT:-}" ] || [ ! -f "${RESUME_CHECKPOINT:-}" ]; then
    _EXACT="${_CKPT_DIR}/ckpt_update_$(printf '%06d' "${TARGET_RESUME_UPDATE}").pkl"
    if [ -f "$_EXACT" ]; then
        RESUME_CHECKPOINT="$_EXACT"
    else
        echo "FATAL: missing ${_EXACT}"
        exit 1
    fi
fi
export RESUME_CHECKPOINT

RESUME_EP=$(
    python - "$RESUME_CHECKPOINT" <<'PY' 2>/dev/null || true
import pickle, sys
from pathlib import Path
p = Path(sys.argv[1])
with p.open("rb") as fh:
    payload = pickle.load(fh)
el = payload.get("episode_life")
if isinstance(el, dict) and "count" in el:
    print(int(float(el["count"])))
else:
    upd = int(payload.get("update", -1))
    print((upd + 1) * 96 if upd >= 0 else "")
PY
)
if [ -z "$RESUME_EP" ]; then
    echo "FATAL: could not read episode_count from ${RESUME_CHECKPOINT}"
    exit 1
fi
export WANDB_RESUME_EPISODE_COUNT="${RESUME_EP}"

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
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-harvest_ippo_rep}"
export WANDB_GROUP_MODE=manual
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
case "${TMPDIR:-}" in
    /scratch-local/*|/scratch-node/*|"")
        export TMPDIR="${HOME}/.cache/wandb_tmp"
        mkdir -p "$TMPDIR"
        ;;
esac

RUN_LABEL="${WANDB_NAME:-ippo_v2_w${WINDOW_LEN}_m${MAX_H}_${PEER_TAG}_${TOUT_TAG}_50m_s${TRAIN_SEED}_${SLURM_JOB_ID}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ippo_v2_w${WINDOW_LEN}_m${MAX_H}_gimm_masked_drop_rew_keepimm_from_u200_50M}"
export WANDB_NAME="${RUN_LABEL}"
export WANDB_TAGS="${WANDB_TAGS:-ippo,test,reputation_v2,drop_reward,keep_gimm,mask_reputation_obs,ablation,continue,forked_history,gimm,50m,n${NUM_AGENTS},plus_patch,w${WINDOW_LEN},m${MAX_H},${TOUT_TAG},src${SOURCE_JOB},e${NUM_ENVS},seed${TRAIN_SEED}}"
export WANDB_SYNC_INTERVAL="${WANDB_SYNC_INTERVAL:-300}"
export WANDB_SYNC_FIRST_DELAY="${WANDB_SYNC_FIRST_DELAY:-45}"
mkdir -p "$WANDB_DIR" "$OUT_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

WANDB_RESUME_ID="${WANDB_RESUME_ID:-}"
if [ -z "$WANDB_RESUME_ID" ] && [ -z "$SOURCE_WANDB_ID" ]; then
    echo "FATAL: set SOURCE_WANDB_ID=<cloud run id> or WANDB_RESUME_ID=<forked id>"
    exit 1
fi
if [ -z "$WANDB_RESUME_ID" ]; then
    echo "=== forking W&B history from ${SOURCE_WANDB_ID} up to ep≤${RESUME_EP} ==="
    _FORK_OUT=$(
        python "$_FORK_PY" \
            --source-id "${SOURCE_WANDB_ID}" \
            --max-ep "${RESUME_EP}" \
            --name "${RUN_LABEL}" \
            --group "${WANDB_RUN_GROUP}" \
            --tags "${WANDB_TAGS}" \
            --dir "${WANDB_DIR}" \
            --notes "Forked ≤ep${RESUME_EP} from ${SOURCE_WANDB_ID} (mask_rep scratch ${SOURCE_JOB}); continue with peer_reward OFF, peer_focus ON, mask_reputation_obs=true → 50M."
    ) || { echo "FATAL: wandb history fork failed"; echo "$_FORK_OUT"; exit 1; }
    echo "$_FORK_OUT"
    WANDB_RESUME_ID=$(echo "$_FORK_OUT" | grep -Eo 'FORKED_WANDB_ID=[a-z0-9]+' | tail -1 | sed 's/^FORKED_WANDB_ID=//')
    if [ -z "$WANDB_RESUME_ID" ]; then
        echo "FATAL: could not parse FORKED_WANDB_ID"
        exit 1
    fi
fi
export WANDB_RESUME_ID
export WANDB_RESUME_EPISODE_COUNT="${RESUME_EP}"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL
export RUN_NOTES="continue masked gimm ${SOURCE_JOB} from u${TARGET_RESUME_UPDATE}/ep≈${RESUME_EP}; drop peer reward ONLY; keep Good immunity + MASK; fork ${SOURCE_WANDB_ID}→${WANDB_RESUME_ID}; →50M"
export RUN_TWEAKS="SEED=${TRAIN_SEED},TOTAL_TIMESTEPS=${TOTAL_TS},NUM_ENVS=${NUM_ENVS},RESUME_CHECKPOINT=${RESUME_CHECKPOINT},WANDB_RESUME_ID=${WANDB_RESUME_ID},WANDB_RESUME_EPISODE_COUNT=${RESUME_EP},ENABLE_POLICE_AGENT=false,REPUTATION_V2.enforcement_enabled=false,REPUTATION_V2.peer_punish_reward_enabled=false,REPUTATION_V2.peer_focus_on_bad=${PEER_FOCUS_ON_BAD},REPUTATION_V2.mask_reputation_obs=${MASK_REP},PROJECT=harvest_ippo_rep"

echo "=== masked gimm drop-rew keep-imm u${TARGET_RESUME_UPDATE} 50M at $(date) on $(hostname) ==="
echo "Job ${SLURM_JOB_ID}  source=${SOURCE_JOB}  wandb_src=${SOURCE_WANDB_ID}"
echo "  RESUME_CHECKPOINT=${RESUME_CHECKPOINT}"
echo "  WANDB_RESUME_ID=${WANDB_RESUME_ID}  episode_count=${RESUME_EP}"
echo "  peer_reward=OFF  peer_focus_on_bad=${PEER_FOCUS_ON_BAD}  mask_reputation_obs=${MASK_REP}  enf=OFF"
echo "  IPPO: ep=${UPDATE_EPOCHS} mb=${NUM_MINIBATCHES} γ=${GAMMA} ENT=${ENT_COEF_VAL} (flat)"
echo "  out=${OUT_DIR}"

bash "$_REGISTER_SH" submitted || true
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

# shellcheck source=/dev/null
source "$_WANDB_SH"
wandb_offline_job_start "$WANDB_DIR" "$WANDB_SYNC_INTERVAL" "$WANDB_SYNC_FIRST_DELAY"

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name "$CONFIG_NAME" \
    SEED="${TRAIN_SEED}" \
    NUM_SEEDS=1 \
    TOTAL_TIMESTEPS="${TOTAL_TS}" \
    ANNEAL_LR_TOTAL_TIMESTEPS="${ANNEAL_HORIZON}" \
    NUM_ENVS="${NUM_ENVS}" \
    ENV_NUM_AGENTS="${NUM_AGENTS}" \
    ENT_COEF="${ENT_COEF_VAL}" \
    ANNEAL_ENT="${ANNEAL_ENT_VAL}" \
    ENT_COEF_END="${ENT_COEF_VAL}" \
    UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    GAMMA="${GAMMA}" \
    PROJECT=harvest_ippo_rep \
    ENABLE_REPUTATION_V2=true \
    ENABLE_REPUTATION=false \
    ENABLE_POLICE_AGENT=false \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
    RENDER_GIF=false \
    WANDB_MODE="${WANDB_MODE}" \
    REPUTATION_V2.window_len="${WINDOW_LEN}" \
    REPUTATION_V2.max_harvests_in_window="${MAX_H}" \
    REPUTATION_V2.last_apple_timeout=true \
    REPUTATION_V2.reset_label_on_respawn=true \
    REPUTATION_V2.immediate_timeout=false \
    REPUTATION_V2.timeout_delay=0 \
    REPUTATION_V2.enforcement_enabled=false \
    REPUTATION_V2.enforcement_delay=0 \
    REPUTATION_V2.enforcement_period=0 \
    REPUTATION_V2.timeout_duration="${TIMEOUT_DUR}" \
    REPUTATION_V2.graded_reputation=false \
    REPUTATION_V2.delay_ramp_enabled=false \
    REPUTATION_V2.enforcement_stop_after_episodes=0 \
    REPUTATION_V2.peer_punish_reward_enabled=false \
    REPUTATION_V2.peer_punish_anneal=false \
    REPUTATION_V2.peer_focus_on_bad="${PEER_FOCUS_ON_BAD}" \
    REPUTATION_V2.mask_reputation_obs="${MASK_REP}" \
    REPUTATION_V2.prob_enforcement_enabled=false \
    REPUTATION_V2.prob_enforcement_decay_enabled=false \
    REPUTATION_V2.prob_enforcement_decay2_episodes=0 \
    hydra.run.dir="${OUT_DIR}" \
    || TRAIN_EXIT=$?
export TRAIN_EXIT

echo "=== Training finished at $(date), exit=$TRAIN_EXIT ==="
ls -la "${OUT_DIR}/checkpoints/" 2>/dev/null | tail -8 || echo "WARNING: no checkpoints"

wandb_offline_job_finish "$WANDB_DIR" || true
bash "$_REGISTER_SH" finished || true
exit $TRAIN_EXIT
