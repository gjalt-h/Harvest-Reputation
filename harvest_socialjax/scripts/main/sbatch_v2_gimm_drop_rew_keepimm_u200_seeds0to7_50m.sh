#!/bin/bash
#SBATCH --job-name=ippo_v2_gimm_drp_ki
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/ippo_v2_gimm_drop_rew_keepimm_u200_seeds0to7_50m_%A_%a.out
#SBATCH --error=logs/ippo_v2_gimm_drop_rew_keepimm_u200_seeds0to7_50m_%A_%a.err
#
# Drop peer reward ONLY; keep Good immunity (peer_focus_on_bad=true).
# 16 tasks = 2 arms × 8 seeds (0–7):
#   0–7  → unmasked gimm γ=0.99 ENT anneal parents
#          parent group:
#            ippo_v2_w2_m1_scratch_br5_gimm_tdur25_ep4_mb25_g99_ent005_floor0005_50kep
#          drop group (same as seed0 kf6s8p31):
#            ippo_v2_w2_m1_br5_gimm_g99_ent005_drop_rew_keepimm_from_u200_50M
#   8–15 → masked gimm (γ=0.995 flat ENT) parents
#          parent group:
#            ippo_v2_w2_m1_scratch_peer_br5_gimm_mask_rep_tdur25_50M
#          drop group (same as seed0 gbx1egfw):
#            ippo_v2_w2_m1_gimm_masked_drop_rew_keepimm_from_u200_50M
#
# Resume ckpt_update_000200.pkl (~ep 19200), fork W&B history ≤ resume ep.
#
# Seed0 already finished for both arms — skip them if you only need 1–7:
#   sbatch --array=1-7,9-15 --export=ALL,SWEEP_DATE=$(date +%Y%m%d_%H%M%S) \
#     harvest_socialjax/scripts/main/sbatch_v2_gimm_drop_rew_keepimm_u200_seeds0to7_50m.sh
#
# Full 0–7 both arms:
#   sbatch --export=ALL,SWEEP_DATE=$(date +%Y%m%d_%H%M%S) \
#     harvest_socialjax/scripts/main/sbatch_v2_gimm_drop_rew_keepimm_u200_seeds0to7_50m.sh

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
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"

TARGET_RESUME_UPDATE="${TARGET_RESUME_UPDATE:-200}"
TARGET_RESUME_EP="${TARGET_RESUME_EP:-$(( (TARGET_RESUME_UPDATE + 1) * 96 ))}"

IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
ARM=$((IDX / 8))
TRAIN_SEED=$((IDX % 8))

PEER_FOCUS_ON_BAD=true
case "$ARM" in
    0)
        ARM_NAME="gimm_g99"
        MASK_REP=false
        PARENT_GROUP="${PARENT_GROUP_GIMM:-ippo_v2_w${WINDOW_LEN}_m${MAX_H}_scratch_br5_gimm_${TOUT_TAG}_ep4_mb25_g99_ent005_floor0005_50kep}"
        DROP_GROUP="${DROP_GROUP_GIMM:-ippo_v2_w${WINDOW_LEN}_m${MAX_H}_br5_gimm_g99_ent005_drop_rew_keepimm_from_u200_50M}"
        PEER_TAG="br5_gimm_g99_ent005_drop_rew_keepimm_from_u${TARGET_RESUME_UPDATE}"
        UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
        NUM_MINIBATCHES="${NUM_MINIBATCHES:-25}"
        GAMMA="${GAMMA:-0.99}"
        PPO_TAG="ep${UPDATE_EPOCHS}_mb${NUM_MINIBATCHES}_g99"
        ENT_START="${ENT_COEF:-0.05}"
        ENT_END="${ENT_COEF_END:-0.005}"
        ENT_HZ="${ANNEAL_ENT_TOTAL_TIMESTEPS:-12504000}"
        ENT_FRAC="${ANNEAL_ENT_FRAC:-0.25}"
        ANNEAL_ENT_VAL=true
        CKPT_GLOBS=(
            "${PROJECT_DIR}"/runs/v2_scratch_peer_br5_gimm_g99_ent005_floor0005_seeds_*/"w${WINDOW_LEN}_m${MAX_H}_br5_gimm_${TOUT_TAG}_${PPO_TAG}_floor0005_s${TRAIN_SEED}"/checkpoints
            "${PROJECT_DIR}"/runs/v2_scratch_peer_br5_gimm_g99_ent005_floor0005_*/"w${WINDOW_LEN}_m${MAX_H}_br5_gimm_${TOUT_TAG}_${PPO_TAG}_floor0005_s${TRAIN_SEED}"/checkpoints
        )
        LOG_GLOB="ippo_v2_scratch_peer_br5_gimm_g99_ent005_floor0005_seeds1to7_50kep_*_${TRAIN_SEED}.out"
        if [ "$TRAIN_SEED" -eq 0 ]; then
            LOG_GLOB="ippo_v2_scratch_peer_br5_gimm_g99_ent005_floor0005_50kep_*.out"
        fi
        ;;
    1)
        ARM_NAME="gimm_masked"
        MASK_REP=true
        PARENT_GROUP="${PARENT_GROUP_MASK:-ippo_v2_w${WINDOW_LEN}_m${MAX_H}_scratch_peer_br5_gimm_mask_rep_${TOUT_TAG}_50M}"
        DROP_GROUP="${DROP_GROUP_MASK:-ippo_v2_w${WINDOW_LEN}_m${MAX_H}_gimm_masked_drop_rew_keepimm_from_u200_50M}"
        PEER_TAG="drop_rew_keep_imm_from_u${TARGET_RESUME_UPDATE}_br5_gimm_mask_rep"
        UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
        NUM_MINIBATCHES="${NUM_MINIBATCHES:-25}"
        GAMMA="${GAMMA_MASK:-0.995}"
        PPO_TAG="ep${UPDATE_EPOCHS}_mb${NUM_MINIBATCHES}"
        ENT_START="${ENT_COEF_MASK:-0.01}"
        ENT_END="${ENT_START}"
        ENT_HZ=""
        ENT_FRAC="0.25"
        ANNEAL_ENT_VAL=false
        CKPT_GLOBS=(
            "${PROJECT_DIR}"/runs/v2_scratch_peer_gimm_mask_rep_seeds_*/"w${WINDOW_LEN}_m${MAX_H}_br5_gimm_mask_rep_${TOUT_TAG}_s${TRAIN_SEED}"/checkpoints
            "${PROJECT_DIR}"/runs/v2_scratch_peer_gimm_mask_rep_*/"w${WINDOW_LEN}_m${MAX_H}_br5_gimm_mask_rep_${TOUT_TAG}_s${TRAIN_SEED}"/checkpoints
        )
        LOG_GLOB="ippo_v2_scratch_peer_br5_gimm_mask_rep_seeds1to7_50m_*_${TRAIN_SEED}.out"
        if [ "$TRAIN_SEED" -eq 0 ]; then
            LOG_GLOB="ippo_v2_scratch_peer_br5_gimm_mask_rep_50m_*.out"
        fi
        ;;
    *)
        echo "FATAL: array task $IDX out of range (0-15)"
        exit 1
        ;;
esac

TOTAL_TS="${TOTAL_TIMESTEPS:-50000000}"
ANNEAL_HORIZON="${ANNEAL_LR_TOTAL_TIMESTEPS:-100000000}"
CONFIG_NAME="${CONFIG_NAME:-ippo_test_reputation_v2}"
SBATCH_FILE="${SBATCH_FILE:-sbatch_v2_gimm_drop_rew_keepimm_u200_seeds0to7_50m.sh}"

SWEEP_DATE="${SWEEP_DATE:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${PROJECT_DIR}/runs/v2_gimm_drop_rew_keepimm_u200_seeds_${SWEEP_DATE}"
OUT_DIR="${OUT_DIR:-${SWEEP_ROOT}/${ARM_NAME}/w${WINDOW_LEN}_m${MAX_H}_${PEER_TAG}_${TOUT_TAG}_${PPO_TAG}_s${TRAIN_SEED}}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/wandb_offline_${SLURM_ARRAY_JOB_ID}_${IDX}_v2_${ARM_NAME}_drop_keepimm_s${TRAIN_SEED}}"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"
_WANDB_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/wandb_offline.sh"
_FORK_PY="${PROJECT_DIR}/harvest_socialjax/scripts/fork_wandb_history.py"

SOURCE_WANDB_ID="${SOURCE_WANDB_ID:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
SOURCE_LOG=""
shopt -s nullglob
_logs=( "${PROJECT_DIR}"/logs/${LOG_GLOB} )
shopt -u nullglob
if [ "${#_logs[@]}" -gt 0 ]; then
    SOURCE_LOG=$(ls -1t "${_logs[@]}" 2>/dev/null | head -1 || true)
fi

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

if [ -z "$SOURCE_WANDB_ID" ] && [ -n "${SOURCE_LOG:-}" ] && [ -f "$SOURCE_LOG" ]; then
    SOURCE_WANDB_ID=$(
        grep -Eo 'WANDB_RUN_ID=[a-z0-9]+' "$SOURCE_LOG" | tail -1 | sed 's/^WANDB_RUN_ID=//' || true
    )
fi
if [ -z "$SOURCE_WANDB_ID" ]; then
    export _PARENT_GROUP="$PARENT_GROUP" _TRAIN_SEED="$TRAIN_SEED"
    SOURCE_WANDB_ID=$(
        python - <<'PY'
import os
import wandb
group = os.environ["_PARENT_GROUP"]
seed = int(os.environ["_TRAIN_SEED"])
api = wandb.Api(timeout=120)
proj = f"{os.environ.get('WANDB_ENTITY','YOUR_ENTITY')}/{os.environ.get('WANDB_PROJECT','harvest_ippo_rep')}"
cands = []
for r in api.runs(proj, filters={"group": group}, order="-created_at"):
    s = (r.config or {}).get("SEED")
    if s is None or int(s) != seed:
        continue
    cands.append(r)
if not cands:
    raise SystemExit(f"no W&B run group={group} SEED={seed}")
fin = [r for r in cands if r.state == "finished"]
pick = fin[0] if fin else cands[0]
print(pick.id)
PY
    ) || { echo "FATAL: could not resolve SOURCE_WANDB_ID for seed=${TRAIN_SEED} group=${PARENT_GROUP}"; exit 1; }
fi
if [ -z "$SOURCE_WANDB_ID" ]; then
    echo "FATAL: empty SOURCE_WANDB_ID for seed=${TRAIN_SEED} arm=${ARM_NAME}"
    exit 1
fi

_CKPT_DIR=""
if [ -n "$RESUME_CHECKPOINT" ] && [ -f "$RESUME_CHECKPOINT" ]; then
    _CKPT_DIR="$(dirname "$RESUME_CHECKPOINT")"
fi
if [ -z "$_CKPT_DIR" ] && [ -n "${SOURCE_LOG:-}" ] && [ -f "$SOURCE_LOG" ]; then
    _OUT=$(grep -Eo 'out=/[^ ]+' "$SOURCE_LOG" | head -1 | sed 's/^out=//' || true)
    if [ -n "$_OUT" ] && [ -d "${_OUT}/checkpoints" ]; then
        _CKPT_DIR="${_OUT}/checkpoints"
    fi
fi
if [ -z "$_CKPT_DIR" ]; then
    shopt -s nullglob
    _best=""
    _best_mtime=0
    for _d in "${CKPT_GLOBS[@]}"; do
        [ -d "$_d" ] || continue
        _exact="${_d}/ckpt_update_$(printf '%06d' "${TARGET_RESUME_UPDATE}").pkl"
        if [ -f "$_exact" ]; then
            _m=$(stat -c %Y "$_exact" 2>/dev/null || echo 0)
            if [ -z "$_best" ] || [ "$_m" -ge "$_best_mtime" ]; then
                _best="$_d"
                _best_mtime="$_m"
            fi
        elif [ -z "$_best" ]; then
            _best="$_d"
        fi
    done
    _CKPT_DIR="$_best"
    shopt -u nullglob
fi
if [ -z "$_CKPT_DIR" ] || [ ! -d "$_CKPT_DIR" ]; then
    echo "FATAL: checkpoint dir missing for ${ARM_NAME} seed=${TRAIN_SEED}"
    echo "  set RESUME_CHECKPOINT=/path/to/ckpt_update_$(printf '%06d' "${TARGET_RESUME_UPDATE}").pkl"
    echo "  parent_group=${PARENT_GROUP}"
    exit 1
fi

if [ -z "${RESUME_CHECKPOINT:-}" ] || [ ! -f "${RESUME_CHECKPOINT:-}" ]; then
    _EXACT="${_CKPT_DIR}/ckpt_update_$(printf '%06d' "${TARGET_RESUME_UPDATE}").pkl"
    if [ -f "$_EXACT" ]; then
        RESUME_CHECKPOINT="$_EXACT"
        echo "  using exact ckpt update=${TARGET_RESUME_UPDATE}: ${RESUME_CHECKPOINT}"
    else
        echo "FATAL: missing ${_EXACT}"
        ls -la "${_CKPT_DIR}" | tail -20 || true
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
RESUME_EP=$(printf '%s\n' "${RESUME_EP}" | grep -E '^[0-9]+$' | tail -1 || true)
if [ -z "$RESUME_EP" ]; then
    echo "FATAL: could not read episode_count from ${RESUME_CHECKPOINT}"
    exit 1
fi
export WANDB_RESUME_EPISODE_COUNT="${RESUME_EP}"

RUN_LABEL="${WANDB_NAME:-ippo_v2_w${WINDOW_LEN}_m${MAX_H}_${PEER_TAG}_${TOUT_TAG}_${PPO_TAG}_50m_s${TRAIN_SEED}_${SLURM_ARRAY_JOB_ID}_${IDX}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${DROP_GROUP}}"
export WANDB_NAME="${RUN_LABEL}"
export WANDB_TAGS="${WANDB_TAGS:-ippo,test,reputation_v2,drop_reward,drop_rew_keepimm,keep_gimm,gimm,continue,forked_history,50m,n${NUM_AGENTS},plus_patch,w${WINDOW_LEN},m${MAX_H},${TOUT_TAG},${PPO_TAG},${ARM_NAME},multi_seed,e${NUM_ENVS},seed${TRAIN_SEED}}"
if [ "$MASK_REP" = true ]; then
    export WANDB_TAGS="${WANDB_TAGS},mask_reputation_obs"
else
    export WANDB_TAGS="${WANDB_TAGS},g99,ent005,unmasked"
fi
export WANDB_SYNC_INTERVAL="${WANDB_SYNC_INTERVAL:-300}"
export WANDB_SYNC_FIRST_DELAY="${WANDB_SYNC_FIRST_DELAY:-45}"
mkdir -p "$WANDB_DIR" "$OUT_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

WANDB_RESUME_ID="${WANDB_RESUME_ID:-}"
if [ -z "$WANDB_RESUME_ID" ]; then
    echo "=== forking W&B history from ${SOURCE_WANDB_ID} (arm=${ARM_NAME} seed=${TRAIN_SEED}) up to ep≤${RESUME_EP} ==="
    _FORK_OUT=$(
        python "$_FORK_PY" \
            --source-id "${SOURCE_WANDB_ID}" \
            --max-ep "${RESUME_EP}" \
            --name "${RUN_LABEL}" \
            --group "${WANDB_RUN_GROUP}" \
            --tags "${WANDB_TAGS}" \
            --dir "${WANDB_DIR}" \
            --notes "Forked ≤ep${RESUME_EP} from ${SOURCE_WANDB_ID} (${ARM_NAME} seed=${TRAIN_SEED}); drop peer_reward ONLY, keep immunity (focus=true) mask=${MASK_REP} → 50M."
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
export RUN_NOTES="continue ${ARM_NAME} seed=${TRAIN_SEED} from u${TARGET_RESUME_UPDATE}/ep≈${RESUME_EP}; DROP peer reward ONLY; keep Good immunity; mask=${MASK_REP}; fork ${SOURCE_WANDB_ID}→${WANDB_RESUME_ID}; →50M"
export RUN_TWEAKS="SEED=${TRAIN_SEED},TOTAL_TIMESTEPS=${TOTAL_TS},NUM_ENVS=${NUM_ENVS},RESUME_CHECKPOINT=${RESUME_CHECKPOINT},WANDB_RESUME_ID=${WANDB_RESUME_ID},WANDB_RESUME_EPISODE_COUNT=${RESUME_EP},UPDATE_EPOCHS=${UPDATE_EPOCHS},NUM_MINIBATCHES=${NUM_MINIBATCHES},GAMMA=${GAMMA},ENT_COEF=${ENT_START},ANNEAL_ENT=${ANNEAL_ENT_VAL},ENT_COEF_END=${ENT_END},ENABLE_POLICE_AGENT=false,REPUTATION_V2.peer_punish_reward_enabled=false,REPUTATION_V2.peer_focus_on_bad=${PEER_FOCUS_ON_BAD},REPUTATION_V2.mask_reputation_obs=${MASK_REP},REPUTATION_V2.enforcement_enabled=false,PROJECT=harvest_ippo_rep"

echo "=== ${ARM_NAME} drop-rew keep-imm u${TARGET_RESUME_UPDATE} seed=${TRAIN_SEED} → 50M at $(date) on $(hostname) ==="
echo "Array ${SLURM_ARRAY_JOB_ID}_${IDX}  arm=${ARM}  parent_group=${PARENT_GROUP}"
echo "  SOURCE_WANDB_ID=${SOURCE_WANDB_ID}"
echo "  RESUME_CHECKPOINT=${RESUME_CHECKPOINT}"
echo "  WANDB_RESUME_ID=${WANDB_RESUME_ID}  episode_count=${RESUME_EP}"
echo "  peer_reward=OFF  peer_focus_on_bad=${PEER_FOCUS_ON_BAD}  mask=${MASK_REP}  enf=OFF"
echo "  IPPO: ep=${UPDATE_EPOCHS} mb=${NUM_MINIBATCHES} γ=${GAMMA} ENT ${ENT_START}→${ENT_END} anneal=${ANNEAL_ENT_VAL}"
echo "  drop_group=${WANDB_RUN_GROUP}"
echo "  NUM_ENVS=${NUM_ENVS} TOTAL_TIMESTEPS=${TOTAL_TS} out=${OUT_DIR}"

bash "$_REGISTER_SH" submitted || true
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

# shellcheck source=/dev/null
source "$_WANDB_SH"
wandb_offline_job_start "$WANDB_DIR" "$WANDB_SYNC_INTERVAL" "$WANDB_SYNC_FIRST_DELAY"

_ENT_ARGS=(
    ENT_COEF="${ENT_START}"
    ANNEAL_ENT="${ANNEAL_ENT_VAL}"
    ENT_COEF_END="${ENT_END}"
)
if [ "$ANNEAL_ENT_VAL" = true ]; then
    _ENT_ARGS+=(
        ANNEAL_ENT_FRAC="${ENT_FRAC}"
        ANNEAL_ENT_TOTAL_TIMESTEPS="${ENT_HZ}"
    )
fi

TRAIN_EXIT=0
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py \
    --config-name "$CONFIG_NAME" \
    SEED="${TRAIN_SEED}" \
    NUM_SEEDS=1 \
    TOTAL_TIMESTEPS="${TOTAL_TS}" \
    ANNEAL_LR_TOTAL_TIMESTEPS="${ANNEAL_HORIZON}" \
    NUM_ENVS="${NUM_ENVS}" \
    ENV_NUM_AGENTS="${NUM_AGENTS}" \
    UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    GAMMA="${GAMMA}" \
    "${_ENT_ARGS[@]}" \
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
