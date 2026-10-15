#!/bin/bash
#SBATCH --job-name=ippo_v2_drop_pol_s17
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=12:00:00
#SBATCH --array=1-7
#SBATCH --output=logs/ippo_v2_drop_police_from_u200_seeds1to7_50m_%A_%a.out
#SBATCH --error=logs/ippo_v2_drop_police_from_u200_seeds1to7_50m_%A_%a.err
#
# Extra seeds 1–7 for drop-police u200 → 50M.
# Matches seed0 drop arm 0 of sbatch_v2_drop_enf_continue_50m.sh
#   (cky5gg5w ← zc1aillq / 27173445).
#
# For each array task s:
#   parent = police1 no-peer seed s (group below)
#   resume ckpt_update_000200.pkl (~ep 19200)
#   fork W&B history ≤ resume ep
#   continue with police/enf/peer OFF, TRAIN_SEED=s
#
# Parent group:
#   ippo_v2_w2_m1_scratch_police1_no_peer_tdur25_50M
# Drop group (same as seed0 drops):
#   ippo_v2_w2_m1_drop_enf_continue_50M
#
#   sbatch --export=ALL,SWEEP_DATE=$(date +%Y%m%d_%H%M%S) \
#     harvest_socialjax/scripts/main/sbatch_v2_drop_police_from_u200_seeds1to7_50m.sh
#
# Optional: SOURCE_WANDB_ID / RESUME_CHECKPOINT / SOURCE_ARRAY_JOB / TARGET_RESUME_UPDATE

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
TRAIN_SEED="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
TARGET_RESUME_UPDATE="${TARGET_RESUME_UPDATE:-200}"
TARGET_RESUME_EP="${TARGET_RESUME_EP:-$(( (TARGET_RESUME_UPDATE + 1) * 96 ))}"

ENT_COEF_VAL="${ENT_COEF:-0.01}"
ANNEAL_ENT_VAL="${ANNEAL_ENT:-false}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-25}"
GAMMA="${GAMMA:-0.995}"
TOTAL_TS="${TOTAL_TIMESTEPS:-50000000}"
ANNEAL_HORIZON="${ANNEAL_LR_TOTAL_TIMESTEPS:-100000000}"
CONFIG_NAME="${CONFIG_NAME:-ippo_test_reputation_v2}"
SBATCH_FILE="${SBATCH_FILE:-sbatch_v2_drop_police_from_u200_seeds1to7_50m.sh}"

ARM_TAG="police1"
PEER_TAG="drop_enf_from_u${TARGET_RESUME_UPDATE}_${ARM_TAG}"
PEER_FOCUS_ON_BAD=false
PARENT_GROUP="${PARENT_GROUP:-ippo_v2_w${WINDOW_LEN}_m${MAX_H}_scratch_police1_no_peer_${TOUT_TAG}_50M}"

SWEEP_DATE="${SWEEP_DATE:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${PROJECT_DIR}/runs/v2_drop_police_from_u200_seeds_${SWEEP_DATE}"
OUT_DIR="${OUT_DIR:-${SWEEP_ROOT}/w${WINDOW_LEN}_m${MAX_H}_${PEER_TAG}_${TOUT_TAG}_s${TRAIN_SEED}}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/wandb_offline_${SLURM_ARRAY_JOB_ID}_${TRAIN_SEED}_v2_drop_enf_${ARM_TAG}_s${TRAIN_SEED}}"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"
_WANDB_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/wandb_offline.sh"
_FORK_PY="${PROJECT_DIR}/harvest_socialjax/scripts/fork_wandb_history.py"

SOURCE_WANDB_ID="${SOURCE_WANDB_ID:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
SOURCE_ARRAY_JOB="${SOURCE_ARRAY_JOB:-}"
SOURCE_LOG=""
if [ -n "$SOURCE_ARRAY_JOB" ]; then
    SOURCE_LOG="${PROJECT_DIR}/logs/ippo_v2_scratch_police1_no_peer_seeds1to7_50m_${SOURCE_ARRAY_JOB}_${TRAIN_SEED}.out"
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

# ---- resolve parent W&B id (group + SEED) ----
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
    if s is None:
        continue
    if int(s) != seed:
        continue
    cands.append(r)
if not cands:
    raise SystemExit(f"no W&B run group={group} SEED={seed}")
# Prefer finished; else most recent
fin = [r for r in cands if r.state == "finished"]
pick = fin[0] if fin else cands[0]
print(pick.id)
PY
    ) || { echo "FATAL: could not resolve SOURCE_WANDB_ID for seed=${TRAIN_SEED} group=${PARENT_GROUP}"; exit 1; }
fi
if [ -z "$SOURCE_WANDB_ID" ]; then
    echo "FATAL: empty SOURCE_WANDB_ID for seed=${TRAIN_SEED}"
    exit 1
fi

# ---- resolve checkpoint dir / u200 ----
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
    _cands=(
        "${PROJECT_DIR}"/runs/v2_scratch_police1_no_peer_seeds_*/"w${WINDOW_LEN}_m${MAX_H}_police1_no_peer_${TOUT_TAG}_s${TRAIN_SEED}"/checkpoints
    )
    _best=""
    _best_mtime=0
    for _d in "${_cands[@]}"; do
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
    echo "FATAL: checkpoint dir missing for police1 seed=${TRAIN_SEED}"
    echo "  set RESUME_CHECKPOINT=/path/to/ckpt_update_$(printf '%06d' "${TARGET_RESUME_UPDATE}").pkl"
    echo "  or SOURCE_ARRAY_JOB=<parent array job id>"
    exit 1
fi

if [ -z "${RESUME_CHECKPOINT:-}" ] || [ ! -f "${RESUME_CHECKPOINT:-}" ]; then
    _EXACT="${_CKPT_DIR}/ckpt_update_$(printf '%06d' "${TARGET_RESUME_UPDATE}").pkl"
    if [ -f "$_EXACT" ]; then
        RESUME_CHECKPOINT="$_EXACT"
        echo "  using exact ckpt update=${TARGET_RESUME_UPDATE}: ${RESUME_CHECKPOINT}"
    else
        export CKPT_DIR="$_CKPT_DIR"
        export TARGET_RESUME_EP
        _PICK=$(
            python - <<'PY' 2>/dev/null || true
import os, pickle
from pathlib import Path
ckpt_dir = Path(os.environ["CKPT_DIR"])
target = int(float(os.environ["TARGET_RESUME_EP"]))

def ep_of(path):
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception:
        return None
    el = payload.get("episode_life")
    if isinstance(el, dict) and "count" in el:
        return int(float(el["count"]))
    upd = int(payload.get("update", -1))
    if upd >= 0:
        return (upd + 1) * 96
    return None

best = None
for p in sorted(ckpt_dir.glob("ckpt_*.pkl")):
    if p.name == "ckpt_latest.pkl":
        continue
    ep = ep_of(p)
    if ep is None:
        continue
    dist = abs(ep - target)
    if best is None or dist < best[0] or (dist == best[0] and ep > best[1]):
        best = (dist, ep, str(p))
if best is None:
    raise SystemExit(0)
print(f"{best[2]}|{best[1]}")
PY
        )
        if [ -z "$_PICK" ]; then
            echo "FATAL: no suitable checkpoint near ep=${TARGET_RESUME_EP} in ${_CKPT_DIR}"
            ls -la "${_CKPT_DIR}" | tail -20 || true
            exit 1
        fi
        RESUME_CHECKPOINT="${_PICK%%|*}"
        echo "  nearest ckpt to ep≈${TARGET_RESUME_EP}: ${RESUME_CHECKPOINT} (ep=${_PICK##*|})"
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

_DIST=$(( RESUME_EP > TARGET_RESUME_EP ? RESUME_EP - TARGET_RESUME_EP : TARGET_RESUME_EP - RESUME_EP ))
if [ "$_DIST" -gt 8000 ]; then
    echo "WARNING: resume_ep=${RESUME_EP} is far from ${TARGET_RESUME_EP} (Δ=${_DIST})" >&2
fi

RUN_LABEL="${WANDB_NAME:-ippo_v2_w${WINDOW_LEN}_m${MAX_H}_${PEER_TAG}_${TOUT_TAG}_50m_s${TRAIN_SEED}_${SLURM_ARRAY_JOB_ID}_${TRAIN_SEED}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ippo_v2_w${WINDOW_LEN}_m${MAX_H}_drop_enf_continue_50M}"
export WANDB_NAME="${RUN_LABEL}"
export WANDB_TAGS="${WANDB_TAGS:-ippo,test,reputation_v2,drop_enf,continue,forked_history,no_peer,no_police,no_enf,50m,n${NUM_AGENTS},plus_patch,w${WINDOW_LEN},m${MAX_H},${TOUT_TAG},${ARM_TAG},multi_seed,e${NUM_ENVS},seed${TRAIN_SEED}}"
export WANDB_SYNC_INTERVAL="${WANDB_SYNC_INTERVAL:-300}"
export WANDB_SYNC_FIRST_DELAY="${WANDB_SYNC_FIRST_DELAY:-45}"
mkdir -p "$WANDB_DIR" "$OUT_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

WANDB_RESUME_ID="${WANDB_RESUME_ID:-}"
if [ -z "$WANDB_RESUME_ID" ]; then
    echo "=== forking W&B history from ${SOURCE_WANDB_ID} (seed=${TRAIN_SEED}) up to ep≤${RESUME_EP} ==="
    _FORK_OUT=$(
        python "$_FORK_PY" \
            --source-id "${SOURCE_WANDB_ID}" \
            --max-ep "${RESUME_EP}" \
            --name "${RUN_LABEL}" \
            --group "${WANDB_RUN_GROUP}" \
            --tags "${WANDB_TAGS}" \
            --dir "${WANDB_DIR}" \
            --notes "Forked ≤ep${RESUME_EP} from ${SOURCE_WANDB_ID} (${ARM_TAG} seed=${TRAIN_SEED}); continue to 50M with enf/police/peer OFF. Source intact."
    ) || { echo "FATAL: wandb history fork failed"; echo "$_FORK_OUT"; exit 1; }
    echo "$_FORK_OUT"
    WANDB_RESUME_ID=$(echo "$_FORK_OUT" | grep -Eo 'FORKED_WANDB_ID=[a-z0-9]+' | tail -1 | sed 's/^FORKED_WANDB_ID=//')
    if [ -z "$WANDB_RESUME_ID" ]; then
        echo "FATAL: could not parse FORKED_WANDB_ID from fork output"
        exit 1
    fi
fi
export WANDB_RESUME_ID
export WANDB_RESUME_EPISODE_COUNT="${RESUME_EP}"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL
export RUN_NOTES="continue ${ARM_TAG} seed=${TRAIN_SEED} from ep≈${RESUME_EP} ckpt; drop enf+police+peer; fork W&B ${SOURCE_WANDB_ID}→${WANDB_RESUME_ID}; →50M"
export RUN_TWEAKS="SEED=${TRAIN_SEED},TOTAL_TIMESTEPS=${TOTAL_TS},NUM_ENVS=${NUM_ENVS},RESUME_CHECKPOINT=${RESUME_CHECKPOINT},WANDB_RESUME_ID=${WANDB_RESUME_ID},WANDB_RESUME_EPISODE_COUNT=${RESUME_EP},ENABLE_POLICE_AGENT=false,REPUTATION_V2.enforcement_enabled=false,REPUTATION_V2.peer_punish_reward_enabled=false,REPUTATION_V2.peer_focus_on_bad=${PEER_FOCUS_ON_BAD},PROJECT=harvest_ippo_rep"

echo "=== drop-police from u${TARGET_RESUME_UPDATE} seed=${TRAIN_SEED} → 50M at $(date) on $(hostname) ==="
echo "Array ${SLURM_ARRAY_JOB_ID}_${TRAIN_SEED}  parent_group=${PARENT_GROUP}  wandb_src=${SOURCE_WANDB_ID}"
echo "  RESUME_CHECKPOINT=${RESUME_CHECKPOINT}"
echo "  WANDB_RESUME_ID=${WANDB_RESUME_ID}  episode_count=${RESUME_EP}"
echo "  ENABLE_POLICE_AGENT=false  peer_reward=OFF  peer_focus_on_bad=${PEER_FOCUS_ON_BAD}  enf=OFF"
echo "  IPPO: ep=${UPDATE_EPOCHS} mb=${NUM_MINIBATCHES} γ=${GAMMA} ENT=${ENT_COEF_VAL}"
echo "  NUM_ENVS=${NUM_ENVS} TOTAL_TIMESTEPS=${TOTAL_TS} out=${OUT_DIR}"

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
