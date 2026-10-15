#!/bin/bash
#SBATCH --job-name=ippo_v2_nopr_50k
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
# Test-layout reputation v2 baselines — NO peer punish — 50k episodes × 8 seeds.
#
# Matched to newest gimm scratch family (br5_gimm / w2/m1/last_apple/tdur25):
#   ENV_SCALE=test, N=4, plus_patch ~98 apples, NUM_ENVS=192
#   IPPO: LR=5e-4 anneal@100M, ep=4, mb=25, γ=0.995, ENT_COEF=0.01 (no anneal)
#   Rep labels ON: w=2/m=1 last_apple, tdur25; system enf OFF; no police/bandits
# Differs from gimm ONLY by: peer_punish_reward_enabled=false (+ reward mode arm).
#
# Array (16 tasks):
#   0–7  → independent PPO   (OPTIMIZE_SHARED_REWARDS=false), SEED = idx
#   8–15 → shared-reward PPO (OPTIMIZE_SHARED_REWARDS=true),  SEED = idx-8
#
# W&B groups (manual, seed-agnostic — same pattern as ippo_test_n_indep/shared):
#   ippo_v2_w2_m1_no_peer_indep_50kep
#   ippo_v2_w2_m1_no_peer_shared_50kep
#
#   sbatch --export=ALL,SWEEP_DATE=$(date +%Y%m%d_%H%M%S) \
#     harvest_socialjax/scripts/main/sbatch_v2_no_peer_indep_shared_50kep.sh
#SBATCH --array=0-15
#SBATCH --time=14:00:00
#SBATCH --output=logs/ippo_v2_no_peer_indep_shared_50kep_%A_%a.out
#SBATCH --error=logs/ippo_v2_no_peer_indep_shared_50kep_%A_%a.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

SWEEP_DATE="${SWEEP_DATE:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${PROJECT_DIR}/runs/v2_no_peer_indep_shared_50kep_${SWEEP_DATE}"

WINDOW_LEN=2
MAX_H=1
NUM_ENVS="${NUM_ENVS:-192}"
NUM_AGENTS="${ENV_NUM_AGENTS:-4}"
TIMEOUT_DUR="${TIMEOUT_DURATION:-25}"
TOUT_TAG="tdur${TIMEOUT_DUR}"

# Explicit IPPO pins (match gimm / ippo_test_reputation_v2.yaml — do not drift).
ENT_COEF_VAL="${ENT_COEF:-0.01}"
ANNEAL_ENT_VAL="${ANNEAL_ENT:-false}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-25}"
GAMMA="${GAMMA:-0.995}"

IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
if [ "$IDX" -lt 8 ]; then
    REWARD_MODE="indep"
    OPT_SHARED=false
    TRAIN_SEED="$IDX"
else
    REWARD_MODE="shared"
    OPT_SHARED=true
    TRAIN_SEED=$((IDX - 8))
fi

# ~50k episodes: 521 updates × 96 eps/update (192 envs × 500 steps / T=1000).
TOTAL_TS="${TOTAL_TIMESTEPS:-50016000}"
ANNEAL_HORIZON="${ANNEAL_LR_TOTAL_TIMESTEPS:-100000000}"
CONFIG_NAME="${CONFIG_NAME:-ippo_test_reputation_v2}"
SBATCH_FILE="${SBATCH_FILE:-sbatch_v2_no_peer_indep_shared_50kep.sh}"
OUT_DIR="${OUT_DIR:-${SWEEP_ROOT}/${REWARD_MODE}_w${WINDOW_LEN}_m${MAX_H}_no_peer_${TOUT_TAG}_s${TRAIN_SEED}}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/wandb_offline_${SLURM_ARRAY_JOB_ID}_${IDX}_v2_no_peer_${REWARD_MODE}_s${TRAIN_SEED}}"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"
_WANDB_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/wandb_offline.sh"

GROUP_STEM="ippo_v2_w${WINDOW_LEN}_m${MAX_H}_no_peer_${REWARD_MODE}_50kep"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="${WANDB_NAME:-${GROUP_STEM}_s${TRAIN_SEED}_${SLURM_ARRAY_JOB_ID}_${IDX}}"
export RUN_NOTES="v2 labels w${WINDOW_LEN}/m${MAX_H} last_apple; NO peer punish; ${REWARD_MODE}; no police/enf/bandits; ENT=${ENT_COEF_VAL} (no anneal); ~50kep seed=${TRAIN_SEED}"
export RUN_TWEAKS="SEED=${TRAIN_SEED},TOTAL_TIMESTEPS=${TOTAL_TS},NUM_ENVS=${NUM_ENVS},ENV_NUM_AGENTS=${NUM_AGENTS},ENT_COEF=${ENT_COEF_VAL},ANNEAL_ENT=${ANNEAL_ENT_VAL},UPDATE_EPOCHS=${UPDATE_EPOCHS},NUM_MINIBATCHES=${NUM_MINIBATCHES},GAMMA=${GAMMA},OPTIMIZE_SHARED_REWARDS=${OPT_SHARED},OPTIMIZE_BASELINE=false,ENABLE_POLICE_AGENT=false,REPUTATION_V2.peer_punish_reward_enabled=false,REPUTATION_V2.enforcement_enabled=false,REPUTATION_V2.prob_enforcement_enabled=false,REPUTATION_V2.timeout_duration=${TIMEOUT_DUR},REPUTATION_V2.window_len=${WINDOW_LEN},REPUTATION_V2.max_harvests_in_window=${MAX_H},REPUTATION_V2.last_apple_timeout=true,PROJECT=harvest_ippo_rep"

echo "=== v2 no-peer ${REWARD_MODE} 50kep seed=${TRAIN_SEED} at $(date) on $(hostname) ==="
echo "Array ${SLURM_ARRAY_JOB_ID}_${IDX}  mode=${REWARD_MODE}  OPTIMIZE_SHARED=${OPT_SHARED}"
echo "  ENV_NUM_AGENTS=${NUM_AGENTS}  plus_patch / test layout; no bandits; no warmstart"
echo "  peer_punish_reward_enabled=false; system enf OFF; no police"
echo "  IPPO: ep=${UPDATE_EPOCHS} mb=${NUM_MINIBATCHES} γ=${GAMMA} ENT_COEF=${ENT_COEF_VAL} ANNEAL_ENT=${ANNEAL_ENT_VAL}"
echo "  NUM_ENVS=${NUM_ENVS}  TOTAL_TIMESTEPS=${TOTAL_TS}  (~50k episodes)"
echo "  W&B group=${GROUP_STEM}  out=${OUT_DIR}"

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
export WANDB_PROJECT="${WANDB_PROJECT:-harvest_ippo_rep}"
export WANDB_GROUP_MODE=manual
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${GROUP_STEM}}"
export WANDB_NAME="${WANDB_NAME:-${RUN_LABEL}}"
export WANDB_TAGS="${WANDB_TAGS:-ippo,test,reputation_v2,no_peer,from_scratch,last_apple,50kep,n${NUM_AGENTS},plus_patch,w${WINDOW_LEN},m${MAX_H},${REWARD_MODE},${TOUT_TAG},e${NUM_ENVS},seed${TRAIN_SEED},ent${ENT_COEF_VAL}}"
export WANDB_SYNC_INTERVAL="${WANDB_SYNC_INTERVAL:-300}"
mkdir -p "$WANDB_DIR" "$OUT_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

bash "$_REGISTER_SH" submitted || true
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import jax; print('jax', jax.__version__, jax.devices(), jax.default_backend())" || exit 1

if [ "$WANDB_MODE" = "offline" ]; then
    # shellcheck source=/dev/null
    source "$_WANDB_SH"
    wandb_offline_job_start "$WANDB_DIR" "$WANDB_SYNC_INTERVAL"
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
    OPTIMIZE_SHARED_REWARDS="${OPT_SHARED}" \
    OPTIMIZE_BASELINE=false \
    CHECKPOINT_EVERY=50 \
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
    REPUTATION_V2.peer_focus_on_bad=false \
    REPUTATION_V2.prob_enforcement_enabled=false \
    REPUTATION_V2.prob_enforcement_decay_enabled=false \
    REPUTATION_V2.prob_enforcement_decay2_episodes=0 \
    hydra.run.dir="${OUT_DIR}" \
    || TRAIN_EXIT=$?
export TRAIN_EXIT

echo "=== Training finished at $(date), exit=$TRAIN_EXIT ==="
ls -la "${OUT_DIR}/checkpoints/" 2>/dev/null | tail -8 || echo "WARNING: no checkpoints"

if [ "$WANDB_MODE" = "offline" ]; then
    wandb_offline_job_finish "$WANDB_DIR"
fi
bash "$_REGISTER_SH" finished || true
exit $TRAIN_EXIT
