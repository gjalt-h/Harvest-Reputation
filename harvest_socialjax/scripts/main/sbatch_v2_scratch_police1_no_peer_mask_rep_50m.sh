#!/bin/bash
#SBATCH --job-name=ippo_v2_pol1_msk
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=14:00:00
#SBATCH --output=logs/ippo_v2_scratch_police1_no_peer_mask_rep_50m_%j.out
#SBATCH --error=logs/ippo_v2_scratch_police1_no_peer_mask_rep_50m_%j.err
#
# Replica of sbatch_v2_scratch_police1_no_peer_50m.sh with reputation OBS masked:
#   4 harvesters + 1 police FROM SCRATCH
#   NUM_ENVS=192, ep=4, mb=25, γ=0.995, ENT=0.01 flat, LR anneal@100M
#   v2 binary labels w2/m1 last_apple tdur25; plus_patch
#   peer punish OFF; peer_focus_on_bad=false; system/prob enf OFF (police only)
#   mask_reputation_obs=true
#     → ch15/ch16 zeroed for EVERY agent including the police CNN
#     → labels still computed (police +1 Bad / −1 Good zap reward uses them)
# TOTAL_TIMESTEPS=50M, SEED=0.
#
#   sbatch --export=ALL,SWEEP_DATE=$(date +%Y%m%d_%H%M%S) \
#     harvest_socialjax/scripts/main/sbatch_v2_scratch_police1_no_peer_mask_rep_50m.sh

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

SWEEP_DATE="${SWEEP_DATE:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${PROJECT_DIR}/runs/v2_scratch_police1_no_peer_mask_rep_${SWEEP_DATE}"

WINDOW_LEN=2
MAX_H=1
NUM_ENVS="${NUM_ENVS:-192}"
NUM_AGENTS="${ENV_NUM_AGENTS:-4}"
NUM_POLICE="${NUM_POLICE_AGENTS:-1}"
TIMEOUT_DUR="${TIMEOUT_DURATION:-25}"
TOUT_TAG="tdur${TIMEOUT_DUR}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
TRAIN_SEED="${SEED:-0}"
PEER_TAG="no_peer_mask_rep"

ENT_COEF_VAL="${ENT_COEF:-0.01}"
ANNEAL_ENT_VAL="${ANNEAL_ENT:-false}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-25}"
GAMMA="${GAMMA:-0.995}"

TOTAL_TS="${TOTAL_TIMESTEPS:-50000000}"
ANNEAL_HORIZON="${ANNEAL_LR_TOTAL_TIMESTEPS:-100000000}"
CONFIG_NAME="${CONFIG_NAME:-ippo_test_reputation_v2_police}"
SBATCH_FILE="${SBATCH_FILE:-sbatch_v2_scratch_police1_no_peer_mask_rep_50m.sh}"
OUT_DIR="${OUT_DIR:-${SWEEP_ROOT}/w${WINDOW_LEN}_m${MAX_H}_police${NUM_POLICE}_${PEER_TAG}_${TOUT_TAG}_s${TRAIN_SEED}}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/wandb_offline_${SLURM_JOB_ID}_v2_scratch_police${NUM_POLICE}_${PEER_TAG}_${TOUT_TAG}_s${TRAIN_SEED}}"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"
_WANDB_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/wandb_offline.sh"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="${WANDB_NAME:-ippo_v2_w${WINDOW_LEN}_m${MAX_H}_scratch_police${NUM_POLICE}_${PEER_TAG}_${TOUT_TAG}_50m_s${TRAIN_SEED}_${SLURM_JOB_ID}}"
export RUN_NOTES="from scratch n=${NUM_AGENTS}+police${NUM_POLICE}; no peer; police-only; mask_reputation_obs; ENT=${ENT_COEF_VAL}; 50M"
export RUN_TWEAKS="SEED=${TRAIN_SEED},TOTAL_TIMESTEPS=${TOTAL_TS},NUM_ENVS=${NUM_ENVS},ENV_NUM_AGENTS=${NUM_AGENTS},ENT_COEF=${ENT_COEF_VAL},ANNEAL_ENT=${ANNEAL_ENT_VAL},UPDATE_EPOCHS=${UPDATE_EPOCHS},NUM_MINIBATCHES=${NUM_MINIBATCHES},GAMMA=${GAMMA},ENABLE_POLICE_AGENT=true,NUM_POLICE_AGENTS=${NUM_POLICE},REPUTATION_V2.peer_punish_reward_enabled=false,REPUTATION_V2.peer_focus_on_bad=false,REPUTATION_V2.mask_reputation_obs=true,REPUTATION_V2.enforcement_enabled=false,REPUTATION_V2.prob_enforcement_enabled=false,PROJECT=harvest_ippo_rep"

echo "=== scratch police${NUM_POLICE} no-peer + mask_rep 50M at $(date) on $(hostname) ==="
echo "Job ${SLURM_JOB_ID}  tag=police${NUM_POLICE}_${PEER_TAG}"
echo "  peer_punish=OFF  peer_focus_on_bad=false  system/prob enf=OFF (police only)"
echo "  mask_reputation_obs=true → police + harvesters: ch15/ch16=0 in CNN obs"
echo "  (labels still used for police zap reward only; not in observations)"
echo "  IPPO: ep=${UPDATE_EPOCHS} mb=${NUM_MINIBATCHES} γ=${GAMMA} ENT=${ENT_COEF_VAL}"
echo "  NUM_ENVS=${NUM_ENVS} TOTAL_TIMESTEPS=${TOTAL_TS} out=${OUT_DIR}"

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
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ippo_v2_w${WINDOW_LEN}_m${MAX_H}_scratch_police${NUM_POLICE}_${PEER_TAG}_${TOUT_TAG}_50M}"
export WANDB_NAME="${WANDB_NAME:-${RUN_LABEL}}"
export WANDB_TAGS="${WANDB_TAGS:-ippo,test,reputation_v2,police,police${NUM_POLICE},no_peer,mask_reputation_obs,from_scratch,last_apple,50m,n${NUM_AGENTS},plus_patch,w${WINDOW_LEN},m${MAX_H},${TOUT_TAG},e${NUM_ENVS},seed${TRAIN_SEED}}"
export WANDB_SYNC_INTERVAL="${WANDB_SYNC_INTERVAL:-300}"
export WANDB_SYNC_FIRST_DELAY="${WANDB_SYNC_FIRST_DELAY:-45}"
mkdir -p "$WANDB_DIR" "$OUT_DIR"
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 JAX_LOG_COMPILES=0

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
    NUM_POLICE_AGENTS="${NUM_POLICE}" \
    ENT_COEF="${ENT_COEF_VAL}" \
    ANNEAL_ENT="${ANNEAL_ENT_VAL}" \
    ENT_COEF_END="${ENT_COEF_VAL}" \
    UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    GAMMA="${GAMMA}" \
    PROJECT=harvest_ippo_rep \
    ENABLE_REPUTATION_V2=true \
    ENABLE_REPUTATION=false \
    ENABLE_POLICE_AGENT=true \
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
    REPUTATION_V2.peer_focus_on_bad=false \
    REPUTATION_V2.mask_reputation_obs=true \
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
