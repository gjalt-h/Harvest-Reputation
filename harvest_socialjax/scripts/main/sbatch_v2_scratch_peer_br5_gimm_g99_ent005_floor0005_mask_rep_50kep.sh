#!/bin/bash
#SBATCH --job-name=ippo_v2_gimm_msk_g99
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=03:00:00
#SBATCH --output=logs/ippo_v2_scratch_peer_br5_gimm_g99_ent005_floor0005_mask_rep_50kep_%j.out
#SBATCH --error=logs/ippo_v2_scratch_peer_br5_gimm_g99_ent005_floor0005_mask_rep_50kep_%j.err
#
# Classic Good-immunity + mask_reputation_obs (ch15/ch16=0).
# Dynamics identical to unmasked twin / e0bm9ei0:
#   peer Bad-zap +5, Good_pen=0 (NO Good punishment)
#   peer_focus_on_bad=true
# Learning matched to peer ±5 (oab6rlo0):
#   ENT 0.05→0.005 q1 (hz=12504000), γ=0.99, ep4/mb25, TOTAL=50.016M
#
# Only diff vs unmasked gimm_g99 twin: mask_reputation_obs=true.
#
#   sbatch --export=ALL,SWEEP_DATE=$(date +%Y%m%d_%H%M%S) \
#     harvest_socialjax/scripts/main/sbatch_v2_scratch_peer_br5_gimm_g99_ent005_floor0005_mask_rep_50kep.sh

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }
mkdir -p logs

SWEEP_DATE="${SWEEP_DATE:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${PROJECT_DIR}/runs/v2_scratch_peer_br5_gimm_g99_ent005_floor0005_mask_rep_${SWEEP_DATE}"

WINDOW_LEN=2
MAX_H=1
NUM_ENVS="${NUM_ENVS:-192}"
NUM_AGENTS="${ENV_NUM_AGENTS:-4}"
BAD_REWARD="${PEER_PUNISH_BAD_REWARD:-5.0}"
GOOD_PENALTY="${PEER_PUNISH_GOOD_PENALTY:-0.0}"
PEER_TAG="br5_gimm_mask_rep"
TIMEOUT_DUR="${TIMEOUT_DURATION:-25}"
TOUT_TAG="tdur${TIMEOUT_DUR}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-25}"
GAMMA="${GAMMA:-0.99}"
PPO_TAG="ep${UPDATE_EPOCHS}_mb${NUM_MINIBATCHES}_g99"
ENT_START="${ENT_COEF:-0.05}"
ENT_END="${ENT_COEF_END:-0.005}"
ENT_FRAC="${ANNEAL_ENT_FRAC:-0.25}"
ENT_TAG="floor0005"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"

TRAIN_SEED="${SEED:-0}"

TOTAL_TS="${TOTAL_TIMESTEPS:-50016000}"
ENT_HZ="${ANNEAL_ENT_TOTAL_TIMESTEPS:-12504000}"
ANNEAL_HORIZON="${ANNEAL_LR_TOTAL_TIMESTEPS:-100000000}"
CONFIG_NAME="${CONFIG_NAME:-ippo_test_reputation_v2}"
SBATCH_FILE="${SBATCH_FILE:-sbatch_v2_scratch_peer_br5_gimm_g99_ent005_floor0005_mask_rep_50kep.sh}"
OUT_DIR="${OUT_DIR:-${SWEEP_ROOT}/w${WINDOW_LEN}_m${MAX_H}_${PEER_TAG}_${TOUT_TAG}_${PPO_TAG}_${ENT_TAG}_s${TRAIN_SEED}}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/wandb_offline_${SLURM_JOB_ID}_v2_scratch_${PEER_TAG}_${TOUT_TAG}_${PPO_TAG}_${ENT_TAG}_s${TRAIN_SEED}}"
_REGISTER_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/register_slurm_run.sh"
_WANDB_SH="${PROJECT_DIR}/harvest_socialjax/scripts/sbatch/wandb_offline.sh"

GROUP_STEM="ippo_v2_w${WINDOW_LEN}_m${MAX_H}_scratch_${PEER_TAG}_${TOUT_TAG}_${PPO_TAG}_ent005_${ENT_TAG}_50kep"

export CONFIG_NAME SBATCH_FILE OUT_DIR WANDB_DIR
export RUN_LABEL="${WANDB_NAME:-${GROUP_STEM}_s${TRAIN_SEED}_${SLURM_JOB_ID}}"
export RUN_NOTES="classic gimm: Bad +${BAD_REWARD} Good_pen=${GOOD_PENALTY} focus=true; learning twin of oab6rlo0 (ENT ${ENT_START}→${ENT_END} γ=${GAMMA}); mask_reputation_obs; ~50kep seed=${TRAIN_SEED}"
export RUN_TWEAKS="SEED=${TRAIN_SEED},TOTAL_TIMESTEPS=${TOTAL_TS},NUM_ENVS=${NUM_ENVS},ENV_NUM_AGENTS=${NUM_AGENTS},UPDATE_EPOCHS=${UPDATE_EPOCHS},NUM_MINIBATCHES=${NUM_MINIBATCHES},GAMMA=${GAMMA},ENT_COEF=${ENT_START},ANNEAL_ENT=true,ENT_COEF_END=${ENT_END},ANNEAL_ENT_FRAC=${ENT_FRAC},ANNEAL_ENT_TOTAL_TIMESTEPS=${ENT_HZ},ENABLE_POLICE_AGENT=false,REPUTATION_V2.peer_punish_reward_enabled=true,REPUTATION_V2.peer_punish_bad_reward=${BAD_REWARD},REPUTATION_V2.peer_punish_good_penalty=${GOOD_PENALTY},REPUTATION_V2.peer_punish_anneal=false,REPUTATION_V2.peer_focus_on_bad=true,REPUTATION_V2.mask_reputation_obs=true,REPUTATION_V2.enforcement_enabled=false,REPUTATION_V2.prob_enforcement_enabled=false,REPUTATION_V2.timeout_duration=${TIMEOUT_DUR},REPUTATION_V2.window_len=${WINDOW_LEN},REPUTATION_V2.max_harvests_in_window=${MAX_H},REPUTATION_V2.last_apple_timeout=true,PROJECT=harvest_ippo_rep"

echo "=== classic br5_gimm (+${BAD_REWARD}/0 + focus) + mask_rep + peer learning ENT ${ENT_START}→${ENT_END} γ=${GAMMA} 50kep seed=${TRAIN_SEED} at $(date) on $(hostname) ==="
echo "Job ${SLURM_JOB_ID}  group=${GROUP_STEM}"
echo "  peer_punish_bad_reward=${BAD_REWARD}  good_penalty=${GOOD_PENALTY}  (NO Good punishment)"
echo "  peer_focus_on_bad=true  mask_reputation_obs=true"
echo "  ANNEAL_ENT=true  ENT_COEF=${ENT_START}  ENT_COEF_END=${ENT_END}  anneal_hz=${ENT_HZ}"
echo "  UPDATE_EPOCHS=${UPDATE_EPOCHS}  NUM_MINIBATCHES=${NUM_MINIBATCHES}  GAMMA=${GAMMA}"
echo "  det/prob system enf OFF; no police; NUM_ENVS=${NUM_ENVS} TOTAL_TIMESTEPS=${TOTAL_TS}"
echo "  out=${OUT_DIR}"

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
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${GROUP_STEM}}"
export WANDB_NAME="${WANDB_NAME:-${RUN_LABEL}}"
export WANDB_TAGS="${WANDB_TAGS:-ippo,test,reputation_v2,peer_punish,peer_focus_on_bad,gimm,classic_gimm,mask_reputation_obs,from_scratch,ent_anneal,last_apple,50kep,n${NUM_AGENTS},plus_patch,w${WINDOW_LEN},m${MAX_H},${PEER_TAG},${TOUT_TAG},${PPO_TAG},${ENT_TAG},e${NUM_ENVS},seed${TRAIN_SEED}}"
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
    UPDATE_EPOCHS="${UPDATE_EPOCHS}" \
    NUM_MINIBATCHES="${NUM_MINIBATCHES}" \
    GAMMA="${GAMMA}" \
    ENT_COEF="${ENT_START}" \
    ANNEAL_ENT=true \
    ENT_COEF_END="${ENT_END}" \
    ANNEAL_ENT_FRAC="${ENT_FRAC}" \
    ANNEAL_ENT_TOTAL_TIMESTEPS="${ENT_HZ}" \
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
    REPUTATION_V2.peer_punish_reward_enabled=true \
    REPUTATION_V2.peer_punish_bad_reward="${BAD_REWARD}" \
    REPUTATION_V2.peer_punish_good_penalty="${GOOD_PENALTY}" \
    REPUTATION_V2.peer_punish_anneal=false \
    REPUTATION_V2.peer_focus_on_bad=true \
    REPUTATION_V2.mask_reputation_obs=true \
    REPUTATION_V2.zap_affects_label=true \
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
