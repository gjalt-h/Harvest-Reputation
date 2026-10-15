# Verified IPPO pins for plus98 / harvest_ippo sweep replicates.
# Source of truth: finished W&B seed-0 runs in YOUR_ENTITY/harvest_ippo
# (e.g. baseline 7mw2i57z, beam fc4s5wr1, fixed98 1w7i8cd1, …).
#
# CRITICAL: repo ippo_test.yaml currently has NUM_ENVS=192, but every verified
# seed-0 sweep run used NUM_ENVS=128. New seeds MUST pass these overrides so
# Hydra does not pick up a drifted yaml default.
#
# Usage (from an sbatch in harvest_socialjax/scripts/):
#   source "${SCRIPT_DIR}/sbatch/plus98_verified_pins.sh"
#   python .../train_ippo_harvest.py --config-name ippo_test \
#       SEED=... "${PLUS98_IPPO_OVERRIDES[@]}" ...

PLUS98_IPPO_OVERRIDES=(
    NUM_SEEDS=1
    NUM_ENVS=128
    NUM_STEPS=500
    LR=5.0e-4
    UPDATE_EPOCHS=2
    NUM_MINIBATCHES=50
    GAMMA=0.99
    GAE_LAMBDA=0.95
    CLIP_EPS=0.2
    ENT_COEF=0.01
    VF_COEF=0.5
    MAX_GRAD_NORM=0.5
    ACTIVATION=relu
    ANNEAL_LR=true
    ANNEAL_LR_TOTAL_TIMESTEPS=100000000
    TOTAL_TIMESTEPS=50000000
    RENDER_GIF=false
    ENABLE_REPUTATION=false
)

# Default env knobs matching test-scale seed-0 runs (override per-arm as needed).
PLUS98_ENV_DEFAULTS=(
    ENV_KWARGS.regrowth_scale=1.0
    ENV_KWARGS.apple_layout_mode=plus_patch
)
