#!/bin/bash
# Submit seeds 1–4 for every harvest_ippo sweep that has plots under
# plots/sweep_comparisons (manifest 2026-09-08). Seed 0 already finished.
#
# Config is pinned to W&B-verified seed-0 runs via plus98_verified_pins.sh
# (NUM_ENVS=128 — NOT the current ippo_test.yaml 192 default).
#
# Usage on Snellius (repo root = ${PROJECT_DIR}):
#   bash harvest_socialjax/scripts/main/submit_plus98_seeds_1to4.sh
#   DRY_RUN=1 bash harvest_socialjax/scripts/main/submit_plus98_seeds_1to4.sh
#   SEEDS="2 3 4" bash ...   # only missing baseline seeds, etc.
#
# After sync, regenerate plots with the extra sweep helpers / your W&B entity.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_DIR="$(cd "${SCRIPT_DIR}/../extra" && pwd)"
# scripts/main → harvest_socialjax → repo root
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}}"
SEEDS="${SEEDS:-1 2 3 4}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BASELINE_SEED1="${SKIP_BASELINE_SEED1:-1}"
CONTINUE_APA18_21_SEED0="${CONTINUE_APA18_21_SEED0:-1}"

cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR"; exit 1; }

submit() {
    local seed="$1"
    shift
    echo "+ SEED=${seed} sbatch --export=ALL,SEED=${seed} $*"
    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi
    SEED="${seed}" sbatch --export=ALL,SEED="${seed}" "$@"
}

echo "=== plus98 seeds [${SEEDS}]  DRY_RUN=${DRY_RUN}  pins=NUM_ENVS=128 (verified) ==="

if [ "$CONTINUE_APA18_21_SEED0" = "1" ]; then
    echo ""
    echo "--- continue seed0 density apa=18 and apa=21 → 50M (indep only) ---"
    echo "+ sbatch --array=0-1 ${EXTRA_DIR}/run_ippo_n4_density_continue_15to50m.sbatch"
    if [ "$DRY_RUN" != "1" ]; then
        # array 0=indep apa18, 1=indep apa21 (see sbatch header)
        sbatch --array=0-1 "${EXTRA_DIR}/run_ippo_n4_density_continue_15to50m.sbatch"
    fi
fi

for seed in $SEEDS; do
    echo ""
    echo "--- SEED=${seed} ---"

    # n_fixed98: N∈{3,5,6}; N=4 = shared baseline
    submit "$seed" "${EXTRA_DIR}/run_ippo_n_sweep_fixed98_50m.sbatch"

    # n_apa24p5: N∈{3,5,6}; N=4 = shared baseline
    submit "$seed" "${EXTRA_DIR}/run_ippo_n_sweep_apa24p5_50m.sbatch"

    # Shared baseline cell (feeds density/regrowth/tag/beam/layout/N-sweeps)
    if [ "$seed" = "1" ] && [ "$SKIP_BASELINE_SEED1" = "1" ]; then
        echo "+ skip baseline seed1 (already finished on W&B; override SKIP_BASELINE_SEED1=0)"
    else
        submit "$seed" "${EXTRA_DIR}/run_ippo_n4_plus98_baseline_seed_50m.sbatch"
    fi

    # n4_apa_density: apa∈{15,18,21,28}; 24.5 = baseline
    submit "$seed" "${EXTRA_DIR}/run_ippo_n4_apa_15_18_21_28_50m.sbatch"

    # n4_layout_blob98: scaffold only; plus_patch = baseline
    submit "$seed" "${EXTRA_DIR}/run_ippo_n4_blob98_scaffold_50m.sbatch"

    # n4_regrowth_plus98: rg∈{0.6,0.8,1.25,1.5}; rg=1.0 = baseline
    submit "$seed" "${EXTRA_DIR}/run_ippo_n4_regrowth_plus98_50m.sbatch"

    # n4_tag_timeout_plus98: tag∈{5,15,50,75}; tag=25 = baseline
    submit "$seed" "${EXTRA_DIR}/run_ippo_n4_tag_timeout_plus98_50m.sbatch"

    # n4_beam_plus98: 6 geometries; beam12×3 = baseline
    submit "$seed" "${EXTRA_DIR}/run_ippo_n4_beam_plus98_50m.sbatch"
done

echo ""
echo "=== submitted (or dry-run). Task estimate for SEEDS='1 2 3 4': ==="
echo "  continue seed0 apa18/21: 2"
echo "  fixed98 arrays 3×4 + apa24p5 3×4 + baseline 3 + density 4×4"
echo "  + blob 1×4 + regrowth 4×4 + tag 4×4 + beam 6×4  ≈ 103 GPU jobs"
echo "After finishes + wandb sync, refresh plots from your W&B entity."
