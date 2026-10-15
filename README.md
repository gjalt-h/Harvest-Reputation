# Commons Harvest + reputation (thesis code)

IPPO on **Commons Harvest: Open** with optional reputation / enforcement.
Depends on a **trimmed** vendored copy of [SocialJax](https://github.com/cooperativex/SocialJax)
under `foundation_code/SocialJax-main/` (Apache-2.0; see that folder’s `LICENSE` and `README.md`).

This release ships **code, configs, and launch scripts** to reproduce thesis
experiments. It does **not** include datasets, checkpoints, or W&B logs.

## Structure

```
README.md
requirements.txt
configs/plot_runs.example.yaml     # copy → plot_runs.yaml for local W&B run IDs
GIFs/                              # curated experiment visualizations
  README.md                        # tiered GIF descriptions (expand to view each GIF)
foundation_code/SocialJax-main/    # trimmed SocialJax
harvest_socialjax/
  algorithms/ippo/                 # train / render / metrics
  configs/ippo/                    # Hydra YAMLs (flat; Hydra path unchanged)
  envs/                            # Harvest Open, reputation v1/v2, police, paper layouts
  wrappers/
  scripts/
    main/                          # thesis-canonical launchers + plot scripts
    extra/                         # env micro-sweeps + reputation v1 (not main results)
    sbatch/                        # shared SLURM / W&B helpers
  results/thesis_tables/           # aggregated CSV summaries for number checks
```

Paper-scale **Lupu / Perolat** ASCII layouts live in `envs/env_scale_presets.py`.

## Setup

```bash
# Python 3.10; GPU: prefer SocialJax conda env or:
pip install -r requirements.txt
# For CUDA, install jax[cuda12]==0.6.2 as in foundation_code/SocialJax-main/requirements.txt

export JAX_PLATFORMS=cpu   # login node / smoke; unset on GPU nodes
```

SocialJax is discovered via `harvest_socialjax/socialjax_path.py`.

### W&B (optional)

```bash
export WANDB_ENTITY=YOUR_ENTITY
export WANDB_PROJECT=harvest_ippo_rep   # or harvest_ippo for no-reputation
export WANDB_API_KEY=...               # if online / sync
```

Project **names** are generic; never commit entity names or API keys.
Plot seed0 IDs: copy `configs/plot_runs.example.yaml` → `configs/plot_runs.yaml`.

## Data

No external dataset. The orchard is generated from `ENV_SCALE`, `ENV_NUM_AGENTS`,
and `ENV_KWARGS`. **Thesis default: N=4**, test-scale, plus-patch ~98 apples.
`ippo_test_shared.yaml` defaults to `ENV_NUM_AGENTS: 4`. Deviations are only in
extra micro-sweep sbatches (documented there).

## Train

```bash
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py --config-name <yaml> [overrides…]
```

From **repo root** on a cluster:

```bash
export WANDB_ENTITY=YOUR_ENTITY
mkdir -p logs
sbatch --export=ALL harvest_socialjax/scripts/main/<script>.sh
```

### Learning stacks

| Stack | γ | Entropy | Used for |
|-------|---|--------|----------|
| **A** | 0.995 | 0.01 flat | Police, enf period=1, matching no-peer baselines |
| **B (g99)** | 0.99 | 0.05→0.005 | Peer ±5, good-immunity, zap ablations, matching no-peer baselines |

Both no-peer baseline launchers are kept. Peer / gimm / zap are **g99 only**.

### Main experiments (`scripts/main/`)

| Experiment | Example launcher | Config |
|------------|------------------|--------|
| Indep ToC | `run_ippo_test.sbatch` | `ippo_test` |
| Shared ToC (N=4) | `run_ippo_shared_n4.sbatch` | `ippo_test_shared` |
| V2 no-peer A | `sbatch_v2_no_peer_indep_shared_50kep.sh` | `ippo_test_reputation_v2` + overrides |
| V2 no-peer B | `sbatch_v2_no_peer_indep_shared_g99_ent005_floor0005_50kep.sh` | same |
| Police1 | `sbatch_v2_scratch_police1_no_peer_*.sh` | `ippo_test_reputation_v2_police` |
| Enf p1 | `sbatch_v2_scratch_enf_period_p1_seeds1to7_50m.sh` | `ippo_test_reputation_v2` |
| Peer ±5 g99 | `sbatch_v2_scratch_peer_br5_g5_g99_*.sh` | `ippo_test_reputation_v2` |
| Good immunity g99 | `sbatch_v2_scratch_peer_br5_gimm_g99_*.sh` | same |
| Zap ablations g99 | `sbatch_v2_scratch_zap_ablations_g99_*.sh` | same |
| Drop @ u200 | `sbatch_v2_*u200*.sh` / drop_* | resume from parent |
| Mask-rep | `*_mask_rep_*.sh` | same as parent |

Exact seeds and Hydra overrides are **inside each sbatch**.

### Extra (`scripts/extra/`)

Env micro-sweeps (N / density / regrowth / tag / beam) and **reputation v1**
launchers / configs — supporting setups, not main thesis cells.

### Curated GIFs

Key dynamics from the experiments are collected in [`GIFs/`](GIFs/).
That folder has its own README with short descriptions and collapsible GIF
embeds: see [`GIFs/README.md`](GIFs/README.md).

To render additional GIFs from checkpoints:

```bash
python harvest_socialjax/scripts/main/render_ippo_from_checkpoint.py --help
sbatch harvest_socialjax/scripts/main/run_ippo_render_from_checkpoint.sbatch
```

### Plots

Thesis comparison scripts: `harvest_socialjax/scripts/main/plot_*.py`.
Set `WANDB_ENTITY` / `WANDB_PROJECT` and optional `configs/plot_runs.yaml`.
Aggregated verification tables: `harvest_socialjax/results/thesis_tables/`.
