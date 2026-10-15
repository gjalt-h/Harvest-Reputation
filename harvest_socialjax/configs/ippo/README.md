# IPPO Hydra configs

Hydra path (from `algorithms/ippo/train_ippo_harvest.py`):
`config_path="../../configs/ippo"`.

## Main (thesis)

| YAML | Role |
|------|------|
| `ippo_test.yaml` | Indep ToC baseline |
| `ippo_test_shared.yaml` | Shared-reward twin (**N=4**) |
| `ippo_test_reputation_v2.yaml` | Reputation v2 stem (most sbatch overrides) |
| `ippo_test_reputation_v2_police.yaml` | Police agent stem |

Final cells are defined by **sbatches** under `scripts/main/` (seeds, γ/ENT, arm knobs).

## Extra (not main results)

| YAML | Role |
|------|------|
| `ippo_test_reputation_*.yaml` (non-v2) | Reputation **v1** |

Launchers: `scripts/extra/` (env micro-sweeps + rep v1).
