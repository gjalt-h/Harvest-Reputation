# harvest_socialjax

Thesis package: Commons Harvest Open (8 actions) + IPPO + reputation.

## Dependency

Unmodified **SocialJax** is expected at `foundation_code/SocialJax-main/`
(trimmed vendored subset) or on `PYTHONPATH`. See `socialjax_path.py`.

## Quick start

```bash
export JAX_PLATFORMS=cpu
python harvest_socialjax/algorithms/ippo/train_ippo_harvest.py --config-name ippo_test
```

Hydra configs: `configs/ippo/`. Launchers: `scripts/main/` (thesis) and
`scripts/extra/` (micro-sweeps + reputation v1).

See the repository root `README.md` for the full experiment table and W&B setup.

## Notes

- Live env is **8 actions** (no gifting).
- Paper Lupu / Perolat maps are ASCII presets in `envs/env_scale_presets.py`.
