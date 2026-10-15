# SocialJax (trimmed vendored subset)

This directory is a **trimmed** copy of [SocialJax](https://github.com/cooperativex/SocialJax)
(Apache-2.0) for use by `harvest_socialjax`.

It keeps only the pieces required at runtime:

- `socialjax/registration.py` — `make()` registry (harvest only)
- `socialjax/environments/` — `MultiAgentEnv`, spaces, `common_harvest` (incl. rendering)
- `socialjax/wrappers/baselines.py` — logging / CT-rollout wrappers

Upstream algorithms, other substrates, checkpoints, and docs were removed for this
thesis release. See the upstream repository for the full project.

License: see `LICENSE` (Apache-2.0).
