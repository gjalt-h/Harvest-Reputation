"""Default Weights & Biases project names for training pipelines.

Override at runtime with environment variables (recommended for public use):

  WANDB_ENTITY   your W&B username or team (required for cloud logging)
  WANDB_PROJECT  project name (defaults below if unset)

Do not commit personal entity names or API keys.
"""

# IPPO harvest runs (no reputation).
IPPO_WANDB_PROJECT = "harvest_ippo"

# IPPO harvest + reputation (v1 or v2).
IPPO_REPUTATION_WANDB_PROJECT = "harvest_ippo_rep"
