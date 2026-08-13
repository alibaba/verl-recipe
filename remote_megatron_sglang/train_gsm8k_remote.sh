#!/usr/bin/env bash
# Launch verl (CPU-only orchestrator) against an external Megatron train cluster
# (PyTorchJob) + external SGLang inference cluster (RoleBasedGroup).
#
# Prereqs (see README.md):
#   1. kubectl apply -f k8s/pytorchjob-megatron.yaml   # Kubeflow training-operator
#   2. kubectl apply -f k8s/rbg-sglang.yaml            # RoleBasedGroup operator
#   3. Both Services reachable from where this driver runs.
set -xeuo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_ENDPOINT=${TRAIN_ENDPOINT:-http://megatron-train-master:8000}
SGLANG_ENDPOINT=${SGLANG_ENDPOINT:-http://sglang-rbg-leader:30000}
WEIGHT_SYNC_TRANSPORT=${WEIGHT_SYNC_TRANSPORT:-nccl_http}
TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}

# Wire the megatron_sglang adapter into verl core with NO per-backend if-branch
# in framework code: VERL_USE_EXTERNAL_MODULES makes `import verl` load the
# recipe's register.py, which registers the RemoteBackend class + its forwarder
# worker. main_ppo then resolves the worker by name via the registry. The env
# var is inherited by Ray worker processes so they self-register too.
export VERL_USE_EXTERNAL_MODULES=recipe.remote_megatron_sglang.register

# Keep verl's own primary config (verl/trainer/config/ppo_trainer.yaml); only add
# the recipe config dir to the Hydra search path so the `remote_backend` group
# resolves. `remote_backend=megatron_sglang` OVERRIDES the optional default in
# ppo_trainer.yaml's defaults list (no `+` — the group is already declared).
python -m verl.trainer.main_ppo \
  hydra.searchpath="[file://${RECIPE_DIR}/config]" \
  trainer.remote_backend=megatron_sglang \
  remote_backend=megatron_sglang \
  remote_backend.megatron_sglang.train_endpoint="${TRAIN_ENDPOINT}" \
  "remote_backend.megatron_sglang.sglang_endpoints=[${SGLANG_ENDPOINT}]" \
  remote_backend.megatron_sglang.weight_sync.transport="${WEIGHT_SYNC_TRANSPORT}" \
  "+actor_rollout_ref.rollout.gateway_url=${SGLANG_ENDPOINT}" \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size=64 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=4
# NOTE: n_gpus_per_node=1 / nnodes=1 gives exactly ONE CPU-only forwarder worker
# (RemoteBackendTrainer sets use_gpu=False), as required by
# MegatronSGLangBackend.requires_single_forwarder(). The "1" is a forwarder
# count, not a GPU allocation.
