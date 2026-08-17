#!/usr/bin/env bash
# Launch verl (CPU-only orchestrator) against an external Megatron train cluster
# (PyTorchJob) + external SGLang inference cluster (RoleBasedGroup or a plain
# Pod + Service).
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
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-$HOME/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-$HOME/data/gsm8k/test.parquet}

# Wire the megatron_sglang adapter into verl core with NO per-backend if-branch
# in framework code: VERL_USE_EXTERNAL_MODULES makes `import verl` load the
# recipe's register.py, which registers the RemoteBackend class, its forwarder
# worker, the external rollout replica, and the `remote_megatron_sglang` V1
# trainer mode. The env var is inherited by Ray worker processes (and by the
# TaskRunnerV1 actor) so they self-register too.
export VERL_USE_EXTERNAL_MODULES=recipe.remote_megatron_sglang.register

# MegatronSGLangRolloutReplica reads the external SGLang base URL(s) from here
# (comma-separated). Kept out of RolloutConfig so the core dataclass schema is
# untouched.
export MEGATRON_SGLANG_ENDPOINTS=${MEGATRON_SGLANG_ENDPOINTS:-$SGLANG_ENDPOINT}

# Keep verl's own primary config (verl/trainer/config/ppo_trainer.yaml); only add
# the recipe config dir to the Hydra search path so the `remote_backend` group
# resolves. The group is NOT declared in ppo_trainer.yaml's defaults list, so it
# has to be *appended* with `+` (both for the group and for the trainer field).
python -m verl.trainer.main_ppo \
  hydra.searchpath="[file://${RECIPE_DIR}/config]" \
  "+remote_backend=megatron_sglang" \
  "+trainer.remote_backend=megatron_sglang" \
  trainer.v1.trainer_mode=remote_megatron_sglang \
  remote_backend.megatron_sglang.train_endpoint="${TRAIN_ENDPOINT}" \
  "remote_backend.megatron_sglang.sglang_endpoints=[${SGLANG_ENDPOINT}]" \
  remote_backend.megatron_sglang.weight_sync.transport="${WEIGHT_SYNC_TRANSPORT}" \
  actor_rollout_ref.rollout.name=megatron_sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size=64 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=4
# NOTE: on this trainer n_gpus_per_node=1 / nnodes=1 is a *forwarder process*
# count, not a GPU allocation: RemoteMegatronSGLangTrainer builds the verl-side
# worker group on a CPU-only (0-GPU) resource pool. The external Megatron
# cluster owns the training GPUs; the external SGLang cluster owns the
# inference GPUs.
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
