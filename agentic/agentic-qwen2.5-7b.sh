#!/bin/bash
#
# Agentic recipe: SWE-Agent + Qwen2.5-7B on ACK/L20
#

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?请设置 MODEL_PATH，例如 /var/model/Qwen2.5-7B-Instruct}"

ENVIRONMENT="${ENVIRONMENT:-docker}"
HARBOR_TASK_DIR="${HARBOR_TASK_DIR:-/home/verl/swebench-verified}"
REMOTE_AGENT_ENVIRONMENT_IMPORT_PATH="harbor.environments.docker.docker:DockerEnvironment"
REMOTE_AGENT_ENVIRONMENT_KWARGS="{}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export LLM_PROXY_IP="${LLM_PROXY_IP:-$(hostname -i)}"

HARBOR_TRAIN_LIMIT="${HARBOR_TRAIN_LIMIT:-null}"
HARBOR_VAL_LIMIT="${HARBOR_VAL_LIMIT:-null}"
HARBOR_OVERWRITE="${HARBOR_OVERWRITE:-false}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-null}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-qwen2.5-7b}"
TRAINER_VAL_BEFORE_TRAIN="${TRAINER_VAL_BEFORE_TRAIN:-true}"
TRAINER_LOG_VAL_GENERATIONS="${TRAINER_LOG_VAL_GENERATIONS:-50}"
TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-1}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-5}"

if [ "${ENVIRONMENT}" = "ack" ]; then
    NAMESPACE="${NAMESPACE:-default}"
    REGISTRY="${REGISTRY:?请设置 REGISTRY，例如 zlaa-test-registry-vpc.us-east-1.cr.aliyuncs.com/zlaa}"
    KUBECONFIG="${KUBECONFIG:-}"
    IMAGE_PULL_SECRET="${IMAGE_PULL_SECRET:-acr-credential-secret-aggregation}"
    BASE_IMAGE_REGISTRY="${BASE_IMAGE_REGISTRY:-}"
    SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-rayclustertest}"
    USE_BUILDKIT="${USE_BUILDKIT:-true}"
    BUILDKIT_ADDRESS="${BUILDKIT_ADDRESS:-tcp://buildkitd:1234}"
    USE_SANDBOX_CLAIM="${USE_SANDBOX_CLAIM:-false}"
    CLAIM_TIMEOUT="${CLAIM_TIMEOUT:-300}"
    SANDBOXSET_REPLICAS="${SANDBOXSET_REPLICAS:-5}"

    REMOTE_AGENT_ENVIRONMENT_IMPORT_PATH="harbor.environments.ack:ACKEnvironment"
    REMOTE_AGENT_ENVIRONMENT_KWARGS="{namespace: '${NAMESPACE}', registry: '${REGISTRY}', kubeconfig: '${KUBECONFIG}', image_pull_secret: '${IMAGE_PULL_SECRET}', base_image_registry: '${BASE_IMAGE_REGISTRY}', service_account: '${SERVICE_ACCOUNT}', use_buildkit: ${USE_BUILDKIT}, buildkit_address: '${BUILDKIT_ADDRESS}', use_sandbox_claim: ${USE_SANDBOX_CLAIM}, claim_timeout: ${CLAIM_TIMEOUT}, sandboxset_replicas: ${SANDBOXSET_REPLICAS}}"
fi

python3 -m recipe.agentic.agentic_main \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.return_raw_chat=true \
    data.train_batch_size=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.prompt_key=instance_id \
    data.train_harbor_dir="${HARBOR_TASK_DIR}" \
    data.val_harbor_dir="${HARBOR_TASK_DIR}" \
    data.harbor_train_limit="${HARBOR_TRAIN_LIMIT}" \
    data.harbor_val_limit="${HARBOR_VAL_LIMIT}" \
    data.harbor_overwrite="${HARBOR_OVERWRITE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    remote_agent.environment_import_path="${REMOTE_AGENT_ENVIRONMENT_IMPORT_PATH}" \
    "+remote_agent.environment_kwargs=${REMOTE_AGENT_ENVIRONMENT_KWARGS}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE='${NCCL_P2P_DISABLE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE='${NCCL_IB_DISABLE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG='${NCCL_DEBUG}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_ASYNC_ERROR_HANDLING='${NCCL_ASYNC_ERROR_HANDLING}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_NCCL_ASYNC_ERROR_HANDLING='${TORCH_NCCL_ASYNC_ERROR_HANDLING}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES='${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LLM_PROXY_IP='${LLM_PROXY_IP}'" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=remote_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/agentic/remote-agent.yaml \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.logger="[console]" \
    trainer.project_name=remote-agent \
    trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train="${TRAINER_VAL_BEFORE_TRAIN}" \
    trainer.log_val_generations="${TRAINER_LOG_VAL_GENERATIONS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${TRAINER_SAVE_FREQ}" \
    trainer.default_local_dir=/var/model/checkpoints/qwen2.5-7b \
    trainer.test_freq="${TRAINER_TEST_FREQ}" \
    trainer.total_epochs=1
