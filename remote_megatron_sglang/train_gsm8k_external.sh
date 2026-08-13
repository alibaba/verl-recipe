#!/usr/bin/env bash
# End-to-end GRPO on gsm8k using an EXTERNAL (out-of-Ray) SGLang for both
# generation and weight sync. Trainer = OneStepOffRayTrainer (disagg).
#
# Prereqs (already set up in this cluster):
#   - external SGLang serving the SAME model at $SGLANG_URL (TP=2), pod joined
#     to Ray with resource "sglang_node".
#   - mooncake env present on all pods; rollout-workers scaled to 0.
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-/mnt/models/Qwen2.5-3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/mnt/data/gsm8k/train_small.parquet}
TEST_FILE=${TEST_FILE:-/mnt/data/gsm8k/test_small.parquet}
SGLANG_ENDPOINT=${SGLANG_ENDPOINT:-http://10.8.0.5:30000}   # cluster-reachable
SGLANG_TP=${SGLANG_TP:-2}

cd /workspace

python3 -m verl.experimental.one_step_off_policy.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${SGLANG_TP} \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=mooncake \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    +actor_rollout_ref.rollout.external_sglang_endpoints="[\"${SGLANG_ENDPOINT}\"]" \
    +actor_rollout_ref.rollout.llm_server_manager_class=verl.workers.rollout.external_sglang.server_manager.ExternalLLMServerManager \
    +actor_rollout_ref.rollout.checkpoint_manager_class=verl.workers.rollout.external_sglang.checkpoint_manager.ExternalCheckpointManager \
    "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.external_sglang={model_path: ${MODEL_PATH}, receivers: [{url: 'http://127.0.0.1:30000', resource: sglang_node, tp_size: ${SGLANG_TP}}]}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name=external_sglang \
    trainer.experiment_name=gsm8k_ext \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=2 "$@"
