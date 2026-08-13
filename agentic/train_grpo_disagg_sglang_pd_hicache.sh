#!/usr/bin/env bash
# Fully Separated Disaggregated GRPO RL Training:
#   Training and Rollout on SEPARATE node pools (cross-node weight sync).
#
#   1. Training-Rollout disaggregation: separate node pools
#   2. SGLang PD disaggregation: prefill + decode on separate GPUs
#   3. Mooncake checkpoint engine: cross-node P2P weight sync (training -> rollout)
#   4. Mooncake HiCache: SGLang L3 KVCache offload
#   5. Mooncake PD transfer: KV cache transfer (prefill -> decode)
#   6. AI Gateway + SGLang Router (verl Ray actor): external LB across instances
#
# Request flow:
#   verl → AI Gateway → K8s Service → SGLang Router (Ray actor) → Prefill/Decode
# Weight update flow (cross-node, bypasses gateway):
#   verl CheckpointEngineManager → Mooncake TransferEngine (P2P) → SGLang servers
#
# Node layout:
#   Training nodes: all GPUs for Actor FSDP + Ref model
#   Rollout nodes:  all GPUs for SGLang PD (e.g. TP=4 prefill + TP=4 decode)
#                   + SGLangRouterActor (CPU, :30080) for AI gateway
#
# Usage:
#   GATEWAY_URL=http://ai-gateway:8080 bash train_grpo_disagg_sglang_pd_hicache.sh
#   # Custom node counts:
#   TRAIN_NNODES=4 ROLLOUT_NNODES=2 GATEWAY_URL=http://gw:8080 bash train_grpo_disagg_sglang_pd_hicache.sh

set -xeuo pipefail

########################### user-adjustable ###########################

# --- Cluster layout (fully separated: training and rollout on different nodes) ---
TRAIN_NNODES=${TRAIN_NNODES:-2}                              # Number of training nodes
ROLLOUT_NNODES=${ROLLOUT_NNODES:-2}                          # Number of rollout nodes
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}                          # GPUs per node (same for both pools)
N_GPUS_TRAINING=${N_GPUS_TRAINING:-${NGPUS_PER_NODE}}        # Training GPUs per node (all)
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-${NGPUS_PER_NODE}}          # Rollout GPUs per node (all)

# --- Model ---
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}

# --- Data ---
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"$HOME/data/gsm8k/test.parquet"}

# --- Training hyperparams ---
train_batch_size=${TRAIN_BATCH_SIZE:-512}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}
total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

# --- Rollout ---
rollout_tp=${ROLLOUT_TP:-4}                                   # Prefill TP size (full node: TP=4)
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.80}
rollout_n=${ROLLOUT_N:-5}

# --- SGLang PD disaggregation ---
PD_ENABLED=${PD_ENABLED:-true}
PD_PREFILL_REPLICAS=${PD_PREFILL_REPLICAS:-1}                 # Must be 1 (current limitation)
PD_DECODE_REPLICAS=${PD_DECODE_REPLICAS:-1}                   # Number of decode servers per replica
PD_DECODE_TP=${PD_DECODE_TP:-4}                               # Decode TP (full node: TP=4)
PD_TRANSFER_BACKEND=${PD_TRANSFER_BACKEND:-mooncake}          # KV cache: prefill -> decode
PD_IB_DEVICE=${PD_IB_DEVICE:-}                                # RDMA NIC, e.g. "mlx5_roce0"

# --- Mooncake checkpoint engine (weight sync: trainer -> rollout) ---
CHECKPOINT_ENGINE_BACKEND=${CHECKPOINT_ENGINE_BACKEND:-mooncake}

# --- Mooncake HiCache (SGLang L3 KVCache offload) ---
ENABLE_HICACHE=${ENABLE_HICACHE:-true}
HICACHE_RATIO=${HICACHE_RATIO:-2.0}
HICACHE_MEM_LAYOUT=${HICACHE_MEM_LAYOUT:-page_first}
HICACHE_WRITE_POLICY=${HICACHE_WRITE_POLICY:-write_through}
HICACHE_PREFETCH_POLICY=${HICACHE_PREFETCH_POLICY:-timeout}

# --- AI Gateway (external load balancer for rollout requests) ---
# Set GATEWAY_URL to route generation through an external AI gateway
# instead of verl's built-in GlobalRequestLoadBalancer.
# The gateway selects the best SGLang Router instance; PD routing
# within each replica is handled by the SGLang Router.
# Leave empty to use verl's built-in load balancer.
GATEWAY_URL=${GATEWAY_URL:-}

# --- SGLang Router (verl-managed Ray actor, not K8s sidecar) ---
# Automatically enabled when GATEWAY_URL is set. The router is launched
# by verl after prefill/decode servers start, using their discovered
# addresses. Override ROUTER_PORT to change the HTTP listen port.
ENABLE_ROUTER=${ENABLE_ROUTER:-}
ROUTER_PORT=${ROUTER_PORT:-30080}
# Auto-enable router when using AI gateway
if [ -n "${GATEWAY_URL}" ] && [ -z "${ENABLE_ROUTER}" ]; then
    ENABLE_ROUTER=true
fi

# --- Mooncake env vars ---
export MOONCAKE_MASTER=${MOONCAKE_MASTER:-127.0.0.1:50051}
export MOONCAKE_TE_META_DATA_SERVER=${MOONCAKE_TE_META_DATA_SERVER:-P2PHANDSHAKE}
export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}
export MOONCAKE_DEVICE=${MOONCAKE_DEVICE:-}
export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-4294967296}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_disagg_mooncake}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_grpo_disagg_sglang_mooncake_$(date +%Y%m%d_%H%M)}

########################### end user-adjustable ###########################

echo "=== Fully Separated Disaggregated Layout ==="
echo "  Training nodes: ${TRAIN_NNODES} × ${N_GPUS_TRAINING} GPUs/node"
echo "  Rollout nodes:  ${ROLLOUT_NNODES} × ${N_GPUS_ROLLOUT} GPUs/node"
echo "  PD mode:  ${PD_ENABLED} (prefill TP=${rollout_tp}, decode TP=${PD_DECODE_TP})"
echo "  Checkpoint engine: ${CHECKPOINT_ENGINE_BACKEND} (cross-node P2P weight sync)"
echo "  HiCache:  ${ENABLE_HICACHE}"
echo "  AI Gateway: ${GATEWAY_URL:-disabled (using built-in LB)}"
echo "  SGLang Router: ${ENABLE_ROUTER:-disabled} (port ${ROUTER_PORT})"
echo "============================================"

########################### build parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

# --- Key: disable hybrid_engine for disaggregated mode ---
DISAGG_CORE=(
    actor_rollout_ref.hybrid_engine=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # Required for disaggregated mode:
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.load_format=safetensors
)

# --- Mooncake checkpoint engine (weight sync: trainer -> rollout) ---
CHECKPOINT_ENGINE=(
    actor_rollout_ref.rollout.checkpoint_engine.backend=${CHECKPOINT_ENGINE_BACKEND}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048
)

# --- SGLang PD disaggregation config ---
PD_DISAGG=()
if [ "${PD_ENABLED}" = "true" ]; then
    PD_DISAGG+=(
        actor_rollout_ref.rollout.disaggregation.enabled=True
        actor_rollout_ref.rollout.disaggregation.prefill_replicas=${PD_PREFILL_REPLICAS}
        actor_rollout_ref.rollout.disaggregation.decode_replicas=${PD_DECODE_REPLICAS}
        actor_rollout_ref.rollout.disaggregation.transfer_backend=${PD_TRANSFER_BACKEND}
    )
    if [ "${PD_DECODE_TP}" != "null" ]; then
        PD_DISAGG+=(
            actor_rollout_ref.rollout.disaggregation.decode_tensor_model_parallel_size=${PD_DECODE_TP}
        )
    fi
    if [ -n "${PD_IB_DEVICE}" ]; then
        PD_DISAGG+=(
            actor_rollout_ref.rollout.disaggregation.ib_device=${PD_IB_DEVICE}
        )
    fi
    # SGLang Router (verl-managed Ray actor for AI gateway integration)
    if [ "${ENABLE_ROUTER}" = "true" ]; then
        PD_DISAGG+=(
            actor_rollout_ref.rollout.disaggregation.enable_router=True
            actor_rollout_ref.rollout.disaggregation.router_port=${ROUTER_PORT}
        )
    fi
fi

# --- Mooncake HiCache KVCache offload (SGLang L3 cache) ---
SGLANG_EXTRA=()
if [ "${ENABLE_HICACHE}" = "true" ]; then
    SGLANG_EXTRA+=(
        +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_hierarchical_cache=True
        +actor_rollout_ref.rollout.engine_kwargs.sglang.hicache_storage_backend=mooncake
        +actor_rollout_ref.rollout.engine_kwargs.sglang.hicache_ratio=${HICACHE_RATIO}
        +actor_rollout_ref.rollout.engine_kwargs.sglang.hicache_mem_layout=${HICACHE_MEM_LAYOUT}
        +actor_rollout_ref.rollout.engine_kwargs.sglang.hicache_write_policy=${HICACHE_WRITE_POLICY}
        +actor_rollout_ref.rollout.engine_kwargs.sglang.hicache_storage_prefetch_policy=${HICACHE_PREFETCH_POLICY}
    )
fi

# --- Rollout correction (required for disaggregated) ---
ALGORITHM=(
    algorithm.rollout_correction.bypass_mode=True
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# --- Trainer resource pool (separate node pool) ---
TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${N_GPUS_TRAINING}
    trainer.nnodes=${TRAIN_NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

# Rollout resource pool (separate node pool)
ROLLOUT_POOL=(
    rollout.nnodes=${ROLLOUT_NNODES}
    rollout.n_gpus_per_node=${N_GPUS_ROLLOUT}
)

# --- AI Gateway (external load balancer) ---
# When set, generation requests are routed through the gateway HTTP endpoint
# instead of verl's built-in Ray RPC + GlobalRequestLoadBalancer.
# Weight updates still go through direct Ray RPC (CheckpointEngineManager).
GATEWAY=()
if [ -n "${GATEWAY_URL}" ]; then
    GATEWAY+=(
        +actor_rollout_ref.rollout.gateway_url="${GATEWAY_URL}"
    )
fi

########################### launch ###########################
# Use the one-step off-policy entry point for training-rollout disaggregation
python3 -m verl.experimental.one_step_off_policy.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${DISAGG_CORE[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${CHECKPOINT_ENGINE[@]}" \
    "${PD_DISAGG[@]}" \
    "${SGLANG_EXTRA[@]}" \
    "${ALGORITHM[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${ROLLOUT_POOL[@]}" \
    "${GATEWAY[@]}" \
    "$@"
