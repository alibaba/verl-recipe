#!/usr/bin/env bash
# =============================================================================
# Agentic RL Training — Disaggregated Mode (SGLang) — unified launcher
#
# Training and rollout on SEPARATE node pools with cross-node weight sync
# (NCCL or Mooncake), plus the agentic proxy server for remote agent execution.
# Invokes recipe.agentic.agentic_disagg_main.
#
# This one script replaces three former launchers, selected via env-var switches:
#   * train_agentic_disagg_sglang_sweagent.sh     -> TRIAL_MODE=local WEIGHT_SYNC=nccl     TOOL_FORMAT=qwen3_coder DATASET=/mnt/data/swe-bench-quick-2 PARAM_OFFLOAD=true
#   * train_agentic_disagg_sglang_sweagent_ack.sh -> TRIAL_MODE=ack   WEIGHT_SYNC=nccl     TOOL_FORMAT=qwen3_coder DATASET=/mnt/data/swe-bench-quick-2 PARAM_OFFLOAD=true
#   * k8s/run_agentic_disagg.sh                    -> TRIAL_MODE=local WEIGHT_SYNC=mooncake TOOL_FORMAT=hermes      DATASET=/mnt/data/swe-bench-verified TRAIN_BATCH_SIZE=1 PARAM_OFFLOAD=false
#
# -----------------------------------------------------------------------------
# ENV-VAR SWITCHES (defaults reproduce the simplest common run: local+nccl+hermes)
# -----------------------------------------------------------------------------
#   MODEL_PATH        (REQUIRED)          Path to the HF model. Script errors if unset.
#
#   TRIAL_MODE        local | ack   (default local)
#       local -> use_local_trial=true, remote_agent.agent_name=swe-agent (env AGENT_NAME),
#                built-in swe-agent, simple environment_kwargs (namespace/image_pull_secret/
#                tolerations). No custom agent.
#       ack   -> use_local_trial=false, remote_agent.agent_name=null,
#                agent_import_path=custom_agents.swe_agent:SweAgentACK (env AGENT_IMPORT_PATH),
#                remote model_name (env REMOTE_MODEL_NAME), harbor_server_url (env
#                HARBOR_SERVER_URL), agent_kwargs, and the FULL environment_kwargs including
#                extra_env (pip/uv mirrors) + pod_overrides (nodepool affinity + PVC mount).
#                Extra ack-only vars: HARBOR_SERVER_URL, AGENT_IMPORT_PATH, REMOTE_MODEL_NAME,
#                PVC_CLAIM_NAME, SANDBOX_NODEPOOL_ID.
#
#   WEIGHT_SYNC       nccl | mooncake  (default nccl)
#       Sets the default rollout checkpoint-engine backend (CHECKPOINT_ENGINE_BACKEND).
#       CHECKPOINT_ENGINE_BACKEND env var, if already set, OVERRIDES this (used by the
#       external_sglang wrappers to inject e.g. external_sglang_nccl).
#
#   TOOL_FORMAT       hermes | qwen3_coder  (default hermes)
#       Sets actor_rollout_ref.rollout.multi_turn.format and proxy_server.tool_format.
#       When qwen3_coder, ALSO adds the qwen3 SGLang parsers:
#         engine_kwargs.sglang.reasoning_parser=qwen3
#         engine_kwargs.sglang.tool_call_parser=qwen3_coder
#
#   DATASET           (default /mnt/data/swe-bench-verified)
#       Harbor data dir for BOTH data.train_harbor_dir and data.val_harbor_dir.
#       Falls back to the legacy HARBOR_DATA_DIR env var if DATASET is unset.
#
#   PARAM_OFFLOAD     true | false  (default true)
#       FSDP param_offload for actor + ref (optimizer_offload stays True).
#
#   Any extra CLI args are passed straight through to agentic_disagg_main via "$@".
#
# -----------------------------------------------------------------------------
# OTHER ADJUSTABLE ENV VARS (shared across all modes; defaults shown)
# -----------------------------------------------------------------------------
#   TRAIN_NNODES=1 ROLLOUT_NNODES=1 NGPUS_PER_NODE=8
#   TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=4096
#   PPO_MAX_TOKEN_LEN_PER_GPU=8192 ACTOR_LR=1e-6 TOTAL_EPOCHS=1 SAVE_FREQ=5 TEST_FREQ=5
#   ROLLOUT_TP=8 ROLLOUT_GPU_MEM_UTIL=0.7 ROLLOUT_N=2
#   PROXY_SERVER_URL=http://llm-proxy-server:80 LLM_PROXY_IP=llm-proxy-server
#   AGENT_NAME=swe-agent (local mode) IMAGE_PULL_SECRET=acr-pro-registry
#   MOONCAKE_MASTER / MOONCAKE_TE_META_DATA_SERVER / MOONCAKE_PROTOCOL / MOONCAKE_DEVICE /
#   MOONCAKE_GLOBAL_SEGMENT_SIZE  (exported for the Mooncake backend)
#   PROJECT_NAME=agentic_swe EXPERIMENT_NAME=<qwen3_coder|qwen3.6_27b>_agentic_<date>
#
# Usage:
#   MODEL_PATH=/mnt/models/Qwen3.6-27B bash train_agentic_disagg.sh
#   MODEL_PATH=/mnt/models/Qwen3-Coder TRIAL_MODE=ack TOOL_FORMAT=qwen3_coder \
#       DATASET=/mnt/data/swe-bench-quick-2 bash train_agentic_disagg.sh
# =============================================================================

set -xeuo pipefail

########################### switches ###########################

# --- Model (required) ---
if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "ERROR: MODEL_PATH is required (path to the HF model)." >&2
    exit 1
fi

TRIAL_MODE=${TRIAL_MODE:-local}
WEIGHT_SYNC=${WEIGHT_SYNC:-nccl}
TOOL_FORMAT=${TOOL_FORMAT:-hermes}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-true}
# DATASET is the harbor data dir; falls back to legacy HARBOR_DATA_DIR for compat.
DATASET=${DATASET:-${HARBOR_DATA_DIR:-/mnt/data/swe-bench-verified}}

# --- WEIGHT_SYNC -> default checkpoint-engine backend (CHECKPOINT_ENGINE_BACKEND wins) ---
case "${WEIGHT_SYNC}" in
    nccl)     DEFAULT_BACKEND=nccl ;;
    mooncake) DEFAULT_BACKEND=mooncake ;;
    *) echo "ERROR: WEIGHT_SYNC must be nccl|mooncake (got '${WEIGHT_SYNC}')." >&2; exit 1 ;;
esac
CHECKPOINT_ENGINE_BACKEND=${CHECKPOINT_ENGINE_BACKEND:-${DEFAULT_BACKEND}}

# --- TOOL_FORMAT -> multi_turn format (+ qwen3 parsers when qwen3_coder) ---
TOOL_FORMAT_ARGS=()
case "${TOOL_FORMAT}" in
    hermes) ;;
    qwen3_coder)
        TOOL_FORMAT_ARGS=(
            ++actor_rollout_ref.rollout.engine_kwargs.sglang.reasoning_parser=qwen3
            ++actor_rollout_ref.rollout.engine_kwargs.sglang.tool_call_parser=qwen3_coder
        )
        ;;
    *) echo "ERROR: TOOL_FORMAT must be hermes|qwen3_coder (got '${TOOL_FORMAT}')." >&2; exit 1 ;;
esac

########################### user-adjustable ###########################

# --- Cluster layout ---
TRAIN_NNODES=${TRAIN_NNODES:-1}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# --- Training hyperparams ---
train_batch_size=${TRAIN_BATCH_SIZE:-4}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
actor_lr=${ACTOR_LR:-1e-6}
total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-5}

# --- Rollout ---
rollout_tp=${ROLLOUT_TP:-8}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.7}
rollout_n=${ROLLOUT_N:-2}

# --- Agentic ---
# Standalone proxy mode: the proxy runs as a separate K8s Service.
# The InferenceWorkerClient in RemoteAgentLoop bridges SGLang to the proxy.
# NOTE: Do NOT export these — collect_yaml_env_overrides() injects them into
# Ray runtime_env.env_vars from the Hydra config. Exporting them here would
# cause collect_yaml_env_overrides to skip them (shell takes priority), so
# rollout workers would never receive them.
# Port :80 must be explicit — urlparse("http://host").port returns None.
PROXY_SERVER_URL=${PROXY_SERVER_URL:-http://llm-proxy-server:80}
LLM_PROXY_IP=${LLM_PROXY_IP:-llm-proxy-server}
IMAGE_PULL_SECRET=${IMAGE_PULL_SECRET:-acr-pro-registry}

# --- Mooncake ---
export MOONCAKE_MASTER=${MOONCAKE_MASTER:-127.0.0.1:50051}
export MOONCAKE_TE_META_DATA_SERVER=${MOONCAKE_TE_META_DATA_SERVER:-P2PHANDSHAKE}
export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}
export MOONCAKE_DEVICE=${MOONCAKE_DEVICE:-}
export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-4294967296}

PROJECT_NAME=${PROJECT_NAME:-agentic_swe}

########################### trial-mode remote_agent overrides ###########################

case "${TRIAL_MODE}" in
    local)
        # LOCAL trial: built-in swe-agent, simple environment_kwargs.
        AGENT_NAME=${AGENT_NAME:-swe-agent}
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3.6_27b_agentic_$(date +%Y%m%d_%H%M)}
        REMOTE_AGENT_ARGS=(
            remote_agent.agent_name="${AGENT_NAME}"
            remote_agent.proxy_server_url="${PROXY_SERVER_URL}"
            remote_agent.use_local_trial=true
            remote_agent.environment_import_path="harbor.environments.ack:ACKEnvironment"
            "++remote_agent.environment_kwargs={namespace: default, image_pull_secret: ${IMAGE_PULL_SECRET}, tolerations: [{key: node-role.alibabacloud.com/lingjun, operator: Exists, effect: NoSchedule}]}"
        )
        ;;
    ack)
        # ACK/remote trial on the kube-rl Harbor server with the custom SweAgentACK.
        # The FULL environment_kwargs (extra_env mirrors + pod_overrides affinity/PVC)
        # is preserved verbatim below — it is too structured to express as flat switches.
        HARBOR_SERVER_URL=${HARBOR_SERVER_URL:-http://kube-rl:8080}
        AGENT_IMPORT_PATH=${AGENT_IMPORT_PATH:-custom_agents.swe_agent:SweAgentACK}
        REMOTE_MODEL_NAME=${REMOTE_MODEL_NAME:-hosted_vllm/Qwen3-Coder}
        PVC_CLAIM_NAME=${PVC_CLAIM_NAME:-ym-dataset}
        SANDBOX_NODEPOOL_ID=${SANDBOX_NODEPOOL_ID:-np1ce790a49e8848d58353337541ba7a5f}
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_coder_agentic_$(date +%Y%m%d_%H%M)}
        REMOTE_AGENT_ARGS=(
            remote_agent.agent_name=null
            remote_agent.agent_import_path="${AGENT_IMPORT_PATH}"
            remote_agent.model_name="${REMOTE_MODEL_NAME}"
            remote_agent.proxy_server_url="${PROXY_SERVER_URL}"
            remote_agent.harbor_server_url="${HARBOR_SERVER_URL}"
            remote_agent.use_local_trial=false
            remote_agent.environment_import_path="harbor.environments.ack:ACKEnvironment"
            remote_agent.agent_kwargs='{total_cost_limit: 0, per_instance_cost_limit: 0}'
            "++remote_agent.environment_kwargs={namespace: default, image_pull_secret: ${IMAGE_PULL_SECRET}, tolerations: [{key: node-role.alibabacloud.com/lingjun, operator: Exists, effect: NoSchedule}], extra_env: [{name: UV_INDEX_URL, value: 'https://mirrors.aliyun.com/pypi/simple/'}, {name: PIP_INDEX_URL, value: 'https://mirrors.aliyun.com/pypi/simple/'}, {name: PIP_TRUSTED_HOST, value: 'mirrors.aliyun.com'}], pod_overrides: {spec: {affinity: {nodeAffinity: {requiredDuringSchedulingIgnoredDuringExecution: {nodeSelectorTerms: [{matchExpressions: [{key: node.alibabacloud.com/nodepool-id, operator: In, values: [${SANDBOX_NODEPOOL_ID}]}]}]}}}, containers: [{volumeMounts: [{name: data, mountPath: /mnt/data, readOnly: true}]}], volumes: [{name: data, persistentVolumeClaim: {claimName: ${PVC_CLAIM_NAME}}}]}}}"
        )
        ;;
    *) echo "ERROR: TRIAL_MODE must be local|ack (got '${TRIAL_MODE}')." >&2; exit 1 ;;
esac

########################### end user-adjustable ###########################

echo "=== Agentic Disaggregated Layout ==="
echo "  Trial mode:        ${TRIAL_MODE}"
echo "  Training nodes:    ${TRAIN_NNODES} x ${NGPUS_PER_NODE} GPUs/node"
echo "  Rollout nodes:     ${ROLLOUT_NNODES} x ${NGPUS_PER_NODE} GPUs/node"
echo "  Checkpoint engine: ${CHECKPOINT_ENGINE_BACKEND}"
echo "  Tool format:       ${TOOL_FORMAT}"
echo "  Param offload:     ${PARAM_OFFLOAD}"
echo "  Proxy:             ${PROXY_SERVER_URL}"
echo "  Data:              ${DATASET}"
echo "=========================================="

python3 -m recipe.agentic.agentic_disagg_main \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.return_raw_chat=true \
    data.train_harbor_dir="${DATASET}" \
    data.val_harbor_dir="${DATASET}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.prompt_key=instance_id \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.format=${TOOL_FORMAT} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=remote_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/agentic/remote-agent.yaml \
    actor_rollout_ref.rollout.checkpoint_engine.backend=${CHECKPOINT_ENGINE_BACKEND} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
    ${TOOL_FORMAT_ARGS[@]+"${TOOL_FORMAT_ARGS[@]}"} \
    algorithm.rollout_correction.bypass_mode=True \
    proxy_server.llm_proxy_ip="${LLM_PROXY_IP}" \
    proxy_server.tool_format=${TOOL_FORMAT} \
    "${REMOTE_AGENT_ARGS[@]}" \
    trainer.balance_batch=True \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${TRAIN_NNODES} \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=${total_epochs} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    rollout.nnodes=${ROLLOUT_NNODES} \
    rollout.n_gpus_per_node=${NGPUS_PER_NODE} \
    "$@"
