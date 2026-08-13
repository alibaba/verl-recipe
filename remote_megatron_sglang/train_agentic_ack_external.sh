#!/usr/bin/env bash
# =============================================================================
# Agentic SWE (ack) training driven by an EXTERNAL, out-of-Ray SGLang.
#
# Thin wrapper over recipe/agentic/train_agentic_disagg.sh (TRIAL_MODE=ack).
# Generation always goes: standing llm-proxy-server (relay) -> verl
# InferenceWorkerClient -> GlobalRequestLoadBalancer -> ExternalSGLangProxyActor
# shim -> external SGLang (ExternalLLMServerManager). agentic_disagg_main calls
# start_lb_registry. The WEIGHT_SYNC switch selects the weight-sync transport:
#
#   WEIGHT_SYNC=mooncake (default)
#       CUDA-IPC ReceiverCE, colocated with the external SGLang.
#       backend            = mooncake              (CHECKPOINT_ENGINE_BACKEND)
#       checkpoint manager = verl.workers.rollout.external_sglang.checkpoint_manager.ExternalCheckpointManager
#       engine_kwargs      = external_sglang.{model_path, receivers:[{url,resource:sglang_node,tp_size}]}
#       SGLANG_TP default  = 2
#       SGLANG_ENDPOINT    = http://10.8.0.5:30000
#       External SGLang must be joined to Ray with resource "sglang_node".
#
#   WEIGHT_SYNC=nccl
#       NCCL-over-HTTP: trainer rank0 forms a NCCL group with the external
#       SGLang's TP workers via HTTP (/init_weights_update_group) and broadcasts
#       full HF weights (/update_weights_from_distributed). No ReceiverCE, no
#       CUDA IPC; SGLang need NOT be in Ray.
#       backend            = external_sglang_nccl  (CHECKPOINT_ENGINE_BACKEND)
#       checkpoint manager = recipe.remote_megatron_sglang.checkpoint_engine.ExternalSGLangCheckpointManager
#       custom_backend_module = recipe.remote_megatron_sglang.checkpoint_engine
#       engine_kwargs      = external_sglang_nccl.{sglang_endpoints:[...]}
#       SGLANG_TP default  = 8
#       SGLANG_ENDPOINT    = http://10.8.0.31:30000  (cluster-reachable pod IP)
#       External SGLang must serve MODEL_PATH with
#         --tool-call-parser qwen3_coder --reasoning-parser qwen3.
#
# External SGLang must already serve MODEL_PATH at SGLANG_ENDPOINT.
#
# ENV-VAR SWITCHES:
#   WEIGHT_SYNC   mooncake | nccl   (default mooncake)
#   MODEL_PATH        (default /mnt/models/Qwen3.6-27B)
#   SGLANG_ENDPOINT   (default per-backend, see above)
#   SGLANG_TP         (default per-backend: mooncake=2, nccl=8) -> ROLLOUT_TP
# Plus every env var of train_agentic_disagg.sh (TRAIN_BATCH_SIZE, TOTAL_EPOCHS,
# ROLLOUT_N, DATASET, TOOL_FORMAT, ...). Extra CLI args pass through via "$@".
#
# Usage:
#   bash recipe/remote_megatron_sglang/train_agentic_ack_external.sh                 # mooncake
#   WEIGHT_SYNC=nccl bash recipe/remote_megatron_sglang/train_agentic_ack_external.sh
# =============================================================================
set -xeuo pipefail

WEIGHT_SYNC=${WEIGHT_SYNC:-mooncake}
MODEL_PATH=${MODEL_PATH:-/mnt/models/Qwen3.6-27B}

case "${WEIGHT_SYNC}" in
    mooncake)
        SGLANG_ENDPOINT=${SGLANG_ENDPOINT:-http://10.8.0.5:30000}
        SGLANG_TP=${SGLANG_TP:-2}
        export CHECKPOINT_ENGINE_BACKEND=mooncake
        WEIGHT_SYNC_ARGS=(
            +actor_rollout_ref.rollout.external_sglang_endpoints="[\"${SGLANG_ENDPOINT}\"]"
            +actor_rollout_ref.rollout.llm_server_manager_class=verl.workers.rollout.external_sglang.server_manager.ExternalLLMServerManager
            +actor_rollout_ref.rollout.checkpoint_manager_class=verl.workers.rollout.external_sglang.checkpoint_manager.ExternalCheckpointManager
            "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.external_sglang={model_path: ${MODEL_PATH}, receivers: [{url: 'http://127.0.0.1:30000', resource: sglang_node, tp_size: ${SGLANG_TP}}]}"
        )
        ;;
    nccl)
        SGLANG_ENDPOINT=${SGLANG_ENDPOINT:-http://10.8.0.31:30000}   # cluster-reachable pod IP
        SGLANG_TP=${SGLANG_TP:-8}
        export CHECKPOINT_ENGINE_BACKEND=external_sglang_nccl
        WEIGHT_SYNC_ARGS=(
            +actor_rollout_ref.rollout.external_sglang_endpoints="[\"${SGLANG_ENDPOINT}\"]"
            +actor_rollout_ref.rollout.llm_server_manager_class=verl.workers.rollout.external_sglang.server_manager.ExternalLLMServerManager
            +actor_rollout_ref.rollout.checkpoint_manager_class=recipe.remote_megatron_sglang.checkpoint_engine.ExternalSGLangCheckpointManager
            actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=recipe.remote_megatron_sglang.checkpoint_engine
            "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.external_sglang_nccl={sglang_endpoints: [\"${SGLANG_ENDPOINT}\"]}"
        )
        ;;
    *) echo "ERROR: WEIGHT_SYNC must be mooncake|nccl (got '${WEIGHT_SYNC}')." >&2; exit 1 ;;
esac

# env consumed by the underlying train_agentic_disagg.sh (ack trial mode).
export MODEL_PATH
export TRIAL_MODE=ack
# Preserve the old _ack.sh defaults the external path relied on: qwen3_coder tool
# format (also enables the qwen3 SGLang parsers) + swe-bench-quick-2 data dir.
export TOOL_FORMAT=${TOOL_FORMAT:-qwen3_coder}
export DATASET=${DATASET:-/mnt/data/swe-bench-quick-2}
export HARBOR_DATA_DIR=${HARBOR_DATA_DIR:-/mnt/data/swe-bench-quick-2}
export ROLLOUT_TP=${SGLANG_TP}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export TRAIN_NNODES=1
export ROLLOUT_NNODES=1
# PROXY_SERVER_URL: intentionally NOT exported so agentic_main injects it into
# the Ray actor runtime_env (shell-set vars are skipped by collect_yaml_env_overrides).
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
export SAVE_FREQ=-1
export TEST_FREQ=-1
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export ROLLOUT_N=${ROLLOUT_N:-2}

bash recipe/agentic/train_agentic_disagg.sh \
    "${WEIGHT_SYNC_ARGS[@]}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    "$@"
