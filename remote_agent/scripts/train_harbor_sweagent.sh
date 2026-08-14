#!/usr/bin/env bash
# Launch remote_agent PPO training with the Harbor + swe-agent runner.
#
# Required:
#   MODEL_PATH       HF model directory
#   TRAIN_TASKS      dir of Harbor task subdirs (each with task.toml + instruction.md)
#   VAL_TASKS        same layout, for validation
#
# Optional (ACK sandbox mode — use when running on a K8s cluster without Docker):
#   SANDBOX_MODE=ack               switch from DockerEnvironment to ACKEnvironment
#   SANDBOX_NAMESPACE=default      K8s namespace for sandbox pods
#   SANDBOX_IMAGE_PULL_SECRET=      name of the K8s imagePullSecret for sandbox images
#   SANDBOX_MEMORY_LIMIT_MULT=      multiplier on task.toml memory (e.g. 4 → 16Gi)
#   SANDBOX_TOLERATIONS_KEY=       toleration key (default: node-role.alibabacloud.com/lingjun)
#
# All additional args are forwarded to python -m remote_agent.main as Hydra overrides.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH to a HF model dir}"
export REMOTE_AGENT_ADVERTISED_HOST="${REMOTE_AGENT_ADVERTISED_HOST:-$(hostname -i | awk '{print $1}')}"
# swe-agent uses litellm under the hood; prefix with openai/ so litellm routes
# via the OpenAI provider to our proxy (OPENAI_BASE_URL).
MODEL_NAME="${MODEL_NAME:-openai/$(basename "${MODEL_PATH}")}"

ARGS=(
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.rollout.remote_agent.runner.name=harbor
  actor_rollout_ref.rollout.remote_agent.runner.kwargs.agent_name=swe-agent
  ++actor_rollout_ref.rollout.remote_agent.runner.kwargs.model_name="${MODEL_NAME}"
  # Pass advertised_host via config (not env) so Ray worker processes inherit it.
  actor_rollout_ref.rollout.remote_agent.proxy.advertised_host="${REMOTE_AGENT_ADVERTISED_HOST}"
  data.train_harbor_dir="${TRAIN_TASKS:-/data/tasks/train}"
  data.val_harbor_dir="${VAL_TASKS:-/data/tasks/val}"
)

if [ "${SANDBOX_MODE:-}" = "ack" ]; then
  NS="${SANDBOX_NAMESPACE:-default}"
  IPS="${SANDBOX_IMAGE_PULL_SECRET:-}"
  MEM_MULT="${SANDBOX_MEMORY_LIMIT_MULT:-}"
  TOL_KEY="${SANDBOX_TOLERATIONS_KEY:-node-role.alibabacloud.com/lingjun}"
  ENV_IMPORT="harbor.environments.ack:ACKEnvironment"

  # Build the environment_kwargs dict (OmegaConf inline YAML)
  KW="namespace: ${NS}"
  [ -n "$IPS" ] && KW="${KW}, image_pull_secret: ${IPS}"
  [ -n "$MEM_MULT" ] && KW="${KW}, memory_limit_multiplier: ${MEM_MULT}"
  KW="${KW}, tolerations: [{key: ${TOL_KEY}, operator: Exists, effect: NoSchedule}]"

  ARGS+=(
    actor_rollout_ref.rollout.remote_agent.runner.kwargs.environment_import_path="${ENV_IMPORT}"
    "++actor_rollout_ref.rollout.remote_agent.runner.kwargs.environment_kwargs={${KW}}"
  )
elif [ "${SANDBOX_MODE:-}" = "e2b" ]; then
  # E2B mode: sandboxes are claimed via HTTP from sandbox-manager/gateway.
  # E2B_API_KEY / E2B_API_URL / E2B_SANDBOX_URL must be set as env vars.
  SBX_SET="${SANDBOX_SET:-slime-sbx-astropy-14309}"
  ENV_IMPORT="harbor.environments.e2b:E2BEnvironment"
  KW="sandbox_set_name: ${SBX_SET}, override_claim_image: true"

  ARGS+=(
    actor_rollout_ref.rollout.remote_agent.runner.kwargs.environment_import_path="${ENV_IMPORT}"
    "++actor_rollout_ref.rollout.remote_agent.runner.kwargs.environment_kwargs={${KW}}"
  )
fi

exec python -m remote_agent.main "${ARGS[@]}" "$@"
