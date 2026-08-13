"""Test script: submit a swe-agent trial via the Harbor Agent Run server.

Mimics kube-rl/python/submit_swe_agent_ack.py but uses the serversdk
located at /workspace/recipe/agentic/serversdk/ on the head pod.

Usage (inside head pod):

    # The Harbor Agent Run server (kube-rl) is already running as a K8s service
    # at http://kube-rl:8080. No need to start it manually.
    #
    # Set env vars and run this script:
    OPENAI_API_KEY="sk-xxx" \
    OPENAI_ENDPOINT="https://dashscope.aliyuncs.com/compatible-mode/v1" \
    MODEL_NAME="openai/qwen3-coder-plus" \
    python3 /workspace/recipe/agentic/test_submit_swe_agent.py

    # To use the LLM proxy (for RL token capture) instead of an external API:
    LLM_PROXY_URL="http://llm-proxy-server:80" \
    MODEL_NAME="hosted_vllm/Qwen3-Coder" \
    python3 /workspace/recipe/agentic/test_submit_swe_agent.py

    # Custom task:
    TASK_PATH="/mnt/data/swe-bench-verified/django__django-11099" \
    python3 /workspace/recipe/agentic/test_submit_swe_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# ---------------------------------------------------------------------------
# Path setup: ensure serversdk is importable from /workspace/recipe/agentic/
# ---------------------------------------------------------------------------
RECIPE_AGENTIC_DIR = "/workspace/recipe/agentic"
if RECIPE_AGENTIC_DIR not in sys.path:
    sys.path.insert(0, RECIPE_AGENTIC_DIR)

from serversdk.client import AgentRunClient  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (all via environment variables with sensible defaults)
# ---------------------------------------------------------------------------

# Harbor Agent Run server URL (kube-rl service in the cluster)
HARBOR_SERVER_URL = os.environ.get("HARBOR_SERVER_URL", "http://kube-rl:8080")

# LLM API credentials (required for external API; ignored when using LLM proxy)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT", "")

# Model name — use "hosted_vllm/..." when LLM_PROXY_URL is set
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/qwen3-coder-plus")

# Task directory on the head pod (SWE-bench Verified dataset)
TASK_PATH = os.environ.get(
    "TASK_PATH",
    "/mnt/data/swe-bench-verified/astropy__astropy-12907",
)

# LLM proxy URL — when set, the remote agent uses this as its LLM endpoint
# so token_ids and logprobs can be captured for RL training.
LLM_PROXY_URL = os.environ.get("LLM_PROXY_URL", "")

# K8s configuration for sandbox pods
IMAGE_PULL_SECRET = os.environ.get("IMAGE_PULL_SECRET", "acr-pro-registry")
PVC_CLAIM_NAME = os.environ.get("PVC_CLAIM_NAME", "ym-dataset")

# Agent configuration: use built-in "swe-agent" (patched with ACK modifications)
# or a custom agent via agent_import_path
AGENT_NAME = os.environ.get("AGENT_NAME", "")
AGENT_IMPORT_PATH = os.environ.get("AGENT_IMPORT_PATH", "custom_agents.swe_agent:SweAgentACK")

# Polling configuration
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "5.0"))
POLL_TIMEOUT = float(os.environ.get("POLL_TIMEOUT", "1800.0"))


def _validate_config() -> None:
    """Validate required environment variables."""
    errors = []

    if not LLM_PROXY_URL:
        # External API mode — need API key and endpoint
        if not OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when LLM_PROXY_URL is not set")
        if not OPENAI_ENDPOINT:
            errors.append("OPENAI_ENDPOINT is required when LLM_PROXY_URL is not set")

    if not os.path.isdir(TASK_PATH):
        errors.append(f"Task directory not found: {TASK_PATH}")

    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        print("\nUsage:")
        print("  OPENAI_API_KEY=sk-xxx OPENAI_ENDPOINT=https://... python3 test_submit_swe_agent.py")
        print("  # or with LLM proxy:")
        print("  LLM_PROXY_URL=http://llm-proxy-server:80 python3 test_submit_swe_agent.py")
        sys.exit(1)


def _build_agent_kwargs() -> dict:
    """Build agent_kwargs for the swe-agent."""
    kwargs: dict = {
        "per_instance_cost_limit": 0,
        "total_cost_limit": 0,
    }

    if LLM_PROXY_URL:
        # When using LLM proxy, the proxy URL is set via llm_proxy_url parameter
        # The agent needs a dummy API key pointing to the proxy
        kwargs["api_base"] = LLM_PROXY_URL
        kwargs["api_key"] = "dummy"  # proxy handles auth
    else:
        kwargs["api_key"] = OPENAI_API_KEY
        kwargs["api_base"] = OPENAI_ENDPOINT

    return kwargs


def _build_environment_kwargs() -> dict:
    """Build environment_kwargs for the ACK environment."""
    extra_env: list[dict] = []

    if LLM_PROXY_URL:
        # When using LLM proxy, set OPENAI_BASE_URL to the proxy
        extra_env.extend([
            {"name": "OPENAI_API_KEY", "value": "dummy"},
            {"name": "OPENAI_BASE_URL", "value": LLM_PROXY_URL},
        ])
    else:
        extra_env.extend([
            {"name": "OPENAI_API_KEY", "value": OPENAI_API_KEY},
            {"name": "OPENAI_BASE_URL", "value": OPENAI_ENDPOINT},
        ])

    # Recommended: internal mirrors for uv and pip
    extra_env.extend([
        {"name": "UV_INDEX_URL", "value": "https://mirrors.aliyun.com/pypi/simple/"},
        {"name": "PIP_INDEX_URL", "value": "https://mirrors.aliyun.com/pypi/simple/"},
        {"name": "PIP_TRUSTED_HOST", "value": "mirrors.aliyun.com"},
    ])

    env_kwargs: dict = {
        "image_pull_secret": IMAGE_PULL_SECRET,
        "extra_env": extra_env,
        # REQUIRED: mount PVC at /mnt/data for pre-cloned sweagent-repo
        "pod_overrides": {
            "spec": {
                "containers": [{
                    "volumeMounts": [
                        {"name": "data", "mountPath": "/mnt/data", "readOnly": True},
                    ],
                }],
                "volumes": [
                    {"name": "data", "persistentVolumeClaim": {"claimName": PVC_CLAIM_NAME}},
                ],
            },
        },
    }

    return env_kwargs


def main() -> None:
    _validate_config()

    print("=" * 60)
    print("SWE-agent Test Submission")
    print("=" * 60)
    print(f"  Server URL:      {HARBOR_SERVER_URL}")
    print(f"  Task path:       {TASK_PATH}")
    print(f"  Model:           {MODEL_NAME}")
    print(f"  Agent:           {AGENT_NAME or AGENT_IMPORT_PATH}")
    print(f"  LLM proxy:       {LLM_PROXY_URL or '(disabled)'}")
    print(f"  Image secret:    {IMAGE_PULL_SECRET}")
    print(f"  PVC claim:       {PVC_CLAIM_NAME}")
    print(f"  Poll timeout:    {POLL_TIMEOUT}s")
    print("=" * 60)

    client = AgentRunClient(HARBOR_SERVER_URL, timeout=POLL_TIMEOUT + 60)

    # Build submission parameters
    submit_kwargs: dict = dict(
        task_path=TASK_PATH,
        job_id="verl-test",
        task_id="verl-test-task-001",
        model_name=MODEL_NAME,
        agent_kwargs=_build_agent_kwargs(),
        environment_kwargs=_build_environment_kwargs(),
        disable_verifier=False,
        max_retries=0,
        poll_interval=POLL_INTERVAL,
        poll_timeout=POLL_TIMEOUT,
    )

    # Use built-in agent name or custom import path
    if AGENT_IMPORT_PATH:
        submit_kwargs["agent_import_path"] = AGENT_IMPORT_PATH
    else:
        submit_kwargs["agent_name"] = AGENT_NAME

    # Set LLM proxy URL for RL token capture
    if LLM_PROXY_URL:
        submit_kwargs["llm_proxy_url"] = LLM_PROXY_URL

    print("\nSubmitting task...")
    result = asyncio.run(client.run_async_task(**submit_kwargs))

    print("\n" + "=" * 60)
    print("Result")
    print("=" * 60)
    print(f"  run_id:    {result.run_id}")
    print(f"  status:    {result.status}")
    print(f"  error:     {result.error}")
    print(f"  rewards:   {result.rewards}")
    if result.token_usage:
        print(f"  tokens:    input={result.token_usage.n_input_tokens} "
              f"output={result.token_usage.n_output_tokens} "
              f"cache={result.token_usage.n_cache_tokens}")
        print(f"  cost:      ${result.token_usage.cost_usd}")
    if result.retry_count:
        print(f"  retries:   {result.retry_count}")
    print("=" * 60)

    # Exit with error code if the run failed
    if result.status in ("failed", "timeout"):
        sys.exit(1)


if __name__ == "__main__":
    main()
