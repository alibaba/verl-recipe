"""API request and response models for the Harbor Agent Run server."""

from __future__ import annotations

import math
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")


class AgentConfig(BaseModel):
    """Agent configuration provided by the caller."""

    name: str | None = None
    import_path: str | None = None
    model_name: str | None = None
    override_timeout_sec: float | None = None
    override_setup_timeout_sec: float | None = None
    max_timeout_sec: float | None = None
    llm_proxy_url: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class VerifierConfig(BaseModel):
    """Verifier configuration provided by the caller."""

    override_timeout_sec: float | None = None
    max_timeout_sec: float | None = None
    disable: bool = False


class AgentRunRequest(BaseModel):
    """Request to submit an agent run."""

    job_id: str = Field(
        description="Identifier for grouping related trials in the frontend. "
        "All trials sharing the same job_id are aggregated under one job."
    )
    task_id: str = Field(
        description="Unique identifier for this task, used for tracing and "
        "joining data with the LLM proxy."
    )
    task_path: str = Field(description="Path to the task directory")
    agent: AgentConfig = Field(description="Agent configuration")
    timeout_multiplier: float = Field(default=1.0, ge=0.1)
    max_retries: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Maximum number of automatic retries on failure. "
        "A trial is retried when it fails (not on timeout). "
        "Each retry uses the same task and configuration.",
    )
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    environment_overrides: dict[str, Any] | None = Field(
        default=None,
        description="Optional overrides for environment config (e.g. override_cpus)",
    )
    environment_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Extra keyword arguments passed directly to the environment constructor, "
        "merged with server defaults and overriding any conflicting keys. "
        "Allowed keys: namespace, context, kubeconfig, image_pull_secret, "
        "service_account, node_selector, tolerations, memory_limit_multiplier, "
        "registry, use_sandbox_claim, claim_timeout, "
        "sandbox_env_vars, sandbox_labels, sandbox_annotations, pod_overrides.",
    )
    llm_proxy_url: str | None = Field(
        default=None,
        description="URL of the LLM proxy server. When set, the agent will use "
        "this as its LLM endpoint so token_ids and logprobs can be captured.",
    )

    @field_validator("environment_kwargs")
    @classmethod
    def validate_environment_kwargs(
        cls, v: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if v is None:
            return v
        allowed_keys = {
            "namespace", "context", "kubeconfig", "image_pull_secret",
            "service_account", "node_selector", "tolerations",
            "memory_limit_multiplier", "registry",
            "use_sandbox_claim", "claim_timeout",
            "sandbox_env_vars", "sandbox_labels", "sandbox_annotations",
            "use_async_exec",
            "pod_overrides",
            # Legacy (deprecated — use pod_overrides instead)
            "pod_privileged", "pod_run_as_user", "pod_run_as_group",
            "pod_capabilities_add", "pod_capabilities_drop",
            "pod_annotations", "pod_labels", "extra_env",
            "extra_volumes", "extra_volume_mounts", "init_containers",
        }
        unknown = set(v.keys()) - allowed_keys
        if unknown:
            raise ValueError(
                f"Unknown environment_kwargs keys: {sorted(unknown)}. "
                f"Allowed keys are: {sorted(allowed_keys)}"
            )
        return v


class RolloutDetailResponse(BaseModel):
    """Rollout detail data returned to RL frameworks."""

    prompt_token_ids: list[list[int]] | None = None
    completion_token_ids: list[list[int]] | None = None
    logprobs: list[list[float]] | None = None


class TokenUsage(BaseModel):
    """Token usage statistics from the agent execution."""

    n_input_tokens: int | None = None
    n_output_tokens: int | None = None
    n_cache_tokens: int | None = None
    cost_usd: float | None = None


class AgentRunResponse(BaseModel):
    """Response from an agent run, containing RL-relevant fields for training."""

    run_id: str
    status: str = Field(description="completed | failed | timeout")
    rewards: dict[str, float | int] | None = None
    rollout_details: list[RolloutDetailResponse] | None = None
    token_usage: TokenUsage | None = None
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Agent metadata (may contain token_ids/mask_ids for legacy RL integrations)",
    )
    error: str | None = None
    result_uri: str | None = Field(
        default=None, description="URI of the full TrialResult in storage"
    )
    retry_count: int = Field(
        default=0,
        description="Number of retries performed before reaching this result.",
    )


# ---------------------------------------------------------------------------
# Paginated responses & filters (reused from Harbor Viewer models)
# ---------------------------------------------------------------------------


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper (same shape as Harbor Viewer)."""

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse[T]":
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total / page_size) if page_size > 0 else 0,
        )


class FilterOption(BaseModel):
    """A filter option with a value and count."""

    value: str
    count: int


class TrialFilters(BaseModel):
    """Available filter options for trials list."""

    statuses: list[FilterOption]
    agents: list[FilterOption]
    tasks: list[FilterOption]


class TrialCompareEntry(BaseModel):
    """Single trial entry in a comparison view."""

    run_id: str
    task_name: str
    agent_name: str | None = None
    status: str
    reward: float | int | None = None
    duration: float | None = None
    n_input_tokens: int | None = None
    n_output_tokens: int | None = None
    n_cache_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    started_at: str | None = None
