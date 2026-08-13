# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data models for the LLM proxy session recording."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class CompletionRecord(BaseModel):
    """Record of a single LLM completion call captured by the proxy."""

    request_messages: list[dict[str, Any]] = Field(
        description="Messages for this turn. For the first turn this is the "
        "full messages array; for subsequent turns only the new messages "
        "(delta since the previous turn) are stored to avoid duplicating "
        "conversation history."
    )
    completion_text: str = Field(description="Generated completion text")
    completion_token_ids: list[int] = Field(
        default_factory=list,
        description="Token IDs of the generated completion (may be empty when cached on worker)",
    )
    completion_logprobs: list[float] = Field(
        default_factory=list,
        description="Log probabilities for each generated token (may be empty when cached on worker)",
    )
    finish_reason: str | None = Field(default=None, description="Reason generation stopped: stop, tool_calls, length")
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Parsed tool calls from the completion, if any"
    )
    timing: dict[str, float] | None = Field(
        default=None,
        description="Per-turn timing data (epoch timestamps and durations in ms)",
    )
    node_id: str | None = Field(
        default=None,
        description="Node hostname where inference occurred",
    )
    gpu_id: str | None = Field(
        default=None,
        description="GPU device ID where inference occurred",
    )
    worker_id: str | None = Field(
        default=None,
        description="Worker client ID that processed this request",
    )
    rank_info: dict[str, str] | None = Field(
        default=None,
        description="Distributed rank information (rank, local_rank, node_rank)",
    )


class SessionRecord(BaseModel):
    """All recorded data for a single agent session (one rollout)."""

    session_id: str
    turns: list[CompletionRecord] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    completed: bool = False


class TrainingRoundTiming(BaseModel):
    """Timing data for a single training round (epoch/step)."""

    epoch: int = Field(description="Training epoch number")
    global_step: int = Field(description="Global step number")
    step_start: float = Field(description="Step start timestamp (epoch seconds)")
    step_end: float | None = Field(default=None, description="Step end timestamp (epoch seconds)")
    inference_start: float | None = Field(default=None, description="Inference/generation phase start timestamp")
    inference_end: float | None = Field(default=None, description="Inference/generation phase end timestamp")
    weight_sync_start: float | None = Field(default=None, description="Weight sync start timestamp")
    weight_sync_end: float | None = Field(default=None, description="Weight sync end timestamp")
    training_start: float | None = Field(default=None, description="Training update phase start timestamp")
    training_end: float | None = Field(default=None, description="Training update phase end timestamp")
    reward_start: float | None = Field(default=None, description="Reward computation start timestamp")
    reward_end: float | None = Field(default=None, description="Reward computation end timestamp")
    log_prob_start: float | None = Field(default=None, description="Log-prob computation start timestamp")
    log_prob_end: float | None = Field(default=None, description="Log-prob computation end timestamp")
    ref_log_prob_start: float | None = Field(default=None, description="Reference log-prob computation start timestamp")
    ref_log_prob_end: float | None = Field(default=None, description="Reference log-prob computation end timestamp")
    critic_start: float | None = Field(default=None, description="Critic computation start timestamp")
    critic_end: float | None = Field(default=None, description="Critic computation end timestamp")
    advantage_start: float | None = Field(default=None, description="Advantage computation start timestamp")
    advantage_end: float | None = Field(default=None, description="Advantage computation end timestamp")
    update_critic_start: float | None = Field(default=None, description="Critic update start timestamp")
    update_critic_end: float | None = Field(default=None, description="Critic update end timestamp")
    update_actor_start: float | None = Field(default=None, description="Actor update start timestamp")
    update_actor_end: float | None = Field(default=None, description="Actor update end timestamp")
    checkpoint_start: float | None = Field(default=None, description="Checkpoint save start timestamp")
    checkpoint_end: float | None = Field(default=None, description="Checkpoint save end timestamp")
    phase_durations: dict[str, float] | None = Field(default=None, description="Raw timing_raw marked_timer durations")
