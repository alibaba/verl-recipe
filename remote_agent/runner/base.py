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
"""The single seam between the framework-agnostic RemoteAgentLoop core and any
external agent framework. A runner's job: given a task and the OpenAI base_url
the external agent must use, run that agent to completion and return a result.
Runners never touch tokens — the proxy records LLM traffic out-of-band."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentTask:
    instance_id: str
    task_path: str | None  # resolved by the core; runner interprets it
    row: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunResult:
    status: str  # "completed" | "failed" | "error"
    rewards: dict[str, float] = field(default_factory=dict)
    error: str | None = None


class ExternalAgentRunner(ABC):
    """Execute an external agent against one task. Subclasses live in
    ``runner/<framework>/`` and may import framework-specific deps."""

    def __init__(self, kwargs: dict[str, Any] | None = None):
        self.kwargs = dict(kwargs or {})

    @abstractmethod
    async def run(
        self, *, task: AgentTask, agent_base_url: str, sampling_params: dict[str, Any], **kwargs
    ) -> AgentRunResult: ...

    def build_dataset(self, data_cfg) -> tuple[str, str] | None:
        """Optional: runners that own a dataset format return (train_files,
        val_files). Called by main.py so the entry point never imports the
        framework directly. Default: no dataset materialization."""
        return None


# --- name registry (mirrors verl's agent-loop @register / RemoteBackendRegistry) ---

_RUNNERS: dict[str, type[ExternalAgentRunner]] = {}
# Names whose class lives in a module we lazily import on first use, so that
# `import harbor` only fires when the harbor runner is actually selected.
_LAZY_MODULES: dict[str, str] = {
    "harbor": "remote_agent.runner.harbor.runner",
    "example": "remote_agent.runner.example_runner",
}


def register_runner(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _RUNNERS[name] = cls
        return cls

    return deco


def create_runner(name: str, kwargs: dict[str, Any]) -> ExternalAgentRunner:
    if name not in _RUNNERS and name in _LAZY_MODULES:
        importlib.import_module(_LAZY_MODULES[name])  # fires @register_runner
    if name not in _RUNNERS:
        raise KeyError(
            f"unknown remote-agent runner {name!r}; registered={sorted(_RUNNERS)} lazy={sorted(_LAZY_MODULES)}"
        )
    return _RUNNERS[name](kwargs)
