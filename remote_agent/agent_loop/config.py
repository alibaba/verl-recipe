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
"""Framework-agnostic core config for RemoteAgentLoop. Framework-specific
settings are opaque and passed through under ``runner.kwargs``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RemoteAgentCoreConfig:
    advertised_host: str = "0.0.0.0"
    proxy_port: int = 0
    tool_format: str = "hermes"

    max_retries: int = 3
    retry_base_delay: float = 1.0
    poll_interval: float = 2.0

    task_roots: list[str] = field(default_factory=list)
    task_template: str = "{instance_id}"

    runner_name: str = "example"
    runner_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dictconfig(cls, cfg) -> RemoteAgentCoreConfig:
        def g(path, default):
            cur = cfg
            for part in path.split("."):
                if cur is None or part not in cur:
                    return default
                cur = cur[part]
            return cur if cur is not None else default

        advertised = os.getenv("REMOTE_AGENT_ADVERTISED_HOST") or g("proxy.advertised_host", "0.0.0.0")
        # deep-convert OmegaConf containers to plain python so downstream
        # pydantic models can serialize runner kwargs (e.g. tolerations)
        runner_kwargs = g("runner.kwargs", {}) or {}
        try:
            from omegaconf import OmegaConf

            runner_kwargs = OmegaConf.to_container(OmegaConf.create(runner_kwargs), resolve=True)
        except Exception:
            runner_kwargs = dict(runner_kwargs)
        return cls(
            advertised_host=advertised,
            proxy_port=int(g("proxy.port", 0)),
            tool_format=g("proxy.tool_format", "hermes"),
            max_retries=int(g("retry.max_retries", 3)),
            retry_base_delay=float(g("retry.retry_base_delay", 1.0)),
            poll_interval=float(g("retry.poll_interval", 2.0)),
            task_roots=list(g("task_path.roots", []) or []),
            task_template=g("task_path.template", "{instance_id}"),
            runner_name=g("runner.name", "example"),
            runner_kwargs=runner_kwargs,
        )
