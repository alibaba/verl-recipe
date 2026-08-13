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
"""A generic, dependency-free ExternalAgentRunner: launch any subprocess agent,
handing it the proxy base_url via standard OpenAI env vars. Doubles as the core
test vehicle (proves RemoteAgentLoop runs with zero framework deps)."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from .base import AgentRunResult, AgentTask, ExternalAgentRunner, register_runner


@register_runner("example")
class ExampleRunner(ExternalAgentRunner):
    """kwargs:
    command: list[str]  — the agent command to spawn (required)
    env:     dict       — extra env vars (optional)
    """

    async def run(
        self, *, task: AgentTask, agent_base_url: str, sampling_params: dict[str, Any], **kwargs
    ) -> AgentRunResult:
        command = self.kwargs.get("command")
        if not command:
            return AgentRunResult(status="error", rewards={}, error="no command configured")

        env = dict(os.environ)
        env.update(self.kwargs.get("env", {}))
        env["OPENAI_BASE_URL"] = agent_base_url
        env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "remote-agent-proxy")

        proc = await asyncio.create_subprocess_exec(
            *command,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        if proc.returncode == 0:
            return AgentRunResult(status="completed", rewards={})
        return AgentRunResult(
            status="error",
            rewards={},
            error=f"agent exited with exit code {proc.returncode}: {out.decode()[:500]}",
        )
