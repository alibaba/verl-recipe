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
"""HarborRunner — executes a Harbor Trial in-process (v1: local mode only).
The ONLY module that imports harbor."""

from __future__ import annotations

import logging
from typing import Any

from ..base import AgentRunResult, AgentTask, ExternalAgentRunner, register_runner

logger = logging.getLogger(__name__)


@register_runner("harbor")
class HarborRunner(ExternalAgentRunner):
    """kwargs: agent_name, agent_import_path, model_name, agent_kwargs,
    environment_import_path (default harbor.environments.docker.docker:DockerEnvironment),
    environment_overrides, environment_kwargs."""

    async def run(
        self, *, task: AgentTask, agent_base_url: str, sampling_params: dict[str, Any], **kwargs
    ) -> AgentRunResult:
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
        from harbor.trial.trial import Trial

        k = self.kwargs
        env_overrides = dict(k.get("environment_overrides", {}))
        env_overrides.setdefault("OPENAI_API_KEY", "remote-agent-proxy")
        env_overrides.setdefault("OPENAI_BASE_URL", agent_base_url)

        agent_kwargs = dict(k.get("agent_kwargs", {}))
        agent_kwargs.update(
            {
                "model_base_url": agent_base_url,
                "api_base": agent_base_url,
                "api_key": "sk-remote-agent",
                "session_id": task.instance_id,
                "temperature": sampling_params.get("temperature", 1.0),
                "top_p": sampling_params.get("top_p", 1.0),
            }
        )

        cfg = TrialConfig(
            trial_name=task.instance_id,
            task=TaskConfig(path=task.task_path),
            agent=AgentConfig(
                name=k.get("agent_name"),
                import_path=k.get("agent_import_path"),
                model_name=k.get("model_name"),
                kwargs=agent_kwargs,
                env=env_overrides,
            ),
            environment=EnvironmentConfig(
                import_path=k.get("environment_import_path", "harbor.environments.docker.docker:DockerEnvironment"),
                env=env_overrides,
                kwargs=dict(k.get("environment_kwargs", {})),
            ),
        )
        try:
            trial = await Trial.create(cfg)
            result = await trial.run()
        except Exception as e:  # noqa: BLE001 — surface as a result, never raise
            logger.exception("harbor trial %s crashed", task.instance_id)
            return AgentRunResult(status="error", rewards={}, error=f"{type(e).__name__}: {e}")

        if result.exception_info is None:
            rewards = (result.verifier_result.rewards if result.verifier_result is not None else {}) or {}
            return AgentRunResult(status="completed", rewards=rewards)

        exc = result.exception_info
        return AgentRunResult(status="error", rewards={}, error=f"{exc.exception_type}: {exc.exception_message}")

    def build_dataset(self, data_cfg) -> tuple[str, str] | None:
        train_root = data_cfg.get("train_harbor_dir")
        val_root = data_cfg.get("val_harbor_dir")
        if not train_root and not val_root:
            return None
        from .dataset import build_verl_parquets

        cache = data_cfg.get("harbor_cache_dir", "~/.cache/verl/remote_agent/harbor")
        return build_verl_parquets(train_root, val_root, cache)
