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
import sys
import types

import pytest


def _install_fake_harbor(monkeypatch, exception_info=None, rewards=None):
    harbor = types.ModuleType("harbor")
    trial_mod = types.ModuleType("harbor.trial.trial")
    cfg_mod = types.ModuleType("harbor.models.trial.config")

    class _Cfg:
        def __init__(self, *a, **k):
            pass

    for name in ("TrialConfig", "TaskConfig", "AgentConfig", "EnvironmentConfig"):
        setattr(cfg_mod, name, _Cfg)

    class _Result:
        def __init__(self):
            self.exception_info = exception_info
            self.verifier_result = types.SimpleNamespace(rewards=rewards or {"reward": 1.0})

    class _Trial:
        @classmethod
        async def create(cls, cfg):
            return cls()

        async def run(self):
            return _Result()

    trial_mod.Trial = _Trial

    monkeypatch.setitem(sys.modules, "harbor", harbor)
    monkeypatch.setitem(sys.modules, "harbor.trial", types.ModuleType("harbor.trial"))
    monkeypatch.setitem(sys.modules, "harbor.trial.trial", trial_mod)
    monkeypatch.setitem(sys.modules, "harbor.models", types.ModuleType("harbor.models"))
    monkeypatch.setitem(sys.modules, "harbor.models.trial", types.ModuleType("harbor.models.trial"))
    monkeypatch.setitem(sys.modules, "harbor.models.trial.config", cfg_mod)


@pytest.mark.asyncio
async def test_harbor_runner_completed(monkeypatch, tmp_path):
    _install_fake_harbor(monkeypatch, exception_info=None, rewards={"reward": 2.0})
    from remote_agent.runner.base import AgentTask
    from remote_agent.runner.harbor.runner import HarborRunner

    runner = HarborRunner(
        {"agent_name": "swe-agent", "environment_import_path": "harbor.environments.docker:DockerEnvironment"}
    )
    task = AgentTask(instance_id="i", task_path=str(tmp_path), row={})
    res = await runner.run(task=task, agent_base_url="http://h:9/i/v1", sampling_params={})
    assert res.status == "completed"
    assert res.rewards == {"reward": 2.0}


@pytest.mark.asyncio
async def test_harbor_runner_exception_is_error(monkeypatch, tmp_path):
    exc = types.SimpleNamespace(exception_type="RuntimeError", exception_message="boom", traceback="tb")
    _install_fake_harbor(monkeypatch, exception_info=exc)
    from remote_agent.runner.base import AgentTask
    from remote_agent.runner.harbor.runner import HarborRunner

    runner = HarborRunner({"agent_name": "swe-agent"})
    res = await runner.run(
        task=AgentTask(instance_id="i", task_path=str(tmp_path), row={}),
        agent_base_url="http://h:9/i/v1",
        sampling_params={},
    )
    assert res.status == "error"
    assert "RuntimeError: boom" in res.error
