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
import pytest

from remote_agent.runner import base


def test_register_and_create_runner():
    @base.register_runner("dummy")
    class Dummy(base.ExternalAgentRunner):
        async def run(self, *, task, agent_base_url, sampling_params, **kwargs):
            return base.AgentRunResult(status="completed", rewards={"r": 1.0})

    runner = base.create_runner("dummy", {})
    assert isinstance(runner, Dummy)


def test_unknown_runner_raises():
    with pytest.raises(KeyError, match="unknown remote-agent runner"):
        base.create_runner("does-not-exist", {})


def test_agent_task_and_result_shapes():
    t = base.AgentTask(instance_id="i1", task_path="/x", row={"a": 1})
    assert t.instance_id == "i1" and t.row["a"] == 1
    r = base.AgentRunResult(status="error", rewards={}, error="boom")
    assert r.status == "error" and r.error == "boom"
