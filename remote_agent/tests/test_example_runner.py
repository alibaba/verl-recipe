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

import pytest

from remote_agent.runner import base
from remote_agent.runner.example_runner import ExampleRunner


@pytest.mark.asyncio
async def test_example_runner_runs_command_with_base_url(tmp_path):
    marker = tmp_path / "seen_url.txt"
    cmd = [sys.executable, "-c", f"import os;open(r'{marker}','w').write(os.environ['OPENAI_BASE_URL'])"]
    runner = ExampleRunner({"command": cmd})
    task = base.AgentTask(instance_id="i", task_path=None, row={})
    res = await runner.run(task=task, agent_base_url="http://h:9/i/v1", sampling_params={})
    assert res.status == "completed"
    assert marker.read_text() == "http://h:9/i/v1"


@pytest.mark.asyncio
async def test_example_runner_nonzero_exit_is_error():
    runner = ExampleRunner({"command": [sys.executable, "-c", "import sys;sys.exit(3)"]})
    task = base.AgentTask(instance_id="i", task_path=None, row={})
    res = await runner.run(task=task, agent_base_url="http://h:9/i/v1", sampling_params={})
    assert res.status == "error"
    assert "exit code 3" in (res.error or "")
