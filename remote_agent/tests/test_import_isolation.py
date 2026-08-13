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
import subprocess
import sys


def test_core_import_does_not_pull_harbor():
    # Fresh interpreter: importing the core must not import harbor.
    code = (
        "import importlib, sys;"
        "importlib.import_module('remote_agent.agent_loop.remote_agent_loop');"
        "importlib.import_module('remote_agent.runner.example_runner');"
        "assert 'harbor' not in sys.modules, sorted(m for m in sys.modules if 'harbor' in m);"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
