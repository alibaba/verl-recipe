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
from omegaconf import OmegaConf

from remote_agent.agent_loop.config import RemoteAgentCoreConfig


def test_from_dictconfig_defaults():
    cfg = OmegaConf.create({"runner": {"name": "example", "kwargs": {}}})
    core = RemoteAgentCoreConfig.from_dictconfig(cfg)
    assert core.proxy_port == 0
    assert core.runner_name == "example"
    assert core.max_retries == 3


def test_advertised_host_env_override(monkeypatch):
    monkeypatch.setenv("REMOTE_AGENT_ADVERTISED_HOST", "1.2.3.4")
    cfg = OmegaConf.create({"proxy": {"advertised_host": "0.0.0.0"}, "runner": {"name": "example"}})
    core = RemoteAgentCoreConfig.from_dictconfig(cfg)
    assert core.advertised_host == "1.2.3.4"
