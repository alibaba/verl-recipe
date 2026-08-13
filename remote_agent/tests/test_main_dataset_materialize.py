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

from omegaconf import OmegaConf

from remote_agent.main import materialize_dataset


def test_materialize_via_runner_hook(monkeypatch):
    from remote_agent.runner import base

    @base.register_runner("ds_stub")
    class _Stub(base.ExternalAgentRunner):
        async def run(self, **k): ...

        def build_dataset(self, data_cfg):
            return ("/tmp/train.parquet", "/tmp/val.parquet")

    cfg = OmegaConf.create(
        {
            "data": {"train_files": None, "val_files": None, "train_harbor_dir": "/x"},
            "actor_rollout_ref": {"rollout": {"remote_agent": {"runner": {"name": "ds_stub"}}}},
        }
    )
    materialize_dataset(cfg)
    assert cfg.data.train_files == "/tmp/train.parquet"
    assert cfg.data.val_files == "/tmp/val.parquet"
    assert "harbor" not in sys.modules
