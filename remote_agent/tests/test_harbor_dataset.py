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
from pathlib import Path

from remote_agent.runner.harbor.dataset import build_verl_parquets


def _make_task(root: Path, name: str):
    d = root / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text("[metadata]\nkey='v'\n")
    (d / "instruction.md").write_text(f"do {name}")


def test_build_parquets(tmp_path):
    train = tmp_path / "train"
    val = tmp_path / "val"
    _make_task(train, "t1")
    _make_task(val, "v1")
    tp, vp = build_verl_parquets(str(train), str(val), str(tmp_path / "cache"))
    import pandas as pd

    df = pd.read_parquet(tp)
    assert df.iloc[0]["instance_id"] == "t1"
    assert df.iloc[0]["data_source"] == "harbor"

    vdf = pd.read_parquet(vp)
    assert vdf.iloc[0]["instance_id"] == "v1"
    assert vdf.iloc[0]["data_source"] == "harbor"
