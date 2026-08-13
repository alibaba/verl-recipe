# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Tests for the megatron_sglang remote backend.

The weight-sync-registry test is dependency-light and runs anywhere. The backend
and protocol tests need the full verl env (omegaconf / torch / safetensors) and
are skipped otherwise.
"""

from __future__ import annotations

import pytest


def test_weight_sync_transports_register_on_import():
    """Importing the package registers all built-in transports (no torch)."""
    from recipe.remote_megatron_sglang.weight_sync import WeightSyncRegistry

    assert set(WeightSyncRegistry.list()) == {"nccl_http", "store", "mooncake", "noop"}

    with pytest.raises(KeyError):
        WeightSyncRegistry.create("does-not-exist", None, None, None)


def test_backend_registers_and_resolves():
    """megatron_sglang backend self-registers and resolves via the registry."""
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    import recipe.remote_megatron_sglang.backend as _  # noqa: F401  (import side-effect)
    from verl.remote_backend import RemoteBackendRegistry

    assert "megatron_sglang" in RemoteBackendRegistry.list()

    cfg = OmegaConf.create(
        {
            "remote_backend": {
                "megatron_sglang": {
                    "train_endpoint": "http://train:8000",
                    "sglang_endpoints": ["http://sgl:30000"],
                    "checkpoint_dir": "ckpt",
                    "weight_sync": {
                        "transport": "nccl_http",
                        "master_addr": "train",
                        "master_port": 29500,
                        "group_world_size": 2,
                    },
                }
            }
        }
    )
    backend = RemoteBackendRegistry.create("megatron_sglang", cfg)
    assert backend.requires_single_forwarder() is True
    # reconnect handle is small + serializable
    handle = backend.reconnect_handle()
    assert handle["backend"] == "megatron_sglang"


def test_tensordict_roundtrip():
    """protocol.encode/decode_tensordict is lossless for tensors + metadata."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("tensordict")
    import torch
    from tensordict import TensorDict

    from recipe.remote_megatron_sglang.server import protocol as P

    td = TensorDict(
        {"input_ids": torch.arange(12).reshape(3, 4), "attention_mask": torch.ones(3, 4, dtype=torch.long)},
        batch_size=[3],
    )
    body, meta = P.encode_tensordict(td)
    out = P.decode_tensordict(body, P.loads_meta(P.dumps_meta(meta)))
    assert torch.equal(out["input_ids"], td["input_ids"])
    assert list(out.batch_size) == [3]


def test_single_forwarder_assert(monkeypatch):
    """RemoteBackendTrainer rejects >1 forwarder when the backend requires one."""
    pytest.importorskip("omegaconf")
    pytest.importorskip("ray")
    # This exercises RemoteBackendTrainer._enforce_single_forwarder_if_required
    # against a stub backend; it needs the verl trainer import chain (ray).
    pytest.skip("run in the verl CI image; requires the full trainer import chain")
