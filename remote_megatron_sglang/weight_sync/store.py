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
"""``store`` weight-sync transport.

Megatron writes an HF-format checkpoint to a shared store (HDFS/S3/PVC), then
SGLang reloads it via ``/update_weights_from_disk``. The slowest transport (full
serialize + reload each step) but requires no cross-cluster NCCL/RDMA
reachability — the reliable fallback when the two clusters can't form a
collective group.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from recipe.remote_megatron_sglang.weight_sync.base import WeightSyncRegistry, WeightSyncTransport

logger = logging.getLogger(__name__)


@WeightSyncRegistry.register("store")
class CheckpointStoreTransport(WeightSyncTransport):
    def __init__(self, cfg, train_client, infer_client):
        self.train_client = train_client
        self.infer_client = infer_client
        # Base URI both clusters can read/write, e.g. hdfs://.../weight_sync or
        # /mnt/shared/weight_sync (a shared PVC). # VALIDATE mount on both pods.
        self.base_uri = cfg.get("store_uri")
        if not self.base_uri:
            raise ValueError("weight_sync.store_uri is required for the 'store' transport.")

    @classmethod
    def from_config(cls, cfg, train_client, infer_client) -> CheckpointStoreTransport:
        return cls(cfg, train_client, infer_client)

    async def setup(self) -> None:  # nothing to prepare
        return

    async def sync(self, step: int) -> dict[str, Any]:
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        path = os.path.join(self.base_uri, f"step_{step}")
        # 1. Megatron exports HF-format weights to the shared store.
        save = await self.train_client.save_weights(path=path)
        # 2. SGLang reloads them from the same path, then flushes KV cache.
        recv = await self.infer_client.update_weights_from_disk(path=path)
        await self.infer_client.flush_cache()
        dt = loop.time() - t0
        return {"weight_sync/transport": "store", "weight_sync/seconds": dt, "weight_sync/path": path, **save, **recv}

    async def teardown(self) -> None:
        return
