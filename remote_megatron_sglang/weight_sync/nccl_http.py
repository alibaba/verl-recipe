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
"""``nccl_http`` weight-sync transport.

Forms a single NCCL process group that spans the Megatron training ranks and
the SGLang TP workers, then each step: Megatron rank 0 ``broadcast``s the
unsharded HF weights while SGLang receives them via
``/update_weights_from_distributed`` (SGLang's ``model.load_weights()`` re-shards
per TP rank and handles QKV / gate-up fusion — no SGLang patch needed). This is
the same transport proven in ``recipe/remote_megatron_sglang/checkpoint_engine.py``, generalized to a
Megatron weight source.

verl only orchestrates the handshake; the tensor bytes travel Megatron→SGLang
directly and never pass through the verl driver.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from recipe.remote_megatron_sglang.weight_sync.base import WeightSyncRegistry, WeightSyncTransport

logger = logging.getLogger(__name__)


@WeightSyncRegistry.register("nccl_http")
class NcclHttpTransport(WeightSyncTransport):
    """Trigger NCCL-over-HTTP weight sync. The heavy lifting (forming one NCCL
    group spanning the Megatron rank(s) + SGLang TP workers, then broadcasting
    HF-layout weights) runs inside the train_server, which is the only side that
    holds the GPU weights and can join the NCCL group. This driver-side transport
    just tells the train_server to sync, passing the SGLang endpoints so it can
    coordinate the group. Intra-node NCCL runs over TCP — no RDMA needed.
    """

    def __init__(self, cfg, train_client, infer_client):
        self.train_client = train_client
        self.infer_client = infer_client
        self.group_name = cfg.get("group_name", "verl_ms_weight_sync")
        self.chunk_tensors = int(cfg.get("chunk_tensors", 32))

    @classmethod
    def from_config(cls, cfg, train_client, infer_client) -> "NcclHttpTransport":
        return cls(cfg, train_client, infer_client)

    async def setup(self) -> None:
        return  # group is formed lazily inside the train_server on first sync

    async def sync(self, step: int) -> dict[str, Any]:
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        res = await self.train_client.push_weights(
            transport="nccl",
            step=step,
            sglang_endpoints=self.infer_client.endpoints,
            group_name=self.group_name,
            chunk_tensors=self.chunk_tensors,
        )
        return {"weight_sync/transport": "nccl_http", "weight_sync/seconds": loop.time() - t0, **res}

    async def teardown(self) -> None:
        return
