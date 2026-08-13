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
"""``mooncake`` weight-sync transport (RDMA point-to-point).

Uses the Mooncake transfer engine for zero-copy RDMA transfer of weights from
Megatron to SGLang. Fastest at scale, most infra setup (both pods need RDMA/HCA
devices). This mirrors the Mooncake path validated in prior external-SGLang
work; the wire details are cluster-specific and marked ``# VALIDATE``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from recipe.remote_megatron_sglang.weight_sync.base import WeightSyncRegistry, WeightSyncTransport

logger = logging.getLogger(__name__)


@WeightSyncRegistry.register("mooncake")
class MooncakeTransport(WeightSyncTransport):
    def __init__(self, cfg, train_client, infer_client):
        self.train_client = train_client
        self.infer_client = infer_client
        # Mooncake metadata/rendezvous server both clusters connect to.
        self.metadata_server = cfg.get("mooncake_metadata_server")
        self.session_id = cfg.get("mooncake_session_id", "verl-remote-megatron-sglang")
        self._ready = False

    @classmethod
    def from_config(cls, cfg, train_client, infer_client) -> MooncakeTransport:
        return cls(cfg, train_client, infer_client)

    async def setup(self) -> None:
        if self._ready:
            return
        # Register both endpoints with the Mooncake session. # VALIDATE: the
        # exact registration handshake depends on your Mooncake + SGLang build.
        await asyncio.gather(
            self.train_client.init_mooncake(metadata_server=self.metadata_server, session_id=self.session_id),
            self.infer_client.init_mooncake(metadata_server=self.metadata_server, session_id=self.session_id),
        )
        self._ready = True
        logger.info("mooncake weight-sync session ready (session=%s)", self.session_id)

    async def sync(self, step: int) -> dict[str, Any]:
        await self.setup()
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        push, recv = await asyncio.gather(
            self.train_client.push_weights(transport="mooncake", step=step, session_id=self.session_id),
            self.infer_client.update_weights_from_mooncake(session_id=self.session_id),
        )
        await self.infer_client.flush_cache()
        dt = loop.time() - t0
        return {"weight_sync/transport": "mooncake", "weight_sync/seconds": dt, **push, **recv}

    async def teardown(self) -> None:
        if not self._ready:
            return
        await asyncio.gather(
            self.train_client.destroy_mooncake(session_id=self.session_id),
            self.infer_client.destroy_mooncake(session_id=self.session_id),
            return_exceptions=True,
        )
        self._ready = False
