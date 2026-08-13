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
"""``noop`` weight-sync transport.

Does not transfer weights. Useful for pipeline bring-up / debugging where you
want to validate the generate → log-prob → advantage → update_actor loop
end-to-end without also standing up a cross-cluster NCCL group or shared store.
The rollout engine keeps serving its initial weights, so training is slightly
off-policy — do NOT use for real training, only for pipeline validation.
"""

from __future__ import annotations

import logging
from typing import Any

from recipe.remote_megatron_sglang.weight_sync.base import WeightSyncRegistry, WeightSyncTransport

logger = logging.getLogger(__name__)


@WeightSyncRegistry.register("noop")
class NoopTransport(WeightSyncTransport):
    def __init__(self, cfg, train_client, infer_client):
        self._warned = False

    @classmethod
    def from_config(cls, cfg, train_client, infer_client) -> "NoopTransport":
        return cls(cfg, train_client, infer_client)

    async def setup(self) -> None:
        return

    async def sync(self, step: int) -> dict[str, Any]:
        if not self._warned:
            logger.warning("weight_sync.transport=noop: NOT syncing weights (pipeline-validation only).")
            self._warned = True
        return {"weight_sync/transport": "noop", "weight_sync/skipped": True}

    async def teardown(self) -> None:
        return
