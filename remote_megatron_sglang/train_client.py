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
"""HTTP client from the verl forwarder to the Megatron ``train_server`` (rank 0)
running inside the PyTorchJob. Implements the contract in ``server/protocol.py``.

Compute calls (``compute_log_prob`` / ``update_actor``) are synchronous: the verl
forwarder's ``@register`` methods for these are sync and are already dispatched
concurrently across the mesh by the single-controller layer. Weight-sync and
checkpoint calls are async so the transport can overlap the Megatron push with
the SGLang receive.
"""

from __future__ import annotations

from typing import Any

from recipe.remote_megatron_sglang.server import protocol as P


class MegatronTrainClient:
    def __init__(self, endpoint: str, timeout: float = 3600.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    # ---- lazy clients (httpx imported lazily so this module compiles/imports
    # ---- on a driver that hasn't installed httpx yet) --------------------- #

    def _sync_client(self):
        import httpx

        return httpx.Client(base_url=self.endpoint, timeout=self.timeout)

    def _async_client(self):
        import httpx

        return httpx.AsyncClient(base_url=self.endpoint, timeout=self.timeout)

    # ---- compute (sync) --------------------------------------------------- #

    def _post_td_sync(self, path: str, td):
        # Frame tensors + metadata into a single body (metadata in a header
        # overflows aiohttp's max header-line length for real verl batches).
        blob = P.frame_tensordict(td)
        with self._sync_client() as c:
            resp = c.post(path, content=blob)
            resp.raise_for_status()
            return P.unframe_tensordict(resp.content)

    def compute_log_prob(self, td):
        """Actor forward (no grad) → TensorDict with ``old_log_probs`` etc."""
        return self._post_td_sync(P.COMPUTE_LOG_PROB, td)

    def compute_ref_log_prob(self, td):
        """Ref-model forward (no grad) → TensorDict with ``ref_log_prob``."""
        return self._post_td_sync(P.COMPUTE_REF_LOG_PROB, td)

    def update_actor(self, td):
        """Forward + backward + optimizer step. Returns a TensorDict whose
        ``meta_info`` carries the training metrics."""
        return self._post_td_sync(P.UPDATE_ACTOR, td)

    # ---- weight sync + checkpoint (async) --------------------------------- #

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._async_client() as c:
            resp = await c.post(path, json=payload)
            resp.raise_for_status()
            return resp.json() if resp.content else {}

    async def push_weights(self, transport: str, step: int, **kwargs) -> dict[str, Any]:
        return await self._post_json(P.PUSH_WEIGHTS, {"transport": transport, "step": step, **kwargs})

    async def save_weights(self, path: str) -> dict[str, Any]:
        return await self._post_json(P.SAVE_WEIGHTS, {"path": path})

    async def save_checkpoint(self, path: str, step: int) -> dict[str, Any]:
        return await self._post_json(P.SAVE_CHECKPOINT, {"path": path, "step": step})

    async def load_checkpoint(self, path: str) -> dict[str, Any]:
        return await self._post_json(P.LOAD_CHECKPOINT, {"path": path})

    async def init_weight_broadcast_group(self, **kwargs) -> dict[str, Any]:
        return await self._post_json(P.INIT_BROADCAST_GROUP, kwargs)

    async def destroy_weight_broadcast_group(self) -> dict[str, Any]:
        return await self._post_json(P.DESTROY_BROADCAST_GROUP, {})

    async def init_mooncake(self, **kwargs) -> dict[str, Any]:
        return await self._post_json(P.INIT_MOONCAKE, kwargs)

    async def destroy_mooncake(self, **kwargs) -> dict[str, Any]:
        return await self._post_json(P.DESTROY_MOONCAKE, kwargs)

    async def health(self) -> bool:
        async with self._async_client() as c:
            resp = await c.get(P.HEALTH)
            return resp.status_code == 200
