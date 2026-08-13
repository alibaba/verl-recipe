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
"""HTTP client to the external SGLang cluster (deployed as a RoleBasedGroup).

This client only drives SGLang's *weight-update* endpoints — the ones the
weight-sync transports call. Text *generation* during rollout is NOT routed
here: it goes through verl's normal rollout path via
``actor_rollout_ref.rollout.gateway_url`` (the SGLang RBG router /
OpenAI-compatible endpoint), reusing ``GatewayLLMServerClient``. Keeping the two
concerns separate mirrors ``recipe/remote_megatron_sglang/checkpoint_engine.py``.

Endpoints used (SGLang HTTP server; see SGLang docs — marked ``# VALIDATE``
against your SGLang version):

* ``/init_weights_update_group``       — form the cross-cluster NCCL group
* ``/update_weights_from_distributed`` — recv broadcast weights, load_weights()
* ``/update_weights_from_disk``        — reload weights from a shared store
* ``/flush_cache``                     — drop KV cache after a weight update
"""

from __future__ import annotations

from typing import Any


class SGLangInferClient:
    def __init__(self, endpoints: list[str], timeout: float = 1800.0):
        # One base URL per SGLang server (the RBG leader, or each TP leader).
        # Weight-update calls fan out to all of them.
        if isinstance(endpoints, str):
            endpoints = [endpoints]
        self.endpoints = [e.rstrip("/") for e in endpoints]
        self.timeout = timeout

    def _client(self):
        import httpx

        return httpx.AsyncClient(timeout=self.timeout)

    async def _post_all(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to every SGLang endpoint (weight ops must hit all TP servers)."""
        import asyncio

        async def _one(base):
            async with self._client() as c:
                resp = await c.post(f"{base}{path}", json=payload)
                resp.raise_for_status()
                return resp.json() if resp.content else {}

        results = await asyncio.gather(*[_one(b) for b in self.endpoints], return_exceptions=True)
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            raise errors[0]
        return {"sglang/endpoints": len(self.endpoints)}

    async def init_weights_update_group(
        self, master_address: str, master_port: int, world_size: int, backend: str = "nccl"
    ) -> dict[str, Any]:
        # rank_offset lets SGLang place its TP workers after the Megatron ranks
        # in the shared group. # VALIDATE payload keys against your SGLang build.
        return await self._post_all(
            "/init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
                "backend": backend,
            },
        )

    async def update_weights_from_distributed(self) -> dict[str, Any]:
        return await self._post_all("/update_weights_from_distributed", {})

    async def update_weights_from_disk(self, path: str) -> dict[str, Any]:
        return await self._post_all("/update_weights_from_disk", {"model_path": path})

    async def update_weights_from_mooncake(self, session_id: str) -> dict[str, Any]:
        return await self._post_all("/update_weights_from_mooncake", {"session_id": session_id})

    async def flush_cache(self) -> dict[str, Any]:
        return await self._post_all("/flush_cache", {})

    async def destroy_weights_update_group(self) -> dict[str, Any]:
        return await self._post_all("/destroy_weights_update_group", {})

    async def init_mooncake(self, metadata_server: str, session_id: str) -> dict[str, Any]:
        return await self._post_all(
            "/init_mooncake", {"metadata_server": metadata_server, "session_id": session_id}
        )

    async def destroy_mooncake(self, session_id: str) -> dict[str, Any]:
        return await self._post_all("/destroy_mooncake", {"session_id": session_id})
