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
"""``MegatronSGLangRolloutReplica`` — routes verl rollout to an *externally*
deployed SGLang cluster (RoleBasedGroup), following the PR's ``ArcticReplica``
pattern (``verl/workers/rollout/remote_rollout/arctic_rollout``).

Unlike Arctic (whose ``generate`` goes through the *same* training backend), our
SGLang is a separate external service, so generation is fully decoupled from the
Megatron training backend: the replica just forwards token-in-token-out
``generate`` to the external SGLang HTTP ``/generate`` endpoint. No GPU is
colocated on the verl side (``rollout_worker_use_gpu() -> False``).

Wiring (mirrors the built-in SGLang replica, minus the local launch):
* ``AgentLoopManager``/``LLMServerClient`` → load balancer → ``server_handle.generate``
* ``server_handle`` is a CPU Ray actor (:class:`ExternalSGLangProxyServer`) whose
  ``generate`` POSTs to the external SGLang and maps the response to the exact
  :class:`~verl.workers.rollout.replica.TokenOutput` the rollout loop expects.

Select it with ``actor_rollout_ref.rollout.name=megatron_sglang``. The external
endpoints come from the ``MEGATRON_SGLANG_ENDPOINTS`` env var (comma-separated
base URLs), set on the driver — kept out of RolloutConfig to avoid touching the
core dataclass schema.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from urllib.parse import urlparse

import ray

from verl.workers.rollout.replica import RolloutReplica, RolloutReplicaRegistry, TokenOutput

logger = logging.getLogger(__name__)

_ENDPOINTS_ENV = "MEGATRON_SGLANG_ENDPOINTS"


def _external_endpoints() -> list[str]:
    raw = os.environ.get(_ENDPOINTS_ENV, "").strip()
    if not raw:
        raise ValueError(
            f"{_ENDPOINTS_ENV} is unset. Set it to the external SGLang base URL(s), "
            "comma-separated, e.g. 'http://sglang-rbg-leader:30000'."
        )
    return [e.strip().rstrip("/") for e in raw.split(",") if e.strip()]


class ExternalSGLangProxyServer:
    """CPU-only server handle that forwards token-in-token-out ``generate`` to an
    externally-deployed SGLang HTTP server. Ray-wrapped by the replica.

    Mirrors the request build + response→TokenOutput mapping of verl's
    ``SGLangHttpServer.generate`` (async_sglang_server.py), but over HTTP to a
    server we do not own, so lifecycle hooks (wake_up/sleep/kv-cache) are no-ops
    except ``clear_kv_cache`` which maps to SGLang ``/flush_cache``.
    """

    def __init__(self, endpoint: str, response_length: int, max_model_len: int, timeout: float = 600.0):
        self.endpoint = endpoint.rstrip("/")
        self.response_length = int(response_length)
        self.max_model_len = int(max_model_len)
        self.timeout = timeout
        self.global_steps = 0

    def _client(self):
        import httpx

        return httpx.AsyncClient(timeout=self.timeout)

    async def generate(
        self,
        prompt_ids,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        bootstrap_host: Optional[str] = None,
        bootstrap_port: Optional[int] = None,
        bootstrap_room: Optional[int] = None,
    ) -> TokenOutput:
        # Normalize prompt_ids to a flat list[int] (mirror SGLangHttpServer).
        if hasattr(prompt_ids, "keys") and "input_ids" in prompt_ids:
            prompt_ids = list(prompt_ids["input_ids"])
        elif hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        elif not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)
        if prompt_ids and not isinstance(prompt_ids[0], int):
            prompt_ids = [int(x) for x in prompt_ids]

        sampling_params = dict(sampling_params)
        # max_new_tokens resolution (mirror SGLangHttpServer).
        if "max_new_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_new_tokens")
        elif "max_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_tokens")
        else:
            max_new_tokens = self.response_length
        max_possible = self.max_model_len - len(prompt_ids) - 1
        max_new_tokens = max(0, min(max_new_tokens, max_possible))
        sampling_params["max_new_tokens"] = max_new_tokens

        return_logprob = bool(sampling_params.pop("logprobs", False))
        request = {
            "rid": request_id,
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
        }
        async with self._client() as c:
            resp = await c.post(f"{self.endpoint}/generate", json=request)
            resp.raise_for_status()
            output = resp.json()

        meta_info = output.get("meta_info", {}) or {}
        finish_reason = meta_info.get("finish_reason")
        finish_reason = finish_reason["type"] if isinstance(finish_reason, dict) else finish_reason

        token_ids = list(output.get("output_ids", []) or [])
        log_probs = None
        if return_logprob:
            otl = meta_info.get("output_token_logprobs") or []
            if otl and len(otl) == len(token_ids):
                log_probs = [float(lp) for lp, _tok, *_ in otl]
            else:
                if len(otl) != len(token_ids):
                    logger.error(
                        "output_token_logprobs len %d != output_ids len %d (rid=%s)",
                        len(otl),
                        len(token_ids),
                        request_id,
                    )
                token_ids, log_probs = [], []

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=finish_reason,
            extra_fields={"global_steps": self.global_steps},
        )

    # ---- address + lifecycle hooks the replica / load balancer call --------- #

    async def get_server_address(self):
        u = urlparse(self.endpoint)
        return u.hostname, (u.port or 80)

    async def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps

    async def clear_kv_cache(self):
        # The external SGLang owns its KV cache; flush after a weight update.
        try:
            async with self._client() as c:
                await c.post(f"{self.endpoint}/flush_cache", json={})
        except Exception as e:  # best-effort
            logger.warning("flush_cache on external SGLang failed: %s", e)

    async def wake_up(self, *args, **kwargs):
        return None

    async def sleep(self, *args, **kwargs):
        return None

    async def abort_all_requests(self):
        return None

    async def resume_generation(self):
        return None

    async def release_kv_cache(self):
        return None

    async def resume_kv_cache(self):
        return None

    async def start_profile(self, **kwargs):
        return None

    async def stop_profile(self):
        return None


class MegatronSGLangRolloutReplica(RolloutReplica):
    """RolloutReplica that points at an external SGLang cluster instead of
    launching one. Selected via ``rollout.name=megatron_sglang``."""

    def rollout_worker_use_gpu(self) -> bool:
        return False  # generation lives in the external SGLang; verl side is CPU-only

    async def launch_servers(self):
        """Create a CPU proxy actor per external endpoint (no local engine)."""
        endpoints = _external_endpoints()
        # One proxy per replica_rank, round-robin over available endpoints.
        endpoint = endpoints[self.replica_rank % len(endpoints)]
        response_length = getattr(self.config, "response_length", 1024)
        max_model_len = getattr(self.config, "max_model_len", None) or (
            getattr(self.config, "prompt_length", 1024) + response_length
        )
        server_cls = ray.remote(ExternalSGLangProxyServer)
        server = server_cls.options(
            name=f"megatron_sglang_proxy_{self.replica_rank}{self.name_suffix}",
            num_cpus=1,
            max_concurrency=self.max_concurrency,
        ).remote(endpoint=endpoint, response_length=response_length, max_model_len=max_model_len)
        self.servers = [server]
        self._server_handle = server
        host, port = await server.get_server_address.remote()
        self._server_address = f"{host}:{port}"
        logger.info("MegatronSGLangRolloutReplica %d → external SGLang %s", self.replica_rank, endpoint)


def _load_megatron_sglang():
    return MegatronSGLangRolloutReplica


# Register so `rollout.name=megatron_sglang` resolves. Imported by the backend
# module (which main_ppo imports when trainer.remote_backend=megatron_sglang).
RolloutReplicaRegistry.register("megatron_sglang", _load_megatron_sglang)
