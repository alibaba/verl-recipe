# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Weight sync to an EXTERNALLY-DEPLOYED SGLang service (not launched by verl / not
in verl's Ray cluster).

This is "Plan A" from
``recipe/agentic/docs/external-sglang-weight-sync-analysis.md``: reuse SGLang's
existing HTTP weight-update API and push full HF-format weights over a NCCL
process group that the trainer (rank 0) and the SGLang TP workers jointly form.

    Trainer FSDP rank 0  ──torch.distributed.broadcast (NCCL)──►  SGLang TP workers
           │  POST /init_weights_update_group        (form group)
           │  POST /update_weights_from_distributed  (recv + model.load_weights)
           │  POST /flush_cache
           ▼

Nothing on the SGLang side needs patching — all of these endpoints already
exist. verl outputs full unsharded HF weights via ``get_per_tensor_param()``;
SGLang's ``update_weights_from_distributed`` calls ``model.load_weights()`` which
re-shards per TP rank and handles QKV / gate-up fusion.

STATUS: VALIDATED (2026-07-08, standalone). The group-formation fix (SGLang's
init_custom_process_group instead of torch.distributed.new_group) makes the path
work end-to-end: a trainer rank-0 Ray actor formed the NCCL group with an
out-of-Ray SGLang (Qwen2.5-3B, TP=2, world_size=3) and broadcast full HF weights
over RoCE — zeroing model.embed_tokens.weight collapsed generation to "!!!!" and
restoring recovered it. Group build ~0.01-0.02s, broadcast of 5.75 GiB ~0.16-1.08s.
See recipe/remote_megatron_sglang/test/external_sglang/validate_weight_sync_nccl_http.py. Still
to wire into a full training run via ExternalSGLangCheckpointManager (below).

Wiring (no core verl edits needed):

    actor_rollout_ref.rollout.checkpoint_engine.backend=external_sglang_nccl
    actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=recipe.remote_megatron_sglang.checkpoint_engine
    actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.external_sglang_nccl.sglang_endpoints=["http://sgl-0:30000","http://sgl-1:30000"]
    actor_rollout_ref.rollout.checkpoint_manager_class=recipe.remote_megatron_sglang.checkpoint_engine.ExternalSGLangCheckpointManager
"""

import asyncio
import logging
import os
from typing import Any, Generator

import requests
import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineManager, CheckpointEngineRegistry
from verl.utils.net_utils import get_free_port
from verl.utils.ray_utils import auto_await

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Small HTTP helpers around SGLang's weight-update endpoints.
# ---------------------------------------------------------------------------
def _post(endpoint: str, path: str, payload: dict, timeout: float = 300.0) -> dict:
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


def _get(endpoint: str, path: str, timeout: float = 60.0) -> dict:
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _sglang_tp_size(endpoint: str) -> int:
    """Query an SGLang instance for its tensor-parallel size.

    # VALIDATE: field name is version dependent. Recent SGLang exposes
    # ``/get_server_info`` with ``tp_size``; older builds use ``/get_model_info``.
    """
    info = _get(endpoint, "get_server_info")
    for key in ("tp_size", "tensor_parallel_size"):
        if key in info:
            return int(info[key])
    # nested under "server_args" on some versions
    args = info.get("server_args", {})
    if "tp_size" in args:
        return int(args["tp_size"])
    raise RuntimeError(f"could not determine tp_size from {endpoint}/get_server_info: {info}")


_DTYPE_TO_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float8_e4m3fn: "float8_e4m3fn",
}


@CheckpointEngineRegistry.register("external_sglang_nccl")
class ExternalSGLangNCCLEngine(CheckpointEngine):
    """Trainer-side checkpoint engine that pushes weights to external SGLang.

    Runs inside each actor Ray worker (``ActorRolloutRefWorker``). Only the
    ``is_master`` (global rank 0) instance talks to SGLang and broadcasts; the
    other ranks still iterate the weight generator so their FSDP
    ``full_tensor()`` all-gathers do not deadlock, then discard.

    Args (from ``checkpoint_engine.engine_kwargs.external_sglang_nccl``):
        bucket_size: injected by verl (unused here; SGLang recv is per-tensor-list).
        sglang_endpoints: list of base URLs of the external SGLang HTTP servers.
        group_name: NCCL group name shared with SGLang. Default "verl_weight_sync".
        chunk_tensors: how many tensors to announce+broadcast per HTTP round trip.
        is_master: injected by verl (rank 0).
    """

    def __init__(
        self,
        bucket_size: int,
        sglang_endpoints: list[str] | None = None,
        group_name: str = "verl_weight_sync",
        chunk_tensors: int = 64,
        is_master: bool = False,
        **kwargs: Any,
    ) -> None:
        self.bucket_size = bucket_size
        self.sglang_endpoints = list(sglang_endpoints or [])
        self.group_name = group_name
        self.chunk_tensors = chunk_tensors
        self.is_master = is_master

        self._group = None  # torch.distributed process group handle
        self._world_size = None
        self._endpoint_rank_offset: dict[str, int] = {}
        if self.is_master and not self.sglang_endpoints:
            raise ValueError("external_sglang_nccl requires engine_kwargs.sglang_endpoints on the master")

    # --- the abstract methods verl's manager may call. We self-drive the group
    #     inside send_weights, so most are no-ops for the external case. ---
    def prepare(self) -> dict[str, Any]:
        return {}

    @classmethod
    def build_topology(cls, trainer_world_size, rollout_world_size, metadata):
        # Not used: external SGLang is not a verl rollout worker group, so the
        # manager below never calls build_process_group().
        return {}, {}

    def init_process_group(self, **kwargs):
        return None

    def finalize(self):
        # PERSISTENT group: keep it across steps (form once, broadcast every
        # step). This avoids the ~0.7s per-step NCCL teardown AND the stale-group
        # 400 that a per-step rebuild hits. The group is torn down only when the
        # process exits; a fresh run cleans any leftover via _build_group_if_needed's
        # destroy-first below. SGLang keeps its side of the group alive too, so
        # subsequent /update_weights_from_distributed reuse it.
        return

    async def receive_weights(self, global_steps: int | None = None):
        # No verl-side receiver exists for external SGLang.
        raise NotImplementedError("external_sglang_nccl has no verl-side receiver; SGLang receives via HTTP")
        yield  # pragma: no cover  (make this an async generator)

    # --- the real work ---
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        # Non-master ranks MUST still drain the generator so FSDP all-gathers
        # inside get_per_tensor_param() complete on every rank.
        if not self.is_master:
            for _ in weights:
                pass
            return

        self._build_group_if_needed()

        loop = asyncio.get_running_loop()
        chunk: list[tuple[str, torch.Tensor]] = []

        async def flush(batch: list[tuple[str, torch.Tensor]]):
            if not batch:
                return
            names = [n for n, _ in batch]
            dtypes = [_DTYPE_TO_STR[t.dtype] for _, t in batch]
            shapes = [list(t.shape) for _, t in batch]

            # 1. tell every SGLang instance to start receiving this batch. These
            #    calls block inside SGLang on the NCCL recv, so run them in
            #    threads while rank 0 broadcasts.
            http_tasks = [
                loop.run_in_executor(
                    None,
                    _post,
                    ep,
                    "update_weights_from_distributed",
                    {
                        "names": names,
                        "dtypes": dtypes,
                        "shapes": shapes,
                        "group_name": self.group_name,
                        "flush_cache": False,
                    },
                )
                for ep in self.sglang_endpoints
            ]

            # 2. broadcast each tensor from src=0 in the same order. cuda tensors
            #    required for the NCCL backend.
            def _broadcast_all():
                for _, tensor in batch:
                    torch.distributed.broadcast(tensor.detach().cuda(), src=0, group=self._group)

            await loop.run_in_executor(None, _broadcast_all)
            await asyncio.gather(*http_tasks)

        for name, tensor in weights:
            chunk.append((name, tensor))
            if len(chunk) >= self.chunk_tensors:
                await flush(chunk)
                chunk = []
        await flush(chunk)

        # flush KV / prefix cache once, after all weights are in place.
        for ep in self.sglang_endpoints:
            _post(ep, "flush_cache", {})
        logger.info(f"external SGLang weight sync done for global_steps={global_steps}")

    def _build_group_if_needed(self):
        if self._group is not None:
            return

        master_addr = os.environ.get("MASTER_ADDR") or _local_ip()
        master_port = get_free_port(master_addr)[0]

        # rank layout: trainer rank 0 == NCCL rank 0, then each SGLang instance's
        # TP workers occupy a contiguous block.
        offset = 1
        self._endpoint_rank_offset = {}
        for ep in self.sglang_endpoints:
            self._endpoint_rank_offset[ep] = offset
            offset += _sglang_tp_size(ep)
        world_size = offset
        self._world_size = world_size

        # Clean up any stale group left by a previous (possibly killed) run —
        # otherwise SGLang's init_weights_update_group returns 400 "group already
        # exists" and the trainer worker crashes. Best-effort; ignore errors.
        for ep in self.sglang_endpoints:
            try:
                _post(ep, "destroy_weights_update_group", {"group_name": self.group_name})
            except Exception:  # noqa: BLE001
                pass

        # Tell SGLang instances to join (each blocks until the group forms), then
        # form the group on our side as rank 0.
        import threading

        def _join(ep: str):
            _post(
                ep,
                "init_weights_update_group",
                {
                    "master_address": master_addr,
                    "master_port": master_port,
                    "rank_offset": self._endpoint_rank_offset[ep],
                    "world_size": world_size,
                    "group_name": self.group_name,
                    "backend": "nccl",
                },
                timeout=600.0,
            )

        threads = [threading.Thread(target=_join, args=(ep,), daemon=True) for ep in self.sglang_endpoints]
        for t in threads:
            t.start()

        # Form the group on the verl side as rank 0 using SGLang's own helper —
        # a fresh NCCL group over a dedicated TCP store (NOT torch.distributed.
        # new_group, which can only sub-slice the existing default PG and cannot
        # span the out-of-process SGLang TP workers). This is the exact primitive
        # SGLang uses on its side, so the two rendezvous at tcp://master:port.
        # rank 0 must own a CUDA device before NCCL init.
        from sglang.srt.utils import init_custom_process_group

        torch.cuda.set_device(torch.cuda.current_device())
        self._group = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=0,
            group_name=self.group_name,
        )
        for t in threads:
            t.join()
        logger.info(f"external SGLang NCCL group formed: world_size={world_size}, master={master_addr}:{master_port}")


def _local_ip() -> str:
    import ray

    return ray.util.get_node_ip_address().strip("[]")


class ExternalSGLangCheckpointManager(CheckpointEngineManager):
    """Checkpoint manager for external SGLang.

    The stock manager (``base.CheckpointEngineManager.update_weights``) builds a
    Ray worker group from ``replica.workers`` and dispatches receive/lifecycle to
    verl-owned CE workers colocated on the SGLang GPUs. External SGLang has none
    of that, so we override to:
      - drive only the trainer worker group (its rank 0 self-pushes over HTTP+NCCL)
      - skip all replica-side orchestration
      - make sleep/wake no-ops (SGLang owns its own memory & lifecycle)

    ``replicas`` is expected to be empty here (see README on suppressing verl's
    own SGLang launch and pointing generation at the external service via
    ``rollout.gateway_url``).
    """

    def sleep_replicas(self):
        return None

    def wake_up_replicas(self):
        return None

    @auto_await
    async def update_weights(self, global_steps: int = None):
        # rank 0 of the trainer forms the NCCL group with the external SGLang and
        # broadcasts inside send_weights(); the other trainer ranks drain the
        # weight generator so their FSDP all-gathers complete. @auto_await lets the
        # sync trainer call site run this off the event loop (blocking ray.get is
        # offloaded to a thread), matching ExternalCheckpointManager.
        import ray

        ray.get(list(self.trainer.update_weights(global_steps=global_steps, mode=self.backend)))
