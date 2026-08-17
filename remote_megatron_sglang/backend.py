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
"""``megatron_sglang`` RemoteBackend adapter.

Drives two external clusters over HTTP:

* an external **Megatron** training cluster (a Kubeflow PyTorchJob) via
  :class:`MegatronTrainClient` — owns ``compute_log_prob`` / ``update_actor`` /
  ``save_checkpoint`` and holds the master weights;
* an external **SGLang** inference cluster (a RoleBasedGroup) via
  :class:`SGLangInferClient` for the weight-update endpoints (generation is
  routed separately through ``rollout.gateway_url``).

``update_weights`` delegates to a pluggable :class:`WeightSyncTransport`
selected by ``remote_backend.megatron_sglang.weight_sync.transport``.

The ABC only fixes lifecycle + weight-sync + checkpoint; the compute methods
(``compute_log_prob`` / ``compute_ref_log_prob`` / ``update_actor``) live here on
the adapter and are called by the forwarder worker.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf
from recipe.remote_megatron_sglang import rollout_replica  # noqa: F401  registers megatron_sglang replica
from recipe.remote_megatron_sglang.infer_client import SGLangInferClient
from recipe.remote_megatron_sglang.train_client import MegatronTrainClient
from recipe.remote_megatron_sglang.weight_sync import WeightSyncRegistry

try:
    from verl.remote_backend import RemoteBackend, RemoteBackendRegistry
except ImportError:  # upstream verl main does not ship verl.remote_backend (PR #6422)
    from recipe.remote_megatron_sglang.remote_backend_compat import RemoteBackend, RemoteBackendRegistry

logger = logging.getLogger(__name__)

_BACKEND_NAME = "megatron_sglang"


@RemoteBackendRegistry.register(_BACKEND_NAME)
class MegatronSGLangBackend(RemoteBackend):
    def __init__(self, cfg: DictConfig, train_client, infer_client, transport):
        self.cfg = cfg
        self.train_client: MegatronTrainClient = train_client
        self.infer_client: SGLangInferClient = infer_client
        self.transport = transport
        self._step = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, main_config: DictConfig, *, handle: dict[str, Any] | None = None) -> MegatronSGLangBackend:
        # Backend knobs live strictly under remote_backend.megatron_sglang.
        cfg = main_config.remote_backend.megatron_sglang

        # We opt OUT of RemoteBackendTrainer's built-in single-forwarder assert
        # (requires_single_forwarder() -> False), because its rollout-replica
        # check misreads agent-loop concurrency as replica count and rollout is
        # served by a decoupled external SGLang. But we still need exactly ONE
        # CPU forwarder (the forwarder relays the whole batch; >1 fragments it
        # and duplicates ONE_TO_ALL calls). Enforce that invariant here instead
        # of in the generic trainer.
        n_gpus = int(main_config.trainer.n_gpus_per_node)
        nnodes = int(main_config.trainer.nnodes)
        if n_gpus * nnodes != 1:
            raise ValueError(
                f"{_BACKEND_NAME} needs a single CPU forwarder: set "
                f"trainer.n_gpus_per_node=1 and trainer.nnodes=1 (got "
                f"{n_gpus}×{nnodes}={n_gpus * nnodes}). The external Megatron "
                "cluster owns the training parallelism; the verl side is one "
                "GPU-less forwarder."
            )

        train_client = MegatronTrainClient(
            endpoint=cfg.train_endpoint,
            timeout=cfg.get("train_timeout", 3600.0),
        )
        infer_client = SGLangInferClient(
            endpoints=OmegaConf.to_container(cfg.sglang_endpoints, resolve=True),
            timeout=cfg.get("infer_timeout", 1800.0),
        )
        transport = WeightSyncRegistry.create(cfg.weight_sync.transport, cfg.weight_sync, train_client, infer_client)
        # `handle` re-attach is a no-op beyond rebuilding the (stateless HTTP)
        # clients: the forwarder receives both main_config and the handle, and
        # everything it needs is derivable from main_config. The handle exists
        # so the ABC contract is honored and future stateful transports can
        # thread session ids through it.
        return cls(cfg, train_client, infer_client, transport)

    def reconnect_handle(self) -> dict[str, Any]:
        # HTTP clients are stateless; the forwarder rebuilds them from
        # main_config. Return a tiny marker so RemoteBackendTrainer has
        # something to thread into wg_kwargs["backend_handle"].
        return {"backend": _BACKEND_NAME, "reattach": True}

    async def destroy(self) -> None:
        try:
            await self.transport.teardown()
        except Exception as e:  # best-effort, idempotent
            logger.warning("weight-sync transport teardown failed: %s", e)

    # ------------------------------------------------------------------ #
    # Weight sync + checkpoint (ABC)
    # ------------------------------------------------------------------ #

    async def update_weights(self) -> dict[str, Any]:
        metrics = await self.transport.sync(step=self._step)
        self._step += 1
        return metrics

    async def save_checkpoint(self, local_path: str | None = None, global_step: int = 0) -> dict[str, Any]:
        # Save to the exact verl-side path (``.../global_step_N/actor``) so
        # verl's native auto-resume finds it: the driver and the train_server
        # share the pod filesystem. Falls back to the configured dir when the
        # caller doesn't thread a path (e.g. a bare ABC call).
        path = local_path or self.cfg.get("checkpoint_dir", "checkpoints/megatron_sglang")
        return await self.train_client.save_checkpoint(path=path, step=global_step)

    async def load_checkpoint(self, local_path: str | None = None) -> dict[str, Any]:
        # Symmetric with save_checkpoint: reload from the same verl-side path.
        path = local_path or self.cfg.get("checkpoint_dir", "checkpoints/megatron_sglang")
        return await self.train_client.load_checkpoint(path=path)

    def requires_single_forwarder(self) -> bool:
        # Opt OUT of RemoteBackendTrainer's generic assert: its rollout-replica
        # check conflates agent-loop concurrency with replica count, and our
        # rollout is served by a decoupled external SGLang. We validate the real
        # invariant (exactly one CPU forwarder) ourselves in from_config, so the
        # generic trainer scaffolding stays untouched.
        return False

    # ------------------------------------------------------------------ #
    # Compute (adapter-owned; called by the forwarder worker)
    # ------------------------------------------------------------------ #

    def compute_log_prob(self, data):
        return self.train_client.compute_log_prob(data)

    def compute_ref_log_prob(self, data):
        return self.train_client.compute_ref_log_prob(data)

    def update_actor(self, data):
        return self.train_client.update_actor(data)
