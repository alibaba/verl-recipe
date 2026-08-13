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
"""CPU-only forwarder worker for the ``megatron_sglang`` remote backend.

This is the verl-side ``actor_rollout`` worker when
``trainer.remote_backend=megatron_sglang``. It owns no GPUs and no model: every
compute call is forwarded over HTTP to the external Megatron training cluster,
and weight sync/checkpoint are forwarded to the pluggable transport / train
server. The single-controller layer dispatches the same method surface the
trainer already calls on ``actor_rollout_wg`` (``init_model`` /
``compute_log_prob`` / ``update_actor`` / ``save_checkpoint`` / ...).

Because :meth:`MegatronSGLangBackend.requires_single_forwarder` is True,
``RemoteBackendTrainer`` guarantees exactly one instance of this worker, so the
mesh-dispatched compute calls forward the whole global batch to Megatron (which
does its own data/tensor/pipeline parallelism internally).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from tensordict import TensorDict

# Import the adapter so the megatron_sglang backend + rollout replica register
# in THIS process too. Ray worker processes don't inherit the driver's registry
# state, so the forwarder must trigger registration itself.
from recipe.remote_megatron_sglang import backend as _backend  # noqa: F401
from verl.remote_backend import RemoteBackendRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register

logger = logging.getLogger(__name__)


def _run_coro(coro):
    """Run an async backend call to completion from a synchronous ``@register``
    method. The Ray worker already runs an event loop (for the async
    ``update_weights`` hook), so a bare ``asyncio.run`` raises "cannot be called
    from a running event loop". When a loop is active we off-load the coroutine
    to a short-lived worker thread with its own loop; otherwise we run it inline.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


class MegatronSGLangForwarderWorker(Worker):
    """See module docstring. Mirrors the ActorRolloutRefWorker method surface
    that ``RayPPOTrainer.fit()`` invokes, but forwards everything to the remote
    backend instead of running an in-process engine."""

    def __init__(
        self,
        config,
        role: str,
        distillation_config=None,
        main_config=None,
        backend_handle: Optional[dict] = None,
        **kwargs,
    ):
        Worker.__init__(self)
        self.config = config
        self.role = role
        self._main_config = main_config
        self._backend_handle = backend_handle or {}
        self.backend = None
        self._is_ref = "ref" in role

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # Re-attach to the driver-side backend (stateless HTTP clients rebuilt
        # from main_config; handle threaded through for contract compliance).
        self.backend = RemoteBackendRegistry.get("megatron_sglang").from_config(
            self._main_config, handle=self._backend_handle
        )
        # Publish a trivial single-forwarder mesh so the mesh-dispatched compute
        # methods below route the whole batch to this one worker.
        self._register_dispatch_collect_info(mesh_name="actor", dp_rank=0, is_collect=True)
        if self._is_ref:
            self._register_dispatch_collect_info(mesh_name="ref", dp_rank=0, is_collect=True)
        logger.info("MegatronSGLangForwarderWorker attached to remote backend (role=%s)", self.role)

    # ------------------------------------------------------------------ #
    # Compute — forwarded to the external Megatron training cluster
    # ------------------------------------------------------------------ #

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        return self.backend.compute_log_prob(data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        return self.backend.compute_ref_log_prob(data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: TensorDict) -> TensorDict:
        return self.backend.update_actor(data)

    # ------------------------------------------------------------------ #
    # Weight sync + checkpoint — forwarded to the transport / train server
    # ------------------------------------------------------------------ #

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        """Trigger the pluggable train→infer weight sync.

        NOTE (integration seam): in this verl version, weight sync is normally
        driven by the CheckpointEngineManager + rollout replicas. With an
        external SGLang (generation via ``rollout.gateway_url``) there is no
        verl-managed replica to sync, so this method must be invoked from the
        rollout/checkpoint path. See README "Open issues". # VALIDATE
        """
        return await self.backend.update_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path=None, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # Forward verl's checkpoint path + step to the backend so the external
        # Megatron engine writes to the same global_step_N/actor dir verl tracks.
        return _run_coro(self.backend.save_checkpoint(local_path=local_path, global_step=global_step))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path=None, hdfs_path=None, del_local_after_load=False):
        # Resume: tell the external Megatron engine to reload from verl's
        # checkpoint path. Symmetric with save_checkpoint above.
        return _run_coro(self.backend.load_checkpoint(local_path=local_path))

    # ------------------------------------------------------------------ #
    # Profiling / async finalize hooks — no-ops on a CPU forwarder
    # ------------------------------------------------------------------ #

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        return None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self, **kwargs) -> None:
        return None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def async_calls_finalize_fn_exec(self, **kwargs) -> None:
        return None
