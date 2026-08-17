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
"""V1 trainer integration for the ``megatron_sglang`` remote backend.

``verl.remote_backend.trainer.RemoteBackendTrainer`` subclasses the *deprecated*
``RayPPOTrainer``, and nothing in ``main_ppo`` selects it. The supported seam in
this verl version is the V1 trainer registry: ``main_ppo`` runs ``TaskRunnerV1``,
which resolves the trainer through
``get_trainer_cls(config.trainer.v1.trainer_mode)``. So we register a V1 trainer
mode instead, and reach the remote backend with **zero verl-core changes**:

    trainer.v1.trainer_mode=remote_megatron_sglang

What this class changes relative to :class:`PPOTrainerSync`:

* ``_init_resource_pool_mgr`` \u2014 the verl-side worker group becomes ONE CPU-only
  process running :class:`MegatronSGLangForwarderWorker` instead of N GPU
  processes running ``ActorRolloutRefWorker``. The external Megatron cluster owns
  the training GPUs; the external SGLang cluster owns the inference GPUs.
* ``main_config`` / ``backend_handle`` reach the forwarder through a thin bound
  subclass, because V1's ``_setup`` builds the actor ``RayClassWithInitArgs``
  with a fixed kwarg set (``config`` / ``distillation_config`` / ``role``).

What it deliberately does NOT change: the weight-sync trigger. V1's ``_setup``
pins ``checkpoint_engine.backend = "naive"``, and with that
``CheckpointEngineManager.update_weights()`` reduces to
``actor_wg.update_weights(global_steps=..., mode="naive")`` \u2014 which lands on
:meth:`MegatronSGLangForwarderWorker.update_weights` and drives the pluggable
transport. The inherited ``PPOTrainerSync`` hooks already call it at the right
points (after checkpoint load, and at the end of every step).
"""

from __future__ import annotations

import logging
import os

import ray

try:
    from verl.remote_backend import RemoteBackendRegistry
except ImportError:  # upstream verl main does not ship verl.remote_backend (PR #6422)
    from recipe.remote_megatron_sglang.remote_backend_compat import RemoteBackendRegistry

from verl.single_controller.ray import RayResourcePool, ResourcePoolManager
from verl.trainer.ppo.utils import Role, need_reference_policy
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

logger = logging.getLogger(__name__)

BACKEND_NAME = "megatron_sglang"
TRAINER_MODE = "remote_megatron_sglang"

_ENDPOINTS_ENV = "MEGATRON_SGLANG_ENDPOINTS"


class CpuOnlyResourcePoolManager(ResourcePoolManager):
    """``ResourcePoolManager`` that allocates CPU-only placement groups.

    The stock manager hardcodes ``use_gpu=True`` and then asserts that the Ray
    cluster has enough GPUs. Both are wrong here: the verl side owns no GPUs, so
    requesting one would (a) fail on a CPU-only driver pool and (b) idle a GPU
    that the external clusters could use. ``RayResourcePool`` already supports
    ``use_gpu=False`` natively (the bundle degrades to ``{"CPU": n}``), so this
    is a pure override with no core patch.
    """

    def create_resource_pool(self):
        for pool_name, process_on_nodes in self.resource_pool_spec.items():
            self.resource_pool_dict[pool_name] = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=False,
                max_colocate_count=self.max_colocate_count,
                name_prefix=pool_name,
            )
        # No _check_resource_available(): it only validates GPU counts, and we
        # request none.


def bind_forwarder_worker(main_config, backend_handle: dict):
    """Return a forwarder-worker subclass with ``main_config`` / ``backend_handle``
    baked in.

    V1's ``_setup`` constructs the actor worker as
    ``RayClassWithInitArgs(cls=..., config=..., distillation_config=..., role=...)``,
    so there is no seam to thread extra constructor kwargs through. Closing over
    them in a subclass is the smallest hook that avoids duplicating ``_setup``.
    Ray pickles the class by value (cloudpickle), so the closure travels to the
    worker process.
    """
    from recipe.remote_megatron_sglang.forwarder_worker import MegatronSGLangForwarderWorker

    class BoundMegatronSGLangForwarderWorker(MegatronSGLangForwarderWorker):
        def __init__(self, config, role, distillation_config=None, **kwargs):
            kwargs.pop("main_config", None)
            kwargs.pop("backend_handle", None)
            super().__init__(
                config=config,
                role=role,
                distillation_config=distillation_config,
                main_config=main_config,
                backend_handle=backend_handle,
                **kwargs,
            )

    return BoundMegatronSGLangForwarderWorker


@register_trainer(TRAINER_MODE)
class RemoteMegatronSGLangTrainer(PPOTrainerSync):
    """Sync PPO/GRPO where both engines are external (Megatron train + SGLang
    infer) and the verl side is a single CPU forwarder. See module docstring."""

    def __init__(self, config):
        super().__init__(config)
        self._validate_config(config)
        # Driver-side backend: validates endpoints/transport up front and
        # produces the reconnect handle the forwarder re-attaches with.
        self.backend = RemoteBackendRegistry.create(BACKEND_NAME, config)

    # ------------------------------------------------------------------ #
    # Config validation
    # ------------------------------------------------------------------ #

    def _validate_config(self, config) -> None:
        n_forwarders = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
        if n_forwarders != 1:
            raise ValueError(
                f"{TRAINER_MODE} needs exactly ONE CPU forwarder: set "
                f"trainer.n_gpus_per_node=1 and trainer.nnodes=1 (got "
                f"{n_forwarders}). On this trainer those two fields are a "
                "*process* count, not a GPU allocation \u2014 the external Megatron "
                "cluster owns the training parallelism."
            )

        rollout = config.actor_rollout_ref.rollout
        if rollout.name != BACKEND_NAME:
            raise ValueError(
                f"actor_rollout_ref.rollout.name must be '{BACKEND_NAME}' so "
                "LLMServerManager picks MegatronSGLangRolloutReplica (which "
                f"proxies to the external SGLang); got {rollout.name!r}."
            )

        rollout_world_size = (
            int(rollout.tensor_model_parallel_size)
            * int(rollout.data_parallel_size)
            * int(rollout.pipeline_model_parallel_size)
        )
        if rollout_world_size != 1:
            raise ValueError(
                "LLMServerManager derives the replica count as "
                "forwarder_world_size // (tp*dp*pp); with one CPU forwarder the "
                "verl-side rollout parallelism must be 1x1x1 (the external "
                "SGLang keeps its own TP). Set "
                "actor_rollout_ref.rollout.{tensor_model_parallel_size,"
                f"data_parallel_size,pipeline_model_parallel_size}}=1; got "
                f"{rollout_world_size}."
            )

        if not os.environ.get(_ENDPOINTS_ENV, "").strip():
            raise ValueError(
                f"{_ENDPOINTS_ENV} is unset. MegatronSGLangRolloutReplica reads "
                "the external SGLang base URL(s) from it (comma-separated), "
                "e.g. MEGATRON_SGLANG_ENDPOINTS=http://sglang-rbg-leader:30000."
            )

        if self.use_critic:
            raise ValueError(
                f"{TRAINER_MODE} has no critic worker: the verl side is a "
                "GPU-less forwarder. Use a critic-free advantage estimator "
                "(e.g. algorithm.adv_estimator=grpo)."
            )

        if config.reward.reward_model.enable:
            raise ValueError(
                f"{TRAINER_MODE} runs on a CPU-only resource pool, so a "
                "colocated/pooled reward model cannot be scheduled. Use a "
                "rule-based reward function."
            )

    # ------------------------------------------------------------------ #
    # Worker group: one CPU forwarder instead of N GPU actors
    # ------------------------------------------------------------------ #

    def _init_resource_pool_mgr(self):
        # Reuse the base mapping (reward/teacher roles, ref-in-actor logic) and
        # then swap the actor worker class + the pool itself.
        super()._init_resource_pool_mgr()

        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        actor_role = (
            Role.ActorRolloutRef if need_reference_policy(self.config) and not ref_in_actor else Role.ActorRollout
        )

        forwarder_cls = bind_forwarder_worker(self.config, self.backend.reconnect_handle())
        self.role_worker_mapping[actor_role] = ray.remote(forwarder_cls)

        # One CPU process, and every role maps onto that single pool.
        pool_name = self.mapping[actor_role]
        self.mapping = dict.fromkeys(self.mapping, pool_name)
        self.resource_pool_manager = CpuOnlyResourcePoolManager(
            resource_pool_spec={pool_name: [1]},
            mapping=self.mapping,
        )
        logger.info("%s: 1 CPU-only forwarder in pool %r (0 GPUs on the verl side)", TRAINER_MODE, pool_name)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def on_train_end(self):
        super().on_train_end()
        backend = getattr(self, "backend", None)
        if backend is None:
            return
        self.backend = None
        try:
            import asyncio

            asyncio.run(backend.destroy())
        except Exception as e:  # best-effort, idempotent
            logger.warning("remote backend destroy failed: %s", e)
