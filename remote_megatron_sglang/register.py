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
"""``VERL_USE_EXTERNAL_MODULES`` entry point for the ``megatron_sglang`` backend.

Point verl at this module and it wires the backend into verl core with no
per-backend if-branch in framework code::

    VERL_USE_EXTERNAL_MODULES=recipe.remote_megatron_sglang.register

verl imports this at ``import verl`` time (in the driver and in every Ray
worker that sets the env var). Its top-level side effects:

1. Import the adapter module — its ``@RemoteBackendRegistry.register(
   "megatron_sglang")`` decorator inserts the :class:`MegatronSGLangBackend`
   into the class registry, and the adapter in turn imports ``rollout_replica``
   which registers the ``megatron_sglang`` rollout replica.
2. Register a lazy loader for the ActorRollout forwarder worker via
   :meth:`RemoteBackendRegistry.register_worker`. ``main_ppo`` reads it back
   with :meth:`RemoteBackendRegistry.get_worker` to pick ``actor_rollout_cls``.
"""

from __future__ import annotations

from recipe.remote_megatron_sglang import backend as _backend  # noqa: F401  registers @register + rollout replica
from verl.remote_backend import RemoteBackendRegistry

_BACKEND_NAME = "megatron_sglang"


def _load_forwarder_worker() -> type:
    # Lazy so wiring the name into the registry never forces the forwarder's
    # (tensordict / httpx) imports until main_ppo actually selects this backend.
    from recipe.remote_megatron_sglang.forwarder_worker import MegatronSGLangForwarderWorker

    return MegatronSGLangForwarderWorker


RemoteBackendRegistry.register_worker(_BACKEND_NAME, _load_forwarder_worker)
