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
"""Pluggable weight-sync transport between the external Megatron training
cluster and the external SGLang inference cluster.

verl (the driver) never touches the weights: it only *triggers* a transfer that
happens directly between the two external clusters. This module defines the
strategy interface + a small registry so the transport is chosen by config
(``remote_backend.megatron_sglang.weight_sync.transport``) without any change to
the backend adapter. Built-in transports:

* ``nccl_http`` — Megatron rank 0 broadcasts unsharded HF weights over a
  cross-cluster NCCL group; SGLang pulls via its HTTP weight-update API. Fast,
  no SGLang patch, needs NCCL/RDMA reachability between clusters.
* ``store``     — Megatron writes a checkpoint to a shared store (HDFS/S3);
  SGLang reloads from disk. Slowest, but reachability-free.
* ``mooncake``  — RDMA point-to-point via the Mooncake transfer engine.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from recipe.remote_megatron_sglang.infer_client import SGLangInferClient
    from recipe.remote_megatron_sglang.train_client import MegatronTrainClient


class WeightSyncTransport(abc.ABC):
    """One weight-sync strategy. Constructed once per backend instance; its
    :meth:`setup` runs lazily before the first :meth:`sync`."""

    @classmethod
    @abc.abstractmethod
    def from_config(
        cls,
        cfg: DictConfig,
        train_client: MegatronTrainClient,
        infer_client: SGLangInferClient,
    ) -> WeightSyncTransport:
        """Build the transport from ``remote_backend.megatron_sglang.weight_sync``."""

    @abc.abstractmethod
    async def setup(self) -> None:
        """Idempotent one-time preparation (e.g. form the NCCL group). Safe to
        call before every :meth:`sync`; implementations must no-op on repeat."""

    @abc.abstractmethod
    async def sync(self, step: int) -> dict[str, Any]:
        """Move current weights train→infer for training ``step``. Returns a
        small metrics dict (timings, bytes) for logging."""

    @abc.abstractmethod
    async def teardown(self) -> None:
        """Release any transport resources. Idempotent."""


class WeightSyncRegistry:
    """name → :class:`WeightSyncTransport` class."""

    _transports: dict[str, type[WeightSyncTransport]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[WeightSyncTransport]], type[WeightSyncTransport]]:
        def _decorator(transport_cls: type[WeightSyncTransport]) -> type[WeightSyncTransport]:
            existing = cls._transports.get(name)
            if existing is not None and existing is not transport_cls:
                raise ValueError(f"Weight-sync transport '{name}' already registered to {existing!r}.")
            cls._transports[name] = transport_cls
            return transport_cls

        return _decorator

    @classmethod
    def create(
        cls,
        name: str,
        cfg: DictConfig,
        train_client: MegatronTrainClient,
        infer_client: SGLangInferClient,
    ) -> WeightSyncTransport:
        if name not in cls._transports:
            raise KeyError(f"Unknown weight-sync transport '{name}'. Registered: {sorted(cls._transports)}.")
        return cls._transports[name].from_config(cfg, train_client, infer_client)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._transports)
