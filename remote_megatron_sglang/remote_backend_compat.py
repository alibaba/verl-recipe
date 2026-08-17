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
"""Inline fallback for :mod:`verl.remote_backend`.

This recipe imports exactly two symbols from that package — the
:class:`RemoteBackend` ABC and :class:`RemoteBackendRegistry`. Upstream verl
main does not ship the package (per PR #6422 the abstraction lives in the
consumer repo), so when ``verl.remote_backend`` is absent the import sites fall
back to the local copies below. When the package IS present (our internal
branch carries it verbatim), the sites prefer it — within one process the
choice is deterministic, so there is ever only one registry, never two.

Keep this file semantically identical to ``verl/remote_backend/base.py`` if you
edit either side.
"""

from __future__ import annotations

import abc
from typing import Any, Callable

from omegaconf import DictConfig


class RemoteBackend(abc.ABC):
    """Out-of-process RL backend that owns its own GPUs.

    Created once on the driver (via ``from_config``); re-attached inside every
    forwarder worker via ``from_config(main_config, handle=...)``.
    """

    @classmethod
    @abc.abstractmethod
    def from_config(cls, main_config: DictConfig, *, handle: dict[str, Any] | None = None) -> RemoteBackend:
        """Sole public constructor. Backend knobs live under
        ``main_config.remote_backend.<name>``; ``handle`` re-attaches to a
        backend described by a previous :meth:`reconnect_handle`."""

    @abc.abstractmethod
    def reconnect_handle(self) -> dict[str, Any]:
        """Serializable handle that, passed back as ``handle=...``, re-attaches
        to *this* backend. ``RemoteBackendTrainer`` threads it into the forwarder
        workers via ``wg_kwargs``."""

    @abc.abstractmethod
    def destroy(self) -> None:
        """Tear the backend down. Must be idempotent."""

    @abc.abstractmethod
    async def update_weights(self) -> dict[str, Any]:
        """Sync trained weights from the training engine to the rollout engine."""

    @abc.abstractmethod
    async def save_checkpoint(self) -> dict[str, Any]:
        """Persist current model + optimizer state."""

    @abc.abstractmethod
    def requires_single_forwarder(self) -> bool:
        """Whether the trainer should assert ``n_gpus_per_node × nnodes == 1``
        and a single rollout replica. Opting out means the backend validates its
        own worker-group config."""


class RemoteBackendRegistry:
    """Process-wide registry of name → backend class + forwarder worker loader.

    Populated by adapter packages via ``@RemoteBackendRegistry.register(name)``
    and :meth:`register_worker`, wired in through
    ``VERL_USE_EXTERNAL_MODULES``.
    """

    _backends: dict[str, type[RemoteBackend]] = {}
    _worker_loaders: dict[str, Callable[[], type]] = {}
    _resolved_workers: dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[RemoteBackend]], type[RemoteBackend]]:
        """Decorator: register the decorated class as backend ``name``. Re-registering
        the same name with the identical class is a no-op; a different class raises."""

        def _decorator(backend_cls: type[RemoteBackend]) -> type[RemoteBackend]:
            existing = cls._backends.get(name)
            if existing is not None and existing is not backend_cls:
                raise ValueError(
                    f"Remote backend name '{name}' is already registered to "
                    f"{existing!r}; cannot re-register to {backend_cls!r}."
                )
            cls._backends[name] = backend_cls
            return backend_cls

        return _decorator

    @classmethod
    def get(cls, name: str) -> type[RemoteBackend]:
        if name not in cls._backends:
            raise KeyError(
                f"Unknown remote backend '{name}'. Registered: "
                f"{sorted(cls._backends)}. Wire the adapter package in via "
                "VERL_USE_EXTERNAL_MODULES=<pkg>.integrations.verl.register "
                "before starting verl."
            )
        return cls._backends[name]

    @classmethod
    def create(cls, name: str, main_config: DictConfig) -> RemoteBackend:
        return cls.get(name).from_config(main_config)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._backends)

    @classmethod
    def register_worker(cls, name: str, loader: Callable[[], type]) -> None:
        """Register a lazy loader for the forwarder worker class matching
        backend ``name``. Same-loader re-registration is a no-op; a different
        loader raises."""

        existing = cls._worker_loaders.get(name)
        if existing is not None and existing is not loader:
            raise ValueError(
                f"Remote backend '{name}' worker loader already registered to "
                f"{existing!r}; cannot re-register it to {loader!r}."
            )
        cls._worker_loaders[name] = loader

    @classmethod
    def get_worker(cls, name: str) -> type | None:
        """Return the forwarder worker class for ``name`` (invoking the loader
        once and caching), or ``None`` if the backend registered none."""
        if name in cls._resolved_workers:
            return cls._resolved_workers[name]
        loader = cls._worker_loaders.get(name)
        if loader is None:
            return None
        cls._resolved_workers[name] = loader()
        return cls._resolved_workers[name]
