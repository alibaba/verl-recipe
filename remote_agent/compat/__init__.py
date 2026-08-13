# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Compat shim: let core verl config conversion tolerate recipe-added keys.

The remote_agent recipe adds a ``remote_agent`` section under
``actor_rollout_ref.rollout`` which core verl's ``RolloutConfig`` does not
know about. Core verl converts rollout configs via
``verl.utils.config.omega_conf_to_dataclass`` in several places (engine
workers, replica, agent loop manager); without this shim those conversions
raise ``TypeError: RolloutConfig.__init__() got an unexpected keyword
argument 'remote_agent'``.

This module wraps ``omega_conf_to_dataclass`` so that:

* typed path: keys not defined on the target dataclass are dropped before
  merging (interpolations are resolved on the original tree first, keeping
  their root context);
* hydra-instantiate path (``_target_`` present, no explicit type): if the
  target is a plain dataclass, the call is redirected to the typed path.

The original config object is never mutated, so ``RemoteAgentLoop`` can still
read ``config.actor_rollout_ref.rollout.remote_agent``.

Install once per process, as early as possible (the remote_agent recipe does
this from its entry point, and ships a ``sitecustomize.py`` hook so Ray worker
processes pick it up automatically via ``PYTHONPATH``).
"""

from __future__ import annotations

_installed = False


def install() -> None:
    """Patch ``verl.utils.config.omega_conf_to_dataclass`` in place."""
    global _installed
    if _installed:
        return
    try:
        import verl.utils.config as _vconf
    except Exception:
        return

    _orig = getattr(_vconf, "omega_conf_to_dataclass", None)
    if _orig is None or getattr(_orig, "_ra_compat", False):
        _installed = True
        return

    from dataclasses import fields, is_dataclass

    from omegaconf import OmegaConf

    def _target_class(cfg):
        """Best-effort resolution of a node's _target_ without interpolation."""
        t = None
        if hasattr(cfg, "_get_node"):
            node = cfg._get_node("_target_")
            if node is not None and hasattr(node, "_value"):
                t = node._value()
        elif isinstance(cfg, dict):
            t = cfg.get("_target_")
        if not (isinstance(t, str) and "." in t) or t.startswith("${"):
            return None
        import importlib

        mod_name, cls_name = t.rsplit(".", 1)
        try:
            return getattr(importlib.import_module(mod_name), cls_name, None)
        except Exception:
            return None

    def omega_conf_to_dataclass(config, dataclass_type=None):
        target_type = dataclass_type
        if target_type is None and config:
            try:
                cls = _target_class(config)
                if cls is not None and is_dataclass(cls):
                    target_type = cls
            except Exception:
                pass
        if target_type is not None and config and is_dataclass(target_type):
            try:
                keys = list(config.keys())
                known = {f.name for f in fields(target_type)}
                if any(k not in known for k in keys):
                    # only intervene when the config carries keys the target
                    # dataclass does not know about (e.g. rollout.remote_agent)
                    resolved = OmegaConf.to_container(config, resolve=True)
                    filtered = {k: v for k, v in resolved.items() if k in known}
                    merged = OmegaConf.merge(
                        OmegaConf.structured(target_type), OmegaConf.create(filtered)
                    )
                    return OmegaConf.to_object(merged)
            except Exception:
                # fall through to the original implementation untouched
                pass
        return _orig(config, dataclass_type)

    omega_conf_to_dataclass._ra_compat = True  # type: ignore[attr-defined]
    _vconf.omega_conf_to_dataclass = omega_conf_to_dataclass
    _installed = True


def setup_worker() -> None:
    """Ray ``worker_process_setup_hooks`` entry: install the config shim.

    Runs once in every Ray worker process before any task/actor executes.
    Agent-loop registration is handled by verl's own
    ``rollout.agent.agent_loop_config_path`` (see config/agent_loop.yaml),
    not here.
    """
    install()
