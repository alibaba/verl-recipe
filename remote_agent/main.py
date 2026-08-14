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
"""Entry point for the remote_agent recipe: standard verl PPO plus an
OpenAI-compatible LLM proxy the external agent records against. Dataset
materialization is delegated to the selected runner's build_dataset() hook, so
this module never imports any agent framework."""

from __future__ import annotations

import logging

from omegaconf import OmegaConf

from remote_agent import compat
from remote_agent.agent_loop.config import RemoteAgentCoreConfig
from remote_agent.runner.base import create_runner

# Make core verl config conversion tolerate the recipe-added
# ``rollout.remote_agent`` section in this (driver) process; Ray worker
# processes get the same treatment via sitecustomize.py (see deployment notes).
compat.install()

logger = logging.getLogger(__name__)


def materialize_dataset(config) -> None:
    """Delegate dataset materialization to the selected runner's build_dataset
    hook, writing the result back into ``data.train_files`` / ``data.val_files``.

    Pure and framework-free: it never imports any agent framework. Runners that
    own a dataset format (e.g. harbor) are imported lazily by ``create_runner``
    only when actually selected.
    """
    ra = config.actor_rollout_ref.rollout.remote_agent
    core = RemoteAgentCoreConfig.from_dictconfig(ra)
    runner = create_runner(core.runner_name, core.runner_kwargs)
    built = runner.build_dataset(config.data)
    if not built:
        return
    train_files, val_files = built
    if train_files:
        OmegaConf.update(config, "data.train_files", train_files, force_add=True)
    if val_files:
        OmegaConf.update(config, "data.val_files", val_files, force_add=True)
    logger.info("materialized dataset: train=%s val=%s", train_files, val_files)


def main() -> None:
    import hydra

    @hydra.main(config_path="config", config_name="remote_agent_trainer", version_base=None)
    def _run(config):
        materialize_dataset(config)
        _run_ppo_with_proxy(config)

    _run()


def _run_ppo_with_proxy(config) -> None:
    """Run verl PPO via ``run_ppo`` with a custom TaskRunner that injects
    ``start_proxy_server`` between ``trainer.init()`` and ``trainer.fit()``.

    Upstream verl (>= origin/main 535c4779) makes ``TaskRunnerV1`` a
    ``@ray.remote`` actor and expects recipes to pass a ``task_runner_class``
    to ``run_ppo()``.  We define ``_ProxyTaskRunner`` that replicates
    ``TaskRunnerV1.run()`` but calls ``start_proxy_server`` right after
    ``trainer.init()`` (so the load balancer exists) and before ``fit()``
    (so the agent loop can reach the proxy during rollout).
    """
    import ray

    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    # Pre-validation (run_ppo does not call validate_config itself).
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    # Connect to the existing KubeRay cluster before run_ppo tries ray.init().
    # Without this, run_ppo may start a new local cluster with 0 GPUs.
    if not ray.is_initialized():
        ray.init(address="auto")

    # -- _ProxyTaskRunner: TaskRunnerV1 with proxy injection ----------------
    # Defined here (not at module level) so that @ray.remote is applied at
    # call time, after ray is available.
    from verl.utils.logging_utils import configure_verl_logging
    from verl.utils.import_utils import load_class_from_fqn
    from pprint import pprint

    @ray.remote
    class _ProxyTaskRunner:
        """TaskRunnerV1 with LLM proxy injection between init() and fit()."""

        def __init__(self):
            self.config = None
            self.trainer = None
            self.agent_loop_manager = None

        def init_agent_loop_manager(self):
            from verl.trainer.ppo.v1 import AgentLoopManagerTQ

            manager_class_fqn = self.config.actor_rollout_ref.rollout.get(
                "agent", {}
            ).get("agent_loop_manager_class")
            if manager_class_fqn:
                agent_loop_manager_cls = load_class_from_fqn(
                    manager_class_fqn, "AgentLoopManager"
                )
            else:
                agent_loop_manager_cls = AgentLoopManagerTQ

            self.agent_loop_manager = agent_loop_manager_cls.create(
                config=self.config,
                llm_client=self.trainer.get_llm_client(),
                teacher_client=self.trainer.get_teacher_client(),
                reward_loop_worker_handles=self.trainer.get_reward_handles(),
            )

        def run(self, config):
            """Run PPO training with proxy injection."""
            configure_verl_logging()

            from verl.trainer.ppo.v1 import get_trainer_cls

            # transfer_queue is optional; use verl's mock-aware import path.
            try:
                import transfer_queue as tq
                _has_tq = True
            except ImportError:
                from verl.utils.transferqueue_utils import tq  # mock, raises on use
                _has_tq = False

            trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)

            config.transfer_queue.enable = _has_tq
            pprint(OmegaConf.to_container(config, resolve=True))
            OmegaConf.resolve(config)
            self.config = config

            if _has_tq:
                tq.init(config.transfer_queue)
            succeeded = False
            try:
                self.trainer = trainer_cls(config=config)
                self.trainer.init()

                # ----- proxy injection -----
                # Start the LLM proxy as a Ray named actor. The proxy runs
                # inside this TaskRunner actor (on a worker node), so use
                # the actor's own IP, NOT the head node's IP.
                core = RemoteAgentCoreConfig.from_dictconfig(
                    config.actor_rollout_ref.rollout.remote_agent
                )
                from remote_agent.proxyserver.ray_actor import start_proxy_server

                actor_ip = ray.util.get_node_ip_address()
                # Override advertised_host with the actor's own IP — the proxy
                # runs inside this TaskRunner actor (on a worker node), not on
                # the head node where the script set REMOTE_AGENT_ADVERTISED_HOST.
                config.actor_rollout_ref.rollout.remote_agent.proxy.advertised_host = actor_ip
                load_balancer = self.trainer.llm_server_manager.global_load_balancer
                proxy_url = start_proxy_server(
                    load_balancer=load_balancer,
                    model_path=config.actor_rollout_ref.model.path,
                    host=actor_ip,
                    port=core.proxy_port,
                    tool_format=core.tool_format,
                )
                print(f"Proxy server started at {proxy_url} (actor ip {actor_ip})")
                # ----- end injection -----

                # Note: V1 trainer _compute_metrics None-tags patch is applied
                # at build-time in the Dockerfile via sed on trainer_base.py.
                # No runtime monkey-patch needed.

                self.init_agent_loop_manager()
                self.trainer.fit(self.agent_loop_manager)
                succeeded = True
            finally:
                try:
                    tracking = getattr(self.trainer, "logger", None)
                    if tracking is not None:
                        tracking.finish(exit_code=0 if succeeded else 1)
                finally:
                    if _has_tq:
                        tq.close()

    # -- launch ---------------------------------------------------------------
    run_ppo(config, task_runner_class=_ProxyTaskRunner)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    main()
