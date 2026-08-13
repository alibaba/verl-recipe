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
# processes get the same treatment via worker_process_setup_hooks below.
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
    """Run standard verl PPO, starting the LLM proxy (a Ray named actor on the
    head node) between ``init_workers()`` and ``fit()``.

    Ported from ``recipe/agentic/agentic_main.py`` (Ray-actor branch). The
    standalone-proxy / ``start_lb_registry`` branch and inline harbor dataset
    materialization are intentionally dropped: dataset materialization now runs
    up-front via :func:`materialize_dataset`, and only the Ray-actor proxy mode
    is supported here.
    """
    import os
    import socket
    from pprint import pprint

    import ray

    # importing this registers the "remote_agent" agent loop:
    from remote_agent.agent_loop import remote_agent_loop  # noqa: F401
    from remote_agent.proxyserver.ray_actor import start_proxy_server
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.trainer.main_ppo import TaskRunner as BaseTaskRunner
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.config import validate_config
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.device import auto_set_device
    from verl.utils.fs import copy_to_local

    core = RemoteAgentCoreConfig.from_dictconfig(config.actor_rollout_ref.rollout.remote_agent)

    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    # Migrate legacy reward_model.* / custom_reward_function / sandbox_fusion -> config.reward.*
    config = migrate_legacy_reward_impl(config)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        # Inject the compat hook into every Ray worker process so core verl
        # config conversion tolerates the recipe-added rollout.remote_agent
        # section (and the remote_agent agent loop gets registered there).
        # Ray requires runtime_env to be JSON-serializable, so pass the hook
        # as a fully-qualified "module:function" string.
        init_kwargs = OmegaConf.to_container(ray_init_kwargs)
        rt_env = init_kwargs.setdefault("runtime_env", {})
        hooks = rt_env.setdefault("worker_process_setup_hooks", [])
        hook_ref = "remote_agent.compat:setup_worker"
        if hook_ref not in hooks:
            hooks.append(hook_ref)
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**init_kwargs)

    runner = BaseTaskRunner()

    print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    actor_rollout_cls, ray_worker_group_cls = runner.add_actor_rollout_worker(config)
    runner.add_critic_worker(config)
    runner.add_reward_model_resource_pool(config)
    runner.add_teacher_model_resource_pool(config)
    runner.add_ref_policy_worker(config, actor_rollout_cls)

    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    resource_pool_manager = runner.init_resource_pool_mgr(config)

    train_dataset = create_rl_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        processor,
        is_train=True,
        max_samples=config.data.get("train_max_samples", -1),
    )
    val_dataset = create_rl_dataset(
        config.data.val_files,
        config.data,
        tokenizer,
        processor,
        is_train=False,
        max_samples=config.data.get("val_max_samples", -1),
    )
    train_sampler = create_rl_sampler(config.data, train_dataset)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=runner.role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
    )
    trainer.init_workers()

    # ----- remote_agent-specific: start the LLM proxy as a Ray named actor -----
    # The proxy holds the verl GlobalRequestLoadBalancer and exposes an
    # OpenAI-compatible HTTP endpoint that RemoteAgentLoop hands to the external
    # agent. It is registered under PROXY_ACTOR_NAME so RemoteAgentLoop can look
    # it up cluster-wide.
    load_balancer = trainer.llm_server_manager.global_load_balancer
    proxy_url = start_proxy_server(
        load_balancer=load_balancer,
        model_path=config.actor_rollout_ref.model.path,
        host="0.0.0.0",
        port=core.proxy_port,
        tool_format=core.tool_format,
    )
    print(f"Proxy server started at {proxy_url}")

    trainer.fit()

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    main()
