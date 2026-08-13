"""Agentic recipe with disaggregated training/rollout (SGLang + Mooncake).

Combines the disaggregated ``OneStepOffRayTrainer`` (separate training and
rollout node pools with cross-node weight sync) with the agentic proxy
server that bridges remote agent HTTP requests to verl's LLM rollout.

Usage::

    python -m recipe.agentic.agentic_disagg_main
"""

import asyncio
import logging
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from recipe.agentic.agentic_main import (
    _materialize_harbor_datasets,
    collect_yaml_env_overrides,
)
from recipe.agentic.timed_trainer import TimedOneStepOffRayTrainer
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


@ray.remote(num_cpus=10, max_concurrency=100)
class AgenticDisaggTaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)

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

        resource_pool_manager = create_resource_pool_manager(config, role_worker_mapping.keys())

        _materialize_harbor_datasets(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = TimedOneStepOffRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()

        # Start proxy server between init_workers() and fit()
        load_balancer = trainer.llm_server_manager.global_load_balancer
        proxy_cfg = config.get("proxy_server", {})
        standalone_proxy_url = os.environ.get("PROXY_SERVER_URL")

        if standalone_proxy_url:
            from recipe.agentic.proxyserver.ray_actor import start_lb_registry

            start_lb_registry(load_balancer=load_balancer)
            print(f"[agentic] standalone proxy: {standalone_proxy_url}")
            trainer._proxy_url = standalone_proxy_url
        else:
            from recipe.agentic.proxyserver.ray_actor import start_proxy_server

            proxy_url = start_proxy_server(
                load_balancer=load_balancer,
                model_path=config.actor_rollout_ref.model.path,
                host=proxy_cfg.get("host", "0.0.0.0"),
                port=proxy_cfg.get("port", 0),
                tool_format=proxy_cfg.get("tool_format", "hermes"),
                debug=proxy_cfg.get("debug", False),
                session_dump_dir=proxy_cfg.get("session_dump_dir"),
            )
            print(f"[agentic] proxy server started at {proxy_url}")
            trainer._proxy_url = proxy_url

        asyncio.run(trainer.fit())


@hydra.main(config_path="config", config_name="agentic_trainer_disagg", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)

    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        yaml_env = collect_yaml_env_overrides(config)
        if yaml_env:
            runtime_env_vars = dict(runtime_env_kwargs.get("env_vars", {}))
            for k, v in yaml_env.items():
                runtime_env_vars.setdefault(k, v)
                os.environ.setdefault(k, v)
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Pin the driver actor to a TRAINER node. It materializes datasets and drives
    # training, so it must NOT be scheduled onto an external SGLang worker pod:
    # those join Ray only to host colocated ReceiverCE actors and advertise spare
    # CPU (via an "sglang*" custom resource) that would otherwise attract this
    # actor, but they are rollout-side and may lack the dataset mount. The Ray head
    # typically has no schedulable CPU. So select an alive node that (a) is not an
    # SGLang worker and (b) has enough CPU, preferring a GPU (trainer) node.
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    _RUNNER_CPUS = 10  # keep in sync with AgenticDisaggTaskRunner num_cpus

    def _res(n, key):
        return n["Resources"].get(key) or 0

    def _is_sglang_worker(n):
        return any(k.startswith("sglang") for k in n["Resources"])

    candidates = [
        n
        for n in ray.nodes()
        if n.get("Alive") and not _is_sglang_worker(n) and _res(n, "CPU") >= _RUNNER_CPUS
    ]
    if not candidates:
        raise RuntimeError("No non-SGLang node with enough CPU to place AgenticDisaggTaskRunner")
    # Prefer trainer nodes (have GPUs); external SGLang workers join with num_gpus=0.
    candidates.sort(key=lambda n: (_res(n, "GPU"), _res(n, "CPU")), reverse=True)
    target_node_id = candidates[0]["NodeID"]

    runner = AgenticDisaggTaskRunner.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=target_node_id, soft=False),
    ).remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    main()
