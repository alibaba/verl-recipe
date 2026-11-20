"""Agentic recipe: PPO training with an LLM proxy server for remote agents.

This recipe uses verl's native V1 TaskRunner lifecycle. It starts the existing
ProxyServerActor after trainer initialization and before the native TransferQueue
AgentLoopManager. Remote agents use the proxy to call the rollout replicas.

Usage::

    python -m recipe.agentic.agentic_main
"""

import json
import logging
import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import TaskRunnerV1
from verl.trainer.main_ppo import run_ppo as native_run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

# yaml 字段 → RemoteAgentLoop / proxy 使用的环境变量名的映射。
# 仅当 yaml 字段不为 None 且 shell 未设同名 env 时才注入，
# 以保证：命令行 override > shell 环境变量 > yaml 默认值。
_REMOTE_AGENT_ENV_MAP: dict[str, str] = {
    "agent_name": "REMOTE_AGENT_NAME",
    "agent_import_path": "REMOTE_AGENT_IMPORT_PATH",
    "model_name": "REMOTE_MODEL_NAME",
    "agent_kwargs": "REMOTE_AGENT_KWARGS",
    "max_retries": "REMOTE_AGENT_MAX_RETRIES",
    "retry_base_delay": "REMOTE_AGENT_RETRY_BASE_DELAY",
    "task_path_template": "REMOTE_AGENT_TASK_PATH_TEMPLATE",
    "environment_overrides": "REMOTE_AGENT_ENVIRONMENT_OVERRIDES",
    "environment_kwargs": "REMOTE_AGENT_ENVIRONMENT_KWARGS",
    "environment_import_path": "REMOTE_AGENT_ENVIRONMENT_IMPORT_PATH",
    "use_local_trial": "REMOTE_AGENT_USE_LOCAL_TRIAL",
    "proxy_server_url": "PROXY_SERVER_URL",
    "harbor_server_url": "HARBOR_SERVER_URL",
    "harbor_timeout": "HARBOR_TIMEOUT",
}
# 这些字段在 RemoteAgentConfig.from_env 中是 JSON 解码的，需要 dump 为字符串。
_REMOTE_AGENT_JSON_FIELDS = {
    "agent_kwargs",
    "environment_overrides",
    "environment_kwargs",
}


def _yaml_value_to_env_str(field: str, value) -> str:
    """将 yaml 中的值转成 RemoteAgentConfig.from_env 能识别的字符串。"""
    if field in _REMOTE_AGENT_JSON_FIELDS:
        # OmegaConf container -> plain python -> json
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def collect_yaml_env_overrides(config) -> dict[str, str]:
    """从 config 中收集需要注入的环境变量，仅覆盖 shell 中未设的项。”"""
    env_vars: dict[str, str] = {}

    proxy_cfg = config.get("proxy_server", {}) or {}
    llm_proxy_ip = proxy_cfg.get("llm_proxy_ip", None) if hasattr(proxy_cfg, "get") else None
    if llm_proxy_ip not in (None, "") and not os.environ.get("LLM_PROXY_IP"):
        env_vars["LLM_PROXY_IP"] = str(llm_proxy_ip)

    remote_agent_cfg = config.get("remote_agent", {}) or {}
    if hasattr(remote_agent_cfg, "get"):
        for field, env_name in _REMOTE_AGENT_ENV_MAP.items():
            value = remote_agent_cfg.get(field, None)
            if value is None:
                continue
            if os.environ.get(env_name):
                # shell 优先
                continue
            env_vars[env_name] = _yaml_value_to_env_str(field, value)

    return env_vars


def _materialize_harbor_datasets(config) -> None:
    """将 Harbor 本地任务目录物化为 parquet，并写回 config.data.train_files / val_files。

    仅在 ``config.data.train_harbor_dir`` / ``val_harbor_dir`` 任意一个被设置时生效。
    未设置的一侧保持原有 ``train_files`` / ``val_files`` 不变。
    """
    data_cfg = config.get("data", {}) or {}
    if not hasattr(data_cfg, "get"):
        return

    train_root = data_cfg.get("train_harbor_dir", None)
    val_root = data_cfg.get("val_harbor_dir", None)
    if not train_root and not val_root:
        return

    from recipe.agentic.dataset import build_verl_parquets

    cache_dir = data_cfg.get("harbor_cache_dir", "~/.cache/verl/agentic/harbor")
    cache_dir = os.path.expanduser(str(cache_dir))

    train_path, val_path = build_verl_parquets(
        train_root=os.path.expanduser(str(train_root)) if train_root else None,
        val_root=os.path.expanduser(str(val_root)) if val_root else None,
        cache_dir=cache_dir,
        data_source=data_cfg.get("harbor_data_source", "harbor"),
        ability=data_cfg.get("harbor_ability", "agent"),
        train_limit=data_cfg.get("harbor_train_limit", None),
        val_limit=data_cfg.get("harbor_val_limit", None),
        overwrite=bool(data_cfg.get("harbor_overwrite", False)),
    )

    if train_path is not None:
        print(f"[agentic] harbor train parquet -> {train_path}")
        OmegaConf.update(config, "data.train_files", [str(train_path)], merge=False)
    if val_path is not None:
        print(f"[agentic] harbor val parquet   -> {val_path}")
        OmegaConf.update(config, "data.val_files", [str(val_path)], merge=False)


@hydra.main(config_path="config", config_name="agentic_trainer", version_base=None)
def main(config):
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    run_ppo(config)


def run_ppo(config, task_runner_class=None) -> None:
    _materialize_harbor_datasets(config)
    yaml_env = collect_yaml_env_overrides(config)
    for key, value in yaml_env.items():
        os.environ.setdefault(key, value)
        OmegaConf.update(config, f"ray_kwargs.ray_init.runtime_env.env_vars.{key}", os.environ[key], force_add=True)
    config.transfer_queue.enable = True
    config.trainer.use_v1 = True
    validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    # The recipe materializes parquet on the driver filesystem. Keep the V1
    # controller local; model workers still span all configured nodes.
    runner = (task_runner_class or TaskRunner).options(resources={f"node:{ray.util.get_node_ip_address()}": 0.001})
    native_run_ppo(config, task_runner_class=runner)


@ray.remote
class TaskRunner(TaskRunnerV1.__ray_metadata__.modified_class):
    """Use the native V1 lifecycle; attach the remote-agent proxy before rollout."""

    def init_agent_loop_manager(self):
        from recipe.agentic.version_metrics import install_version_metrics

        install_version_metrics(self.trainer)
        load_balancer = self.trainer.llm_server_manager.global_load_balancer
        proxy_cfg = self.config.get("proxy_server", {})
        standalone_proxy_url = os.environ.get("PROXY_SERVER_URL")

        if standalone_proxy_url:
            from recipe.agentic.proxyserver.ray_actor import start_lb_registry

            start_lb_registry(load_balancer=load_balancer)
            print(
                f"[agentic] standalone proxy mode: PROXY_SERVER_URL="
                f"{standalone_proxy_url}; started lightweight"
                f" LoadBalancerRegistry (no local HTTP proxy)."
            )
        else:
            from recipe.agentic.proxyserver.ray_actor import start_proxy_server

            proxy_url = start_proxy_server(
                load_balancer=load_balancer,
                model_path=self.config.actor_rollout_ref.model.path,
                host=proxy_cfg.get("host", "0.0.0.0"),
                port=proxy_cfg.get("port", 0),
                tool_format=proxy_cfg.get("tool_format", "hermes"),
                debug=proxy_cfg.get("debug", False),
                session_dump_dir=proxy_cfg.get("session_dump_dir"),
            )
            print(f"Proxy server started at {proxy_url}")

        super().init_agent_loop_manager()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    main()
