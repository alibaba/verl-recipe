"""V1 lifecycle integration checks; requires a compatible verl installation."""

import inspect
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf
from recipe.agentic import agentic_main

from verl.trainer.main_ppo import TaskRunnerV1


def test_native_run_and_cleanup_are_inherited():
    runner = agentic_main.TaskRunner.__ray_metadata__.modified_class
    native = TaskRunnerV1.__ray_metadata__.modified_class
    assert inspect.unwrap(runner.run).__code__ is inspect.unwrap(native.run).__code__


def test_proxy_precedes_native_agent_loop_manager():
    runner = object.__new__(agentic_main.TaskRunner.__ray_metadata__.modified_class)
    runner.config = OmegaConf.create(
        {"proxy_server": {"port": 8001}, "actor_rollout_ref": {"model": {"path": "/model"}}}
    )
    lb = object()
    runner.trainer = SimpleNamespace(
        llm_server_manager=SimpleNamespace(global_load_balancer=lb), _compute_metrics=lambda *a, **kw: None
    )
    native = TaskRunnerV1.__ray_metadata__.modified_class
    events = []
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "recipe.agentic.proxyserver.ray_actor.start_proxy_server",
            side_effect=lambda **kw: events.append(("proxy", kw)) or "http://test",
        ),
        patch.object(native, "init_agent_loop_manager", lambda self: events.append(("manager", None))),
    ):
        runner.init_agent_loop_manager()
    assert [x[0] for x in events] == ["proxy", "manager"]
    assert events[0][1]["load_balancer"] is lb


def test_standalone_registry_precedes_native_agent_loop_manager():
    runner = object.__new__(agentic_main.TaskRunner.__ray_metadata__.modified_class)
    runner.config = OmegaConf.create({"proxy_server": {}, "actor_rollout_ref": {"model": {"path": "/model"}}})
    lb = object()
    runner.trainer = SimpleNamespace(
        llm_server_manager=SimpleNamespace(global_load_balancer=lb), _compute_metrics=lambda *a, **kw: None
    )
    native = TaskRunnerV1.__ray_metadata__.modified_class
    events = []
    with (
        patch.dict("os.environ", {"PROXY_SERVER_URL": "http://external"}, clear=True),
        patch(
            "recipe.agentic.proxyserver.ray_actor.start_lb_registry",
            side_effect=lambda **kw: events.append(("registry", kw)),
        ),
        patch.object(native, "init_agent_loop_manager", lambda self: events.append(("manager", None))),
    ):
        runner.init_agent_loop_manager()
    assert [x[0] for x in events] == ["registry", "manager"]
    assert events[0][1]["load_balancer"] is lb
