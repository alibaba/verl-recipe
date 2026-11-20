"""Behavior-policy metadata must survive the proxy and reconstruction boundaries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from recipe.agentic.agent_loop.remote_agent_loop import RemoteAgentLoop
from recipe.agentic.proxyserver.vllm_provider import VLLMRayProvider
from recipe.agentic.proxyserver.worker_client import InferenceWorkerClient


@pytest.mark.parametrize("max_tokens", [None, 8192])
def test_provider_preserves_zero_version(max_tokens):
    provider = object.__new__(VLLMRayProvider)
    provider.tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: [1],
        decode=lambda *args, **kwargs: "answer",
        eos_token_id=0,
    )
    output = SimpleNamespace(token_ids=[2], log_probs=[-0.1], stop_reason="stop", extra_fields={"global_steps": 0})
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return output

    server = SimpleNamespace(generate=SimpleNamespace(remote=generate))

    async def acquire(**kwargs):
        return "server", server

    provider._lb = SimpleNamespace(
        acquire_server=SimpleNamespace(remote=acquire),
        release_server=SimpleNamespace(remote=lambda **kwargs: None),
    )
    metadata = {}
    with patch("ray.get", side_effect=lambda value: value):
        result = asyncio.run(
            provider._generate(
                [{"role": "user", "content": "hello"}], "session", max_tokens=max_tokens, metadata_out=metadata
            )
        )
    assert result[:2] == ([2], [-0.1])
    assert metadata == {"global_steps": 0}
    if max_tokens is None:
        assert "max_tokens" not in calls[0]["sampling_params"]
    else:
        assert calls[0]["sampling_params"]["max_tokens"] == max_tokens


def test_worker_caches_generation_version():
    class Provider:
        async def _generate(self, *, metadata_out, **kwargs):
            metadata_out["global_steps"] = 0
            return [2], [-0.1], "stop", "answer", "server"

        async def _parse_tool_calls(self, *args):
            return "answer", [], "stop"

    worker = object.__new__(InferenceWorkerClient)
    worker._ensure_provider = lambda: None
    worker._provider = Provider()
    worker._session_cache = {}
    result = asyncio.run(worker._do_inference({"session_id": "session", "messages": []}))
    assert result["error"] is None
    assert worker.get_session_cache("session")[0]["global_steps"] == 0


@pytest.mark.parametrize("versions", [[0], [4], [0, 4], []])
def test_reconstructed_version_bounds(versions):
    loop = object.__new__(RemoteAgentLoop)
    loop._tokenize_messages = lambda messages: [1]
    loop._tokenize_tool_messages = lambda messages: []
    loop.response_length = 100
    turn = SimpleNamespace(
        completion_token_ids=[2], completion_logprobs=[-0.1], completion_text="answer", request_messages=[]
    )
    cache = [{"token_ids": [2], "logprobs": [-0.1], "global_steps": version} for version in versions]
    output = loop._reconstruct_output(SimpleNamespace(turns=[turn] * len(versions)), [], cache)
    assert output.extra_fields == {
        "min_global_steps": min(versions) if versions else None,
        "max_global_steps": max(versions) if versions else None,
    }


@pytest.mark.parametrize("requested", [None, 8192])
def test_ray_local_handler_to_reconstruction(requested):
    from recipe.agentic.proxyserver.ray_actor import _make_litellm_handler
    from recipe.agentic.proxyserver.recorder import SessionRecorder

    provider = object.__new__(VLLMRayProvider)
    provider.tokenizer = SimpleNamespace(
        apply_chat_template=lambda *a, **kw: [1], decode=lambda *a, **kw: "answer", eos_token_id=0
    )
    calls = []
    output = SimpleNamespace(token_ids=[2], log_probs=[-0.1], stop_reason="stop", extra_fields={"global_steps": 0})

    def generate(**kwargs):
        calls.append(kwargs)
        return output

    async def acquire(**kwargs):
        return "server", SimpleNamespace(generate=SimpleNamespace(remote=generate))

    async def parse(*args, **kwargs):
        return "answer", [], "stop"

    provider._lb = SimpleNamespace(
        acquire_server=SimpleNamespace(remote=acquire), release_server=SimpleNamespace(remote=lambda **kw: None)
    )
    provider._parse_tool_calls = parse
    recorder = SessionRecorder()
    recorder.create_session("session")
    proxy = SimpleNamespace(recorder=recorder)
    body = {} if requested is None else {"max_tokens": requested}

    import litellm

    with (
        patch("ray.get", side_effect=lambda value: value),
        patch.object(litellm, "custom_provider_map", [{"provider": "verl-vllm", "custom_handler": provider}]),
    ):
        response = asyncio.run(
            _make_litellm_handler()(proxy, "session", [{"role": "user", "content": "hello"}], body, False)
        )
    assert response.status_code == 200
    session = recorder.get_session("session")
    assert session.turns[0].global_steps == 0
    assert session.turns[0].completion_token_ids == [2]
    if requested is None:
        assert "max_tokens" not in calls[0]["sampling_params"]
    else:
        assert calls[0]["sampling_params"]["max_tokens"] == requested
    loop = object.__new__(RemoteAgentLoop)
    loop._tokenize_messages = lambda messages: [1]
    loop.response_length = 100
    result = loop._reconstruct_output(session, [], [])
    assert result.extra_fields == {"min_global_steps": 0, "max_global_steps": 0}


def test_missing_record_version_is_unknown():
    from recipe.agentic.proxyserver.recorder import SessionRecorder

    recorder = SessionRecorder()
    recorder.create_session("session")
    recorder.record_completion(
        "session", [{"role": "user", "content": "hello"}], "answer", token_ids=[2], logprobs=[-0.1]
    )
    loop = object.__new__(RemoteAgentLoop)
    loop._tokenize_messages = lambda messages: [1]
    loop.response_length = 100
    result = loop._reconstruct_output(recorder.get_session("session"), [], [])
    assert result.extra_fields == {"min_global_steps": None, "max_global_steps": None}
