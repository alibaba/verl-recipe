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
import sys
import types

import pytest

from remote_agent.proxyserver.models import CompletionRecord, SessionRecord
from remote_agent.runner import base
from remote_agent.runner.base import AgentRunResult


class _FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        ids = list(range(1, len(messages) + 1))
        return ids if tokenize else "x"

    def encode(self, text, add_special_tokens=False):
        return [7, 8]


@pytest.mark.asyncio
async def test_core_reconstructs_and_stays_harbor_free(monkeypatch):
    from remote_agent.agent_loop import remote_agent_loop as ral

    session = SessionRecord(
        session_id="t",
        turns=[
            CompletionRecord(
                request_messages=[{"role": "user", "content": "hi"}],
                completion_text="ans",
                completion_token_ids=[11, 12],
                completion_logprobs=[-0.1, -0.2],
            ),
        ],
    )
    monkeypatch.setattr(ral.RemoteAgentLoop, "_get_proxy_url", lambda self: "http://p:1")

    async def _noop(self, *a, **k):
        return None

    monkeypatch.setattr(ral.RemoteAgentLoop, "_register_session", _noop)
    monkeypatch.setattr(ral.RemoteAgentLoop, "_complete_session", _noop)
    monkeypatch.setattr(ral.RemoteAgentLoop, "_delete_session", _noop)

    async def _get(self, url, tid):
        return session

    monkeypatch.setattr(ral.RemoteAgentLoop, "_get_session_data", _get)

    @base.register_runner("fake_ok")
    class _OK(base.ExternalAgentRunner):
        async def run(self, *, task, agent_base_url, sampling_params, **kw):
            return AgentRunResult(status="completed", rewards={"reward": 1.0})

    loop = ral.RemoteAgentLoop.__new__(ral.RemoteAgentLoop)  # bypass verl __init__
    loop.tokenizer = _FakeTokenizer()
    loop.processor = None
    loop.response_length = 128
    loop._runner = base.create_runner("fake_ok", {})
    loop.core = types.SimpleNamespace(
        max_retries=1, retry_base_delay=0.0, advertised_host="h", task_roots=[], task_template="{instance_id}"
    )
    loop._inference_worker = None

    monkeypatch.setenv("REMOTE_AGENT_ADVERTISED_HOST", "h")
    out = await loop.run({"temperature": 1.0}, raw_prompt=[{"role": "user", "content": "hi"}], instance_id="i1")

    assert out.response_ids[:2] == [11, 12]
    assert out.response_mask[:2] == [1, 1]
    assert out.reward_score == 1.0
    # Runner status is surfaced via extra_fields (metrics would drop it).
    assert out.extra_fields["runner_status"] == "completed"
    assert "harbor" not in sys.modules  # proves the core never imported harbor


def _build_loop(monkeypatch):
    """Construct a RemoteAgentLoop bypassing verl __init__ (shared helper)."""
    from remote_agent.agent_loop import remote_agent_loop as ral

    monkeypatch.setattr(ral.RemoteAgentLoop, "_get_proxy_url", lambda self: "http://p:1")

    async def _noop(self, *a, **k):
        return None

    monkeypatch.setattr(ral.RemoteAgentLoop, "_register_session", _noop)
    monkeypatch.setattr(ral.RemoteAgentLoop, "_complete_session", _noop)
    monkeypatch.setattr(ral.RemoteAgentLoop, "_delete_session", _noop)

    @base.register_runner("fake_ok2")
    class _OK(base.ExternalAgentRunner):
        async def run(self, *, task, agent_base_url, sampling_params, **kw):
            return AgentRunResult(status="completed", rewards={"reward": 1.0})

    loop = ral.RemoteAgentLoop.__new__(ral.RemoteAgentLoop)  # bypass verl __init__
    loop.tokenizer = _FakeTokenizer()
    loop.processor = None
    loop.response_length = 128
    loop._runner = base.create_runner("fake_ok2", {})
    loop.core = types.SimpleNamespace(
        max_retries=1, retry_base_delay=0.0, advertised_host="h", task_roots=[], task_template="{instance_id}"
    )
    monkeypatch.setenv("REMOTE_AGENT_ADVERTISED_HOST", "h")
    return ral, loop


@pytest.mark.asyncio
async def test_run_is_fail_open_on_error(monkeypatch):
    """run() must NEVER raise: an error inside the body yields minimal output."""
    ral, loop = _build_loop(monkeypatch)

    async def _boom(self, url, tid):
        raise RuntimeError("proxy exploded")

    monkeypatch.setattr(ral.RemoteAgentLoop, "_get_session_data", _boom)

    out = await loop.run({"temperature": 1.0}, instance_id="i1", problem_statement="solve it")

    # Does not raise; returns a usable EOS-fallback output.
    assert out.response_ids == [loop.tokenizer.eos_token_id]
    assert out.response_mask == [0]
    assert out.reward_score == 0.0


@pytest.mark.asyncio
async def test_run_no_recorded_turns_fallback(monkeypatch):
    """A session with no recorded turns yields the minimal single-EOS output."""
    ral, loop = _build_loop(monkeypatch)

    async def _get_empty(self, url, tid):
        return SessionRecord(session_id="t", turns=[])

    monkeypatch.setattr(ral.RemoteAgentLoop, "_get_session_data", _get_empty)

    out = await loop.run({"temperature": 1.0}, instance_id="i1", problem_statement="solve it")

    assert out.response_ids == [loop.tokenizer.eos_token_id]
    assert out.response_mask == [0]
    assert out.num_turns == 0
