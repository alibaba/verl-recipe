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
import importlib

from remote_agent.proxyserver.models import CompletionRecord, SessionRecord


def test_session_record_roundtrip():
    turn = CompletionRecord(
        request_messages=[{"role": "user", "content": "hi"}],
        completion_text="hello",
        completion_token_ids=[1, 2, 3],
        completion_logprobs=[-0.1, -0.2, -0.3],
    )
    rec = SessionRecord(session_id="s1", turns=[turn])
    assert rec.turns[0].completion_token_ids == [1, 2, 3]
    assert rec.completed is False


def test_proxy_package_imports_cleanly():
    # ``vllm_provider`` is intentionally excluded: it imports ``litellm`` (a heavy
    # optional dep not installed in this environment). ``ray_actor`` imports it
    # lazily inside a function, so ray_actor itself still imports cleanly here.
    for mod in ["models", "recorder", "relay", "server", "ray_actor", "proxy_server"]:
        importlib.import_module(f"remote_agent.proxyserver.{mod}")
