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
"""RemoteAgentLoop: a verl-compatible agent loop that delegates execution
to a remote third-party agent while capturing LLM traffic via an HTTP proxy.

The proxy runs as a Ray named actor (``ProxyServerActor``) on the head node,
managed by :mod:`remote_agent.proxyserver.proxy_server`.  Each
``RemoteAgentLoop`` instance communicates with the proxy entirely via HTTP —
it does not hold a direct Python reference to the proxy object.

Agent execution itself is delegated to a pluggable
:class:`~remote_agent.runner.base.ExternalAgentRunner`, selected by name at
config time.  The core loop never imports any framework-specific dependency;
that coupling lives entirely inside the runner.

URL scheme
~~~~~~~~~~
The proxy base URL (cluster-internal) is, e.g., ``http://10.0.1.5:9123``.
For a given ``trial_id`` the *external* agent receives::

    agent_base_url = f"http://{advertised_host}:{port}/{trial_id}/v1"

where ``advertised_host`` (config ``proxy.advertised_host`` or env
``REMOTE_AGENT_ADVERTISED_HOST``) specifies an IP reachable from outside the
Ray cluster.  When the LLM proxy is co-located with the trainer, this is just
the trainer node's externally reachable IP.

Session management (register / get / complete / delete) is done via the
proxy's REST endpoints, allowing ``RemoteAgentLoop`` to run on any node in the
Ray cluster.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

import aiohttp
from transformers import AutoProcessor, AutoTokenizer

from remote_agent.agent_loop.config import RemoteAgentCoreConfig
from remote_agent.proxyserver.models import SessionRecord
from remote_agent.runner.base import AgentRunResult, AgentTask, create_runner

logger = logging.getLogger(__name__)
# Make sure diagnostic INFO/DEBUG lines surface in Ray worker stdout even
# when the root logger is configured at WARNING level.
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s:%(name)s:%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)
logger.propagate = True
# Allow operator override; default to DEBUG while debugging local trial issues.
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "DEBUG"))
# Late imports to avoid hard dependency on verl at module level.
try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, DictConfigWrap, register
    from verl.workers.rollout.llm_server import LLMServerClient
except ImportError:  # pragma: no cover
    from dataclasses import dataclass
    from dataclasses import field as dc_field

    class AgentLoopBase:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    @dataclass
    class AgentLoopOutput:  # type: ignore[no-redef]
        prompt_ids: list[int] = dc_field(default_factory=list)
        response_ids: list[int] = dc_field(default_factory=list)
        response_mask: list[int] = dc_field(default_factory=list)
        response_logprobs: Optional[list[float]] = None
        routed_experts: Any = None
        multi_modal_data: Optional[dict] = None
        reward_score: Optional[float] = None
        num_turns: int = 0
        metrics: Any = None
        extra_fields: dict = dc_field(default_factory=dict)

    class LLMServerClient:  # type: ignore[no-redef]
        pass

    class DictConfigWrap:  # type: ignore[no-redef]
        pass

    def register(name):  # type: ignore[no-redef]
        def decorator(cls):
            return cls

        return decorator


# ---------------------------------------------------------------------------
# RemoteAgentLoop
# ---------------------------------------------------------------------------


@register("remote_agent")
class RemoteAgentLoop(AgentLoopBase):
    """Agent loop that delegates execution to a remote third-party agent.

    Instead of running the agent loop locally (like verl's
    ``ToolAgentLoop``), this implementation:

    1. Discovers the proxy server URL via a lightweight Ray RPC
       (``get_proxy_url()``).  The proxy itself is a Ray named actor
       managed by :mod:`remote_agent.proxyserver.proxy_server`.
    2. Registers a unique *trial_id* with the proxy via HTTP.
    3. Delegates the actual agent run to a pluggable
       :class:`~remote_agent.runner.base.ExternalAgentRunner`, telling it to
       use ``http://{advertised_host}:{port}/{trial_id}/v1`` as the OpenAI
       ``base_url`` (cluster-external URL).
    4. After the agent finishes, collects the recorded ``token_ids`` and
       ``logprobs`` from the proxy session via HTTP and reconstructs a
       verl-compatible ``AgentLoopOutput``.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config

        self.core = RemoteAgentCoreConfig.from_dictconfig(config.actor_rollout_ref.rollout.remote_agent)
        self._runner = create_runner(self.core.runner_name, self.core.runner_kwargs)
        self.response_length = config.actor_rollout_ref.rollout.response_length

    # ------------------------------------------------------------------
    # Task-path resolution helpers
    # ------------------------------------------------------------------

    def _collect_task_roots(self) -> list[Path]:
        """Collect local task roots from config + env.

        Combines ``self.core.task_roots`` (from ``task_path.roots`` in the
        config) with any path listed in ``$REMOTE_AGENT_TASK_DIRS`` (``:`` or
        ``,`` separated).  Duplicates are removed while preserving order.
        """
        dirs: list[Path] = [Path(os.path.expanduser(r)) for r in self.core.task_roots]

        env_dirs = os.getenv("REMOTE_AGENT_TASK_DIRS")
        if env_dirs:
            for raw in env_dirs.replace(":", ",").split(","):
                raw = raw.strip()
                if raw:
                    dirs.append(Path(os.path.expanduser(raw)))

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[Path] = []
        for d in dirs:
            key = str(d)
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    def _resolve_task_path(
        self,
        instance_id: str,
        explicit_local_path: str | None = None,
    ) -> str:
        """Resolve ``task_path`` for a given ``instance_id``.

        Resolution order:

        1. ``explicit_local_path`` (the ``local_task_path`` column emitted by
           the dataset) if it points to an existing directory — this is the
           fast path when the dataset row already carries the absolute path.
        2. ``<root>/<instance_id>`` for each configured task root
           (``task_path.roots``, then ``$REMOTE_AGENT_TASK_DIRS``).
        3. ``self.core.task_template.format(instance_id=...)`` — kept for
           backward compatibility with the template-driven workflow.
        """
        if explicit_local_path:
            candidate = Path(os.path.expanduser(str(explicit_local_path)))
            if candidate.is_dir():
                return str(candidate)

        if instance_id:
            for root in self._collect_task_roots():
                candidate = root / instance_id
                if candidate.is_dir():
                    return str(candidate)

        return self.core.task_template.format(instance_id=instance_id)

    # ------------------------------------------------------------------
    # HTTP helpers — interact with the proxy via its REST endpoints
    # ------------------------------------------------------------------

    def _get_proxy_url(self) -> str:
        """Discover the proxy server URL via a lightweight Ray RPC to the
        named proxy actor.

        Raises ``RuntimeError`` if the proxy cannot be found.
        """
        from remote_agent.proxyserver.ray_actor import get_proxy_url

        url = get_proxy_url()
        if url is None:
            raise RuntimeError(
                "Proxy server actor not found.  Make sure the recipe calls start_proxy_server() before training starts."
            )
        return url

    async def _proxy_request(
        self,
        method: str,
        url: str,
        *,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> tuple[int, Any]:
        """Send an HTTP request to the proxy with automatic retry.

        Retries on connection errors and 5xx responses with exponential
        backoff.  Returns ``(status_code, json_body)``.
        """
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(method, url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        body = await resp.json() if resp.content_type == "application/json" else await resp.text()
                        if resp.status < 500:
                            return resp.status, body
                        last_error = RuntimeError(f"Proxy returned {resp.status}: {str(body)[:200]}")
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last_error = e

            if attempt + 1 < max_retries:
                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    "Proxy request %s %s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    method,
                    url,
                    attempt + 1,
                    max_retries,
                    last_error,
                    delay,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(f"Proxy request {method} {url} failed after {max_retries} attempts: {last_error}")

    async def _register_session(self, proxy_url: str, trial_id: str) -> None:
        """POST /sessions/{trial_id} to register a new session."""
        status, _ = await self._proxy_request("POST", f"{proxy_url}/sessions/{trial_id}")
        if status >= 400:
            raise RuntimeError(f"Failed to register session {trial_id}: HTTP {status}")

    async def _get_session_data(self, proxy_url: str, trial_id: str) -> SessionRecord | None:
        """GET /sessions/{trial_id} to retrieve recorded session data."""
        status, body = await self._proxy_request("GET", f"{proxy_url}/sessions/{trial_id}")
        if status == 404:
            return None
        if status >= 400:
            raise RuntimeError(f"Failed to get session {trial_id}: HTTP {status}")
        return SessionRecord(**body)

    async def _complete_session(self, proxy_url: str, trial_id: str) -> None:
        """POST /sessions/{trial_id}/complete to mark session completed."""
        status, _ = await self._proxy_request("POST", f"{proxy_url}/sessions/{trial_id}/complete")
        if status >= 400:
            raise RuntimeError(f"Failed to complete session {trial_id}: HTTP {status}")

    async def _delete_session(self, proxy_url: str, trial_id: str) -> None:
        """DELETE /sessions/{trial_id} to remove session data."""
        status, _ = await self._proxy_request("DELETE", f"{proxy_url}/sessions/{trial_id}")
        if status >= 400:
            raise RuntimeError(f"Failed to delete session {trial_id}: HTTP {status}")

    async def _reset_session(self, proxy_url: str, trial_id: str) -> None:
        """POST /sessions/{trial_id}/reset to clear recorded turns before retry."""
        status, _ = await self._proxy_request("POST", f"{proxy_url}/sessions/{trial_id}/reset")
        if status >= 400:
            raise RuntimeError(f"Failed to reset session {trial_id}: HTTP {status}")

    # ------------------------------------------------------------------

    def _agent_base_url(self, proxy_url: str, trial_id: str) -> str:
        """Build the cluster-external OpenAI ``base_url`` the agent must use."""
        from urllib.parse import urlparse

        parsed = urlparse(proxy_url)
        port = parsed.port
        host = os.getenv("REMOTE_AGENT_ADVERTISED_HOST", self.core.advertised_host)
        if host == "0.0.0.0":
            logger.warning("advertised_host is 0.0.0.0; the external agent may not reach the proxy")
        if port is None:
            # urlparse yields port=None when proxy_url carries no explicit port
            # (e.g. "http://proxy"). Emitting ":None" would produce an
            # unreachable URL, so fall back to the URL's default scheme port
            # (http→80) and warn rather than silently building a broken URL.
            port = 443 if parsed.scheme == "https" else 80
            logger.warning(
                "proxy_url %r has no explicit port; defaulting to %d",
                proxy_url,
                port,
            )
        return f"http://{host}:{port}/{trial_id}/v1"

    async def _submit_with_retry(self, proxy_url, trial_id, task, sampling_params):
        """Delegate the agent run to the runner with exponential backoff retry.

        On each retry the proxy session is reset (via HTTP) to avoid
        double-recording from partial runs.  Returns an ``AgentRunResult``.
        """
        agent_base_url = self._agent_base_url(proxy_url, trial_id)
        last_error = None
        for attempt in range(self.core.max_retries):
            if attempt > 0:
                await asyncio.sleep(self.core.retry_base_delay * (2 ** (attempt - 1)))
                try:
                    await self._reset_session(proxy_url, trial_id)
                except Exception as e:
                    logger.warning("reset %s failed (non-fatal): %s", trial_id, e)
            try:
                res = await self._runner.run(task=task, agent_base_url=agent_base_url, sampling_params=sampling_params)
                if res.status == "completed":
                    return res
                last_error = RuntimeError(f"runner {res.status}: {res.error}")
            except Exception as e:
                last_error = e
            logger.warning("trial %s attempt %d/%d: %s", trial_id, attempt + 1, self.core.max_retries, last_error)
        return AgentRunResult(status="error", rewards={}, error=str(last_error))

    def _minimal_output(
        self,
        kwargs: dict[str, Any],
        runner_status: str | None = None,
        runner_error: str | None = None,
        reward_score: float = 0.0,
    ) -> AgentLoopOutput:
        """Build a minimal, always-valid ``AgentLoopOutput``.

        Used both when the proxy recorded no turns and as the catch-all
        fallback when ``run()`` hits an unexpected error.  The response is a
        single EOS token so downstream ``tokenizer.pad`` produces a proper
        tensor instead of choking on an empty list (which would crash with
        ``AttributeError`` on ``.dim()``).  ``reward_score`` defaults to 0.0 so
        ``_compute_score`` is skipped — the remote agent dataset may lack
        fields (e.g. ``data_source``) that the reward manager requires.

        The fallback prompt is built from the first available prompt source so
        it is non-empty for both the template workflow (``problem_statement``)
        and the harbor dataset (which carries ``prompt``/``raw_prompt`` chat
        messages instead).  Runner status/error are surfaced via
        ``extra_fields`` (``metrics`` is a fixed-schema ``AgentLoopMetrics`` that
        would silently drop these keys).
        """
        raw_prompt = kwargs.get("raw_prompt")
        prompt = kwargs.get("prompt")
        problem_statement = kwargs.get("problem_statement")
        if isinstance(raw_prompt, list) and raw_prompt:
            messages = raw_prompt
        elif isinstance(prompt, list) and prompt:
            messages = prompt
        elif problem_statement:
            messages = [{"role": "user", "content": problem_statement}]
        else:
            messages = [{"role": "user", "content": ""}]

        prompt_ids = self._tokenize_messages(messages)
        eos_token_id = self.tokenizer.eos_token_id

        extra_fields: dict[str, Any] = {}
        if runner_status is not None:
            extra_fields["runner_status"] = runner_status
        if runner_error is not None:
            extra_fields["runner_error"] = runner_error

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=[eos_token_id],
            response_mask=[0],
            response_logprobs=[0.0],
            reward_score=reward_score,
            num_turns=0,
            metrics={},
            extra_fields=extra_fields,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run a single rollout via a remote agent.

        This method upholds a hard fail-open contract: it MUST NEVER raise.
        The caller gathers rollout tasks without ``return_exceptions=True``, so
        a single raising ``run()`` would abort the entire batch.  On any
        failure a minimal (but valid) ``AgentLoopOutput`` is returned instead.

        Args:
            sampling_params: LLM sampling parameters.
            **kwargs: Dataset fields.  ``instance_id`` identifies the task,
                ``local_task_path`` optionally carries the resolved task
                directory, and ``problem_statement`` is used to build the
                fallback prompt.

        Returns:
            ``AgentLoopOutput`` with reconstructed token_ids, masks, and
            logprobs captured by the proxy — or a minimal EOS-only output on
            any failure.
        """
        try:
            import shortuuid

            _uid = shortuuid.uuid()
        except ImportError:  # pragma: no cover - shortuuid is a verl runtime dep
            import uuid

            _uid = uuid.uuid4().hex
        instance_id = kwargs.get("instance_id", "")
        trial_id = instance_id + "-" + _uid
        self._sampling = sampling_params

        proxy_url = None
        try:
            # 1. Discover proxy URL (cluster-internal) via Ray named actor.
            proxy_url = self._get_proxy_url()

            # 2. Register session with the proxy via HTTP.
            await self._register_session(proxy_url, trial_id)

            try:
                # 3. Resolve the task and delegate the run to the runner.
                task = AgentTask(
                    instance_id=instance_id,
                    task_path=self._resolve_task_path(instance_id, kwargs.get("local_task_path")),
                    row=dict(kwargs),
                )
                result = await self._submit_with_retry(proxy_url, trial_id, task, sampling_params)

                # 4. Collect session data from the proxy via HTTP.
                #    Fetch BEFORE delete — the session is removed in finally.
                session = await self._get_session_data(proxy_url, trial_id)
                await self._complete_session(proxy_url, trial_id)

                # Surface the runner result via extra_fields.  verl re-validates
                # ``.metrics`` into a fixed-schema ``AgentLoopMetrics`` that drops
                # unknown keys, so runner status/error would be lost there.
                if session is None or not session.turns:
                    logger.warning(
                        "Session %s has no recorded turns — the agent may not have called the proxy.",
                        trial_id,
                    )
                    # Use the runner reward if available (the agent may have
                    # succeeded even though the proxy didn't record turns, e.g.
                    # due to a streaming disconnect); fall back to 0.0.
                    reward_score = 0.0
                    if result.rewards:
                        reward_score = sum(result.rewards.values())
                    return self._minimal_output(
                        kwargs,
                        runner_status=result.status,
                        runner_error=result.error,
                        reward_score=reward_score,
                    )

                # 5. Reconstruct verl output from the proxy's session recording.
                initial_messages = session.turns[0].request_messages
                output = self._reconstruct_output(session, initial_messages, worker_cache=[])
                output.extra_fields = {
                    **(output.extra_fields or {}),
                    "runner_status": result.status,
                }
                if result.error:
                    output.extra_fields["runner_error"] = result.error

                if result.rewards:
                    output.reward_score = sum(result.rewards.values())

                return output

            finally:
                if proxy_url is not None:
                    try:
                        await self._delete_session(proxy_url, trial_id)
                    except Exception as e:
                        logger.warning("delete session %s failed (non-fatal): %s", trial_id, e)

        except Exception:
            logger.exception(
                "RemoteAgentLoop.run failed for %s; returning minimal output",
                trial_id,
            )
            return self._minimal_output(kwargs)

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize message content fields.

        Some agents send ``content`` as a list of content-parts (OpenAI
        multi-part format).  The tokenizer's ``apply_chat_template``
        expects plain strings, so we flatten them here.
        """
        normalized = []
        for msg in messages:
            msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif "text" in part:
                            text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                msg["content"] = "\n".join(text_parts) if text_parts else ""
            normalized.append(msg)
        return normalized

    def _tokenize_messages(self, messages: list[dict[str, Any]]) -> list[int]:
        """Tokenize messages using the chat template."""
        messages = self._normalize_messages(messages)
        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            model_inputs = self.processor(text=[raw_prompt], return_tensors="pt")
            return model_inputs["input_ids"].squeeze(0).tolist()
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        return self._as_id_list(encoded)

    @staticmethod
    def _as_id_list(encoded) -> list[int]:
        """Normalize apply_chat_template output (list | BatchEncoding | tensor)."""
        input_ids = getattr(encoded, "input_ids", None)
        if input_ids is not None and not isinstance(encoded, list):
            seq = input_ids
            if hasattr(seq, "ndim") and seq.ndim == 2:
                seq = seq[0]
            elif isinstance(seq, list) and seq and isinstance(seq[0], list):
                seq = seq[0]
            return seq.tolist() if hasattr(seq, "tolist") else list(seq)
        return encoded

    def _tokenize_tool_messages(self, messages: list[dict[str, Any]]) -> list[int]:
        """Tokenize tool / user response messages for the inter-turn gap."""
        if not messages:
            return []
        messages = self._normalize_messages(messages)

        has_user = any(m.get("role") == "user" for m in messages)
        if not has_user:
            prefix_msgs = [{"role": "user", "content": "x"}]
        else:
            prefix_msgs = []
        full_msgs = prefix_msgs + messages

        if prefix_msgs:
            if self.processor is not None:
                raw_pfx = self.processor.apply_chat_template(
                    prefix_msgs,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                pfx_ids = self.processor(text=[raw_pfx], return_tensors="pt")["input_ids"].squeeze(0).tolist()
            else:
                pfx_ids = self._as_id_list(
                    self.tokenizer.apply_chat_template(
                        prefix_msgs,
                        add_generation_prompt=True,
                        tokenize=True,
                    )
                )
        else:
            pfx_ids = []

        if self.processor is not None:
            raw = self.processor.apply_chat_template(
                full_msgs,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.processor(text=[raw], return_tensors="pt")
            ids = inputs["input_ids"].squeeze(0).tolist()
        else:
            ids = self._as_id_list(
                self.tokenizer.apply_chat_template(
                    full_msgs,
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )

        if pfx_ids and ids[: len(pfx_ids)] == pfx_ids:
            ids = ids[len(pfx_ids) :]

        sys_ids = self._get_system_prompt_ids()
        if sys_ids and ids[: len(sys_ids)] == sys_ids:
            ids = ids[len(sys_ids) :]
        return ids

    def _get_system_prompt_ids(self) -> list[int]:
        if hasattr(self, "_cached_sys_ids"):
            return self._cached_sys_ids
        try:
            if self.processor is not None:
                raw = self.processor.apply_chat_template(
                    [],
                    add_generation_prompt=False,
                    tokenize=False,
                )
                inputs = self.processor(text=[raw], return_tensors="pt")
                self._cached_sys_ids = inputs["input_ids"].squeeze(0).tolist()
            else:
                self._cached_sys_ids = self.tokenizer.apply_chat_template(
                    [],
                    add_generation_prompt=False,
                    tokenize=True,
                )
        except Exception:
            self._cached_sys_ids = []
        return self._cached_sys_ids

    # ------------------------------------------------------------------
    # Output reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_output(
        self,
        session: SessionRecord,
        initial_messages: list[dict[str, Any]],
        worker_cache: list[dict] | None = None,
    ) -> AgentLoopOutput:
        """Build ``AgentLoopOutput`` from the proxy's session recording.

        Token IDs and logprobs are read from *worker_cache* when provided;
        otherwise from the recorded turn (``completion_token_ids`` /
        ``completion_logprobs``).  If neither is available (e.g. worker
        restarted), the completion text is re-tokenized and logprobs default
        to 0.0.

        For each LLM turn the completion tokens get ``mask=1``; for every
        inter-turn gap (tool / user responses) the tokens get ``mask=0``
        with ``logprobs=0.0``.
        """
        prompt_ids = self._tokenize_messages(initial_messages)
        cache = worker_cache or []

        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        num_turns = 0

        for i, turn in enumerate(session.turns):
            # Get token_ids/logprobs from worker cache, fall back to re-tokenization
            if i < len(cache):
                turn_ids = cache[i]["token_ids"]
                turn_logprobs = cache[i]["logprobs"]
            elif turn.completion_token_ids:
                turn_ids = turn.completion_token_ids
                turn_logprobs = turn.completion_logprobs
            else:
                turn_ids = self.tokenizer.encode(turn.completion_text, add_special_tokens=False)
                turn_logprobs = [0.0] * len(turn_ids)

            response_ids.extend(turn_ids)
            response_mask.extend([1] * len(turn_ids))
            response_logprobs.extend(turn_logprobs)
            num_turns += 1

            # Inter-turn content (tool / user messages) → mask=0
            # With delta storage, next_turn.request_messages already
            # contains only the new messages added since the previous
            # turn (first turn stores full messages, subsequent turns
            # store deltas).
            if i + 1 < len(session.turns):
                next_turn = session.turns[i + 1]
                new_messages = next_turn.request_messages
                tool_messages = [m for m in new_messages if m.get("role") in ("tool", "user", "system")]
                if tool_messages:
                    tool_ids = self._tokenize_tool_messages(tool_messages)
                    response_ids.extend(tool_ids)
                    response_mask.extend([0] * len(tool_ids))
                    response_logprobs.extend([0.0] * len(tool_ids))
                    num_turns += len(tool_messages)

        # Truncate to response_length
        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]
        response_logprobs = response_logprobs[: self.response_length]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else None,
            num_turns=num_turns,
            metrics={},
        )
