"""SDK client for submitting agent runs to the Harbor Agent Run server.

Usage::

    from sdk.client import AgentRunClient

    client = AgentRunClient(server_url="http://harbor-server:8080")
    result = client.submit(
        task_path="/data/tasks/my-task",
        agent_name="claude_code",
        model_name="anthropic/claude-sonnet-4-20250514",
    )
    print(result.rewards)
    print(result.rollout_details)
"""

from __future__ import annotations

import asyncio
import io
import logging
import socket
import tarfile
from pathlib import Path
from typing import Any

import httpx

from .models import (
    AgentConfig,
    AgentRunRequest,
    AgentRunResponse,
    VerifierConfig,
)

# Public alias: SDK historically returned ``AgentRunResult``; keep the name
# stable while reusing the server-side response schema verbatim.
AgentRunResult = AgentRunResponse

logger = logging.getLogger(__name__)


class AgentRunError(Exception):
    """Raised when the server returns an HTTP error."""


def _create_task_archive(task_path: str) -> tuple[bytes, str]:
    """Create a gzipped tar archive of the local task directory.

    Returns:
        A tuple of (archive_bytes, original_dir_name).
    """
    task_dir = Path(task_path).resolve()
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Archive the directory under its own name so the server can
        # find it as <tmp>/<dir_name>/
        tar.add(str(task_dir), arcname=task_dir.name)
    buf.seek(0)
    return buf.getvalue(), task_dir.name


def _build_async_client(
    read_timeout: float,
    connect_timeout: float = 10.0,
) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` with TCP keepalive enabled.

    Long-lived connections through a Kubernetes ``Service`` (ClusterIP) can be
    silently dropped by conntrack / load-balancer idle timeouts.  Without
    ``SO_KEEPALIVE`` the client never notices the peer is gone and blocks until
    its (often very large) read timeout expires.  Enabling keepalive — together
    with the short per-request ``read_timeout`` used for polling — makes a dead
    connection surface as an error within seconds instead of hanging for hours.
    """
    keepalive_options: list[tuple[int, int, int]] = [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ]
    # Linux-specific probe tuning: start after 30s idle, probe every 10s,
    # give up after 3 failed probes (~1 min to detect a dead peer).
    for opt_name, value in (
        ("TCP_KEEPIDLE", 30),
        ("TCP_KEEPINTVL", 10),
        ("TCP_KEEPCNT", 3),
    ):
        opt = getattr(socket, opt_name, None)
        if opt is not None:
            keepalive_options.append((socket.IPPROTO_TCP, opt, value))

    timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
    try:
        transport = httpx.AsyncHTTPTransport(socket_options=keepalive_options)
        return httpx.AsyncClient(timeout=timeout, transport=transport)
    except TypeError:
        # httpx too old to accept socket_options — fall back without keepalive.
        return httpx.AsyncClient(timeout=timeout)


class AgentRunClient:
    """Synchronous and async client for submitting agent runs.

    The API is designed to feel similar to locally instantiating a Harbor
    Trial, but the actual execution happens on the remote server.
    The local task directory is automatically archived and uploaded to
    the server with each request.
    """

    def __init__(
        self,
        server_url: str,
        timeout: float = 1200.0,
    ):
        """Initialize the client.

        Args:
            server_url: Base URL of the Harbor Agent Run server
                (e.g. ``http://localhost:8080``).
            timeout: HTTP request timeout in seconds. Since agent runs can
                take a long time, the default is 30 minutes.
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def submit(
        self,
        task_path: str,
        job_id: str,
        task_id: str,
        agent_name: str | None = None,
        agent_import_path: str | None = None,
        model_name: str | None = None,
        agent_kwargs: dict[str, Any] | None = None,
        timeout_multiplier: float = 1.0,
        max_retries: int = 0,
        disable_verifier: bool = False,
        environment_overrides: dict[str, Any] | None = None,
        environment_kwargs: dict[str, Any] | None = None,
        llm_proxy_url: str | None = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Submit an agent run and block until the result is available.

        The local ``task_path`` directory is archived and uploaded to the
        server automatically.

        Args:
            task_path: Path to the local task directory. The directory and
                its contents will be uploaded to the server.
            agent_name: Name of a built-in Harbor agent (e.g. ``"claude_code"``).
            agent_import_path: Python import path of a custom agent
                (e.g. ``"my_agents:MyAgent"``). Mutually exclusive with
                ``agent_name``.
            model_name: LLM model name passed to the agent.
            agent_kwargs: Additional keyword arguments for the agent.
            timeout_multiplier: Multiplier for all timeout values.
            max_retries: Maximum number of automatic retries on failure.
            disable_verifier: If ``True``, skip verification after the agent run.
            environment_overrides: Optional dict to override environment params
                (e.g. ``{"override_cpus": 4}``).
            environment_kwargs: Extra keyword arguments forwarded to the
                environment constructor (see ``AgentRunRequest`` for the
                allowed key set).
            llm_proxy_url: URL of the LLM proxy server. When set, the remote
                agent will use this as its LLM endpoint so that token_ids
                and logprobs can be captured for RL training.
            **kwargs: Extra fields forwarded to the agent config.

        Returns:
            An :class:`AgentRunResult` containing ``rewards`` and
            ``rollout_details``.

        Raises:
            AgentRunError: If the server returns an HTTP error.
            FileNotFoundError: If ``task_path`` does not exist locally.
        """
        coro_factory = lambda: self.submit_async(
            task_path=task_path,
            job_id=job_id,
            task_id=task_id,
            agent_name=agent_name,
            agent_import_path=agent_import_path,
            model_name=model_name,
            agent_kwargs=agent_kwargs,
            timeout_multiplier=timeout_multiplier,
            max_retries=max_retries,
            disable_verifier=disable_verifier,
            environment_overrides=environment_overrides,
            environment_kwargs=environment_kwargs,
            llm_proxy_url=llm_proxy_url,
            **kwargs,
        )

        # Detect a running event loop without using the deprecated
        # ``asyncio.get_event_loop()`` helper. If one is running we have to
        # off-load to a worker thread because ``asyncio.run`` cannot be
        # nested inside an active loop.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            in_loop = False
        else:
            in_loop = True

        # NOTE: stay outside the ``except`` block before launching the work,
        # otherwise any downstream exception inherits the RuntimeError above
        # as ``__context__`` and clutters tracebacks with a misleading
        # "During handling of the above exception" frame.
        if not in_loop:
            return asyncio.run(coro_factory())

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro_factory())
            return future.result()

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def submit_async(
        self,
        task_path: str,
        job_id: str,
        task_id: str,
        agent_name: str | None = None,
        agent_import_path: str | None = None,
        model_name: str | None = None,
        agent_kwargs: dict[str, Any] | None = None,
        timeout_multiplier: float = 1.0,
        max_retries: int = 0,
        disable_verifier: bool = False,
        environment_overrides: dict[str, Any] | None = None,
        environment_kwargs: dict[str, Any] | None = None,
        llm_proxy_url: str | None = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Async version of :meth:`submit`.

        Archives the local task directory and uploads it to the server.
        """
        merged_kwargs = dict(agent_kwargs or {})
        merged_kwargs.update(kwargs)

        # Build the request using server-side pydantic models so the
        # payload is guaranteed to match the API contract.
        request = AgentRunRequest(
            task_path=task_path,
            job_id=job_id,
            task_id=task_id,
            agent=AgentConfig(
                name=agent_name,
                import_path=agent_import_path,
                model_name=model_name,
                # ``llm_proxy_url`` lives at the top level of the request,
                # not inside ``AgentConfig``. Do NOT pass it here.
                kwargs=merged_kwargs,
            ),
            timeout_multiplier=timeout_multiplier,
            max_retries=max_retries,
            verifier=VerifierConfig(disable=disable_verifier),
            environment_overrides=environment_overrides,
            environment_kwargs=environment_kwargs,
            llm_proxy_url=llm_proxy_url,
        )

        # Archive the local task directory off the event loop.
        archive_bytes, _dir_name = await asyncio.to_thread(
            _create_task_archive, task_path
        )

        url = f"{self.server_url}/api/v1/runs"
        logger.info("Submitting agent run to %s task=%s", url, task_path)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    data={
                        "metadata": request.model_dump_json(exclude_none=True)
                    },
                    files={
                        "task_archive": (
                            "task.tar.gz",
                            archive_bytes,
                            "application/gzip",
                        )
                    },
                )
        except Exception as e:
            return AgentRunResult(
                run_id="",
                status="timeout",
                rewards={"reward": 0.0},
                rollout_details=None,
                error=str(e),
            )

        if response.status_code != 200:
            return AgentRunResult(
                run_id="",
                status="failed",
                rewards={"reward": 0.0},
                rollout_details=None,
                error=(
                    f"Server returned HTTP {response.status_code}: "
                    f"{response.text}"
                ),
            )

        return AgentRunResult.model_validate(response.json())

    # ------------------------------------------------------------------
    # Server-side Async API (/api/v1/runs/async)
    # ------------------------------------------------------------------

    async def submit_async_task(
        self,
        task_path: str,
        job_id: str,
        task_id: str,
        agent_name: str | None = None,
        agent_import_path: str | None = None,
        model_name: str | None = None,
        agent_kwargs: dict[str, Any] | None = None,
        timeout_multiplier: float = 1.0,
        max_retries: int = 0,
        disable_verifier: bool = False,
        environment_overrides: dict[str, Any] | None = None,
        environment_kwargs: dict[str, Any] | None = None,
        llm_proxy_url: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a task to the server's async queue.

        Returns the ``run_id`` for subsequent polling.
        """
        merged_kwargs = dict(agent_kwargs or {})
        merged_kwargs.update(kwargs)

        request = AgentRunRequest(
            task_path=task_path,
            job_id=job_id,
            task_id=task_id,
            agent=AgentConfig(
                name=agent_name,
                import_path=agent_import_path,
                model_name=model_name,
                kwargs=merged_kwargs,
            ),
            timeout_multiplier=timeout_multiplier,
            max_retries=max_retries,
            verifier=VerifierConfig(disable=disable_verifier),
            environment_overrides=environment_overrides,
            environment_kwargs=environment_kwargs,
            llm_proxy_url=llm_proxy_url,
        )

        archive_bytes, _dir_name = await asyncio.to_thread(
            _create_task_archive, task_path
        )

        url = f"{self.server_url}/api/v1/runs/async"
        logger.info("Submitting async task to %s task=%s", url, task_path)

        # The async submit only enqueues the run and returns a ``run_id``
        # immediately, so bound it well below ``self.timeout`` (which governs
        # the overall trial wait) and enable keepalive.
        async with _build_async_client(min(self.timeout, 600.0)) as client:
            response = await client.post(
                url,
                data={
                    "metadata": request.model_dump_json(exclude_none=True)
                },
                files={
                    "task_archive": (
                        "task.tar.gz",
                        archive_bytes,
                        "application/gzip",
                    )
                },
            )

        if response.status_code != 200:
            raise AgentRunError(
                f"Server returned HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        return data["run_id"]

    async def poll_async_task(
        self,
        run_id: str,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
        request_timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Poll an async task until it reaches a terminal status.

        Each poll is a short, individually-bounded request (``request_timeout``)
        over a keepalive connection, and transient network / HTTP errors are
        tolerated — a dropped connection or a briefly-unavailable server is
        retried on the next tick rather than aborting the whole trial.  This is
        what makes the remote path robust to Kubernetes ``Service`` connections
        being silently dropped during a long-running trial.

        Returns the final status dict from the server.  Raises
        :class:`AgentRunError` only if no terminal status is observed before
        ``timeout`` seconds elapse.
        """
        import time

        deadline = time.monotonic() + timeout
        last_error: str | None = None
        async with _build_async_client(request_timeout) as client:
            while time.monotonic() < deadline:
                try:
                    resp = await client.get(
                        f"{self.server_url}/api/v1/runs/async/{run_id}/status"
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        status = data.get("status")
                        if status in ("completed", "failed", "timeout"):
                            return data
                        logger.debug(
                            "run_id=%s status=%s, polling...", run_id, status
                        )
                    elif resp.status_code == 404:
                        # The run may not be registered yet right after submit;
                        # keep polling until it appears (or the deadline hits).
                        last_error = "HTTP 404 (run not yet registered)"
                        logger.debug(
                            "run_id=%s status poll: %s", run_id, last_error
                        )
                    else:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        logger.warning(
                            "run_id=%s status poll failed: %s", run_id, last_error
                        )
                except (httpx.HTTPError, OSError, asyncio.TimeoutError) as e:
                    last_error = f"{type(e).__name__}: {e}"
                    logger.warning(
                        "run_id=%s status poll error (will retry): %s",
                        run_id, last_error,
                    )
                await asyncio.sleep(poll_interval)

        raise AgentRunError(
            f"Timed out waiting for run_id={run_id} after {timeout}s"
            + (f" (last error: {last_error})" if last_error else "")
        )

    async def get_async_result(self, run_id: str) -> dict[str, Any] | None:
        """Fetch the result of a completed async task."""
        async with _build_async_client(60.0) as client:
            resp = await client.get(
                f"{self.server_url}/api/v1/runs/async/{run_id}/result"
            )
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise AgentRunError(
                f"Result fetch failed: HTTP {resp.status_code}"
            )
        return resp.json()

    async def run_async_task(
        self,
        poll_interval: float = 5.0,
        poll_timeout: float = 600.0,
        **submit_kwargs: Any,
    ) -> AgentRunResult:
        """Submit to async queue, poll until done, return result.

        Convenience wrapper combining :meth:`submit_async_task`,
        :meth:`poll_async_task`, and :meth:`get_async_result`.

        Unlike the blocking :meth:`submit_async`, the result is retrieved by
        polling a short-lived status endpoint, so a connection silently dropped
        mid-trial (e.g. by a Kubernetes ``Service``) surfaces as a quick
        retryable error instead of hanging until the request timeout.
        """
        run_id = await self.submit_async_task(**submit_kwargs)
        logger.info("Async task submitted: run_id=%s", run_id)

        status_data = await self.poll_async_task(
            run_id, poll_interval=poll_interval, timeout=poll_timeout
        )

        result_data = None
        try:
            result_data = await self.get_async_result(run_id)
        except (AgentRunError, httpx.HTTPError, OSError, asyncio.TimeoutError) as e:
            # The trial already reached a terminal status; don't re-run it just
            # because the full-result fetch hit a transient error — fall back to
            # the rewards recorded in the status document.
            logger.warning(
                "run_id=%s: fetching full result failed (%s); "
                "falling back to status document", run_id, e,
            )

        if result_data:
            return AgentRunResult.model_validate(result_data)

        return AgentRunResult(
            run_id=run_id,
            status=status_data.get("status", "failed"),
            rewards=status_data.get("rewards") or {"reward": 0.0},
            rollout_details=None,
            error=status_data.get("error"),
        )

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    async def submit_batch_async(
        self,
        requests: list[dict[str, Any]],
    ) -> list[AgentRunResult]:
        """Submit multiple agent runs concurrently.

        Args:
            requests: List of keyword argument dicts, each passed to
                :meth:`submit_async`.

        Returns:
            List of results in the same order as the input requests.
        """
        tasks = [self.submit_async(**req) for req in requests]
        return await asyncio.gather(*tasks)

    def submit_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[AgentRunResult]:
        """Synchronous version of :meth:`submit_batch_async`."""
        return asyncio.run(self.submit_batch_async(requests))
