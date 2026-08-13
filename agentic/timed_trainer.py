"""Trainer subclass that records per-step timing data to the proxy server.

Extends :class:`OneStepOffRayTrainer` to capture start/end timestamps for
the inference, weight-sync, and training-update phases of each ``fit_step``.
After each step, the timing record is sent (fire-and-forget) to the LLM
proxy server's ``/training/timing`` endpoint for persistent storage.

Usage::

    from recipe.agentic.timed_trainer import TimedOneStepOffRayTrainer

    trainer = TimedOneStepOffRayTrainer(
        config=config,
        tokenizer=tokenizer,
        ...,
        proxy_url="http://10.0.1.5:9123",  # optional; auto-discovered via Ray if omitted
    )
"""

from __future__ import annotations

import asyncio
import logging
import time

import aiohttp

from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer

logger = logging.getLogger(__name__)


class TimedOneStepOffRayTrainer(OneStepOffRayTrainer):
    """OneStepOffRayTrainer with per-step timing recording.

    Overrides :meth:`fit_step`, :meth:`_fit_generate`, and
    :meth:`_fit_update_weights` to capture start/end timestamps for
    inference, weight sync, and training phases.  Sends timing data to
    the proxy server after each step.

    The ``_timing_data`` dict is reset at the beginning of each
    ``fit_step`` and populated by the overridden sub-methods as they
    execute.  After the step completes, a :class:`TrainingRoundTiming`
    record is assembled and posted to the proxy.
    """

    def __init__(self, *args, proxy_url: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._proxy_url = proxy_url
        self._timing_data: dict = {}
        self._last_timing_task: asyncio.Task | None = None

    def _get_proxy_url(self) -> str | None:
        """Get proxy server URL.

        Returns the explicitly-set URL if available; otherwise falls back
        to Ray actor discovery.
        """
        if self._proxy_url:
            return self._proxy_url
        # Try Ray actor discovery
        try:
            from recipe.agentic.proxyserver.ray_actor import get_proxy_url

            return get_proxy_url()
        except Exception:
            return None

    async def _send_training_timing(self, timing_data: dict):
        """Send timing data to proxy server (fire-and-forget).

        Uses a short timeout so that a slow or unreachable proxy never
        blocks the training loop.
        """
        proxy_url = self._get_proxy_url()
        if not proxy_url:
            logger.debug("No proxy URL available, skipping timing recording")
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_url}/training/timing",
                    json=timing_data,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status >= 400:
                        logger.warning(
                            "Failed to send training timing: status=%d",
                            resp.status,
                        )
        except Exception as e:
            logger.warning("Failed to send training timing: %s", e)

    def _fit_update_weights(self):
        self._timing_data["weight_sync_start"] = time.time()
        try:
            super()._fit_update_weights()
        finally:
            self._timing_data["weight_sync_end"] = time.time()

    def _fit_compute_reward(self, batch):
        self._timing_data["reward_start"] = time.time()
        try:
            return super()._fit_compute_reward(batch)
        finally:
            self._timing_data["reward_end"] = time.time()

    def _fit_compute_log_prob(self, batch):
        self._timing_data["log_prob_start"] = time.time()
        try:
            return super()._fit_compute_log_prob(batch)
        finally:
            self._timing_data["log_prob_end"] = time.time()

    def _fit_compute_ref_log_prob(self, batch):
        self._timing_data["ref_log_prob_start"] = time.time()
        try:
            return super()._fit_compute_ref_log_prob(batch)
        finally:
            self._timing_data["ref_log_prob_end"] = time.time()

    def _fit_compute_critic(self, batch):
        self._timing_data["critic_start"] = time.time()
        try:
            return super()._fit_compute_critic(batch)
        finally:
            self._timing_data["critic_end"] = time.time()

    def _fit_compute_advantage(self, batch):
        self._timing_data["advantage_start"] = time.time()
        try:
            return super()._fit_compute_advantage(batch)
        finally:
            self._timing_data["advantage_end"] = time.time()

    def _fit_update_critic(self, batch):
        self._timing_data["update_critic_start"] = time.time()
        try:
            return super()._fit_update_critic(batch)
        finally:
            self._timing_data["update_critic_end"] = time.time()

    def _fit_update_actor(self, batch):
        self._timing_data["update_actor_start"] = time.time()
        try:
            return super()._fit_update_actor(batch)
        finally:
            self._timing_data["update_actor_end"] = time.time()

    def _fit_save_checkpoint(self):
        self._timing_data["checkpoint_start"] = time.time()
        try:
            super()._fit_save_checkpoint()
        finally:
            self._timing_data["checkpoint_end"] = time.time()

    async def _fit_generate(self, batch_data_future, continuous_iterator):
        """Override to capture inference timestamps.

        The parent's ``_fit_generate`` calls ``_fit_update_weights``
        internally, so ``weight_sync_start`` / ``weight_sync_end`` will
        be set by our override.
        """
        self._timing_data["inference_start"] = time.time()
        result = await super()._fit_generate(batch_data_future, continuous_iterator)
        self._timing_data["inference_end"] = time.time()
        return result

    async def fit_step(self, batch_data_future, continuous_iterator):
        """Override to capture step-level timestamps and send timing to proxy.

        After the parent ``fit_step`` completes, assembles a
        :class:`TrainingRoundTiming` record and posts it to the proxy
        server via ``asyncio.create_task`` (non-blocking).
        """
        self._timing_data = {}
        step_start = time.time()
        self._timing_data["step_start"] = step_start

        # Call parent fit_step
        result = await super().fit_step(batch_data_future, continuous_iterator)

        step_end = time.time()
        self._timing_data["step_end"] = step_end

        # Build timing record
        # Convert numpy int64/float64 to native Python types for JSON serialization
        phase_durations = None
        if self.timing_raw:
            phase_durations = {k: float(v) if hasattr(v, 'item') else v for k, v in self.timing_raw.items()}
        td = self._timing_data
        timing_record = {
            "epoch": int(self.epoch),
            "global_step": int(self.global_steps),
            "step_start": step_start,
            "step_end": step_end,
            "inference_start": td.get("inference_start"),
            "inference_end": td.get("inference_end"),
            "weight_sync_start": td.get("weight_sync_start"),
            "weight_sync_end": td.get("weight_sync_end"),
            "training_start": td.get("inference_end"),
            "training_end": step_end,
            "reward_start": td.get("reward_start"),
            "reward_end": td.get("reward_end"),
            "log_prob_start": td.get("log_prob_start"),
            "log_prob_end": td.get("log_prob_end"),
            "ref_log_prob_start": td.get("ref_log_prob_start"),
            "ref_log_prob_end": td.get("ref_log_prob_end"),
            "critic_start": td.get("critic_start"),
            "critic_end": td.get("critic_end"),
            "advantage_start": td.get("advantage_start"),
            "advantage_end": td.get("advantage_end"),
            "update_critic_start": td.get("update_critic_start"),
            "update_critic_end": td.get("update_critic_end"),
            "update_actor_start": td.get("update_actor_start"),
            "update_actor_end": td.get("update_actor_end"),
            "checkpoint_start": td.get("checkpoint_start"),
            "checkpoint_end": td.get("checkpoint_end"),
            "phase_durations": phase_durations,
        }

        # Send to proxy (fire-and-forget, don't block training)
        self._last_timing_task = asyncio.create_task(self._send_training_timing(timing_record))
        self._last_timing_task.add_done_callback(lambda t: t.exception() if not t.cancelled() and not t.done() else None)

        return result

    async def _flush_pending_timing(self):
        """Await the last pending timing task to ensure it completes before exit."""
        if self._last_timing_task is not None and not self._last_timing_task.done():
            try:
                await self._last_timing_task
            except Exception:
                pass
            self._last_timing_task = None

    async def fit(self):
        """Override to flush the last timing task before returning."""
        try:
            await super().fit()
        finally:
            await self._flush_pending_timing()
