"""Tests for TimedOneStepOffRayTrainer phase timing and the recording pipeline.

The trainer tests mock the heavy ML imports (verl, ray, torch) so they can run
in a bare Python environment with only pytest, pydantic, and aiohttp installed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import types
from unittest.mock import MagicMock

import pytest
from recipe.agentic.proxyserver.models import TrainingRoundTiming
from recipe.agentic.proxyserver.recorder import SessionRecorder

# ---------------------------------------------------------------------------
# TrainingRoundTiming model tests
# ---------------------------------------------------------------------------


class TestTrainingRoundTimingModel:
    """Verify the Pydantic model accepts all phase fields."""

    def test_minimal_fields(self):
        t = TrainingRoundTiming(epoch=0, global_step=1, step_start=1000.0)
        assert t.epoch == 0
        assert t.global_step == 1
        assert t.step_end is None

    def test_all_phase_fields(self):
        now = time.time()
        data = {
            "epoch": 1,
            "global_step": 5,
            "step_start": now,
            "step_end": now + 100,
            "inference_start": now + 1,
            "inference_end": now + 20,
            "weight_sync_start": now + 2,
            "weight_sync_end": now + 10,
            "training_start": now + 20,
            "training_end": now + 100,
            "reward_start": now + 21,
            "reward_end": now + 30,
            "log_prob_start": now + 31,
            "log_prob_end": now + 40,
            "ref_log_prob_start": now + 41,
            "ref_log_prob_end": now + 50,
            "critic_start": now + 51,
            "critic_end": now + 60,
            "advantage_start": now + 61,
            "advantage_end": now + 65,
            "update_critic_start": now + 66,
            "update_critic_end": now + 75,
            "update_actor_start": now + 76,
            "update_actor_end": now + 90,
            "checkpoint_start": now + 91,
            "checkpoint_end": now + 95,
            "phase_durations": {"step": 100.0, "gen": 20.0},
        }
        t = TrainingRoundTiming.model_validate(data)
        assert t.reward_start == data["reward_start"]
        assert t.update_actor_end == data["update_actor_end"]
        assert t.checkpoint_end == data["checkpoint_end"]
        assert t.phase_durations == {"step": 100.0, "gen": 20.0}

    def test_round_trip_json(self):
        now = time.time()
        t = TrainingRoundTiming(
            epoch=0,
            global_step=2,
            step_start=now,
            step_end=now + 10,
            reward_start=now + 1,
            reward_end=now + 3,
            update_actor_start=now + 5,
            update_actor_end=now + 8,
        )
        dumped = json.loads(t.model_dump_json())
        restored = TrainingRoundTiming.model_validate(dumped)
        assert restored.reward_start == t.reward_start
        assert restored.update_actor_end == t.update_actor_end


# ---------------------------------------------------------------------------
# SessionRecorder timing persistence tests
# ---------------------------------------------------------------------------


class TestRecorderTimingPersistence:
    """Verify record / list / load round-trips all phase fields."""

    def _make_timing(self, epoch=0, step=1, **overrides) -> TrainingRoundTiming:
        now = time.time()
        defaults = dict(
            epoch=epoch,
            global_step=step,
            step_start=now,
            step_end=now + 60,
            inference_start=now + 1,
            inference_end=now + 10,
            weight_sync_start=now + 2,
            weight_sync_end=now + 5,
            reward_start=now + 11,
            reward_end=now + 15,
            log_prob_start=now + 16,
            log_prob_end=now + 20,
            ref_log_prob_start=now + 21,
            ref_log_prob_end=now + 25,
            critic_start=now + 26,
            critic_end=now + 30,
            advantage_start=now + 31,
            advantage_end=now + 33,
            update_critic_start=now + 34,
            update_critic_end=now + 40,
            update_actor_start=now + 41,
            update_actor_end=now + 50,
            checkpoint_start=now + 51,
            checkpoint_end=now + 55,
        )
        defaults.update(overrides)
        return TrainingRoundTiming(**defaults)

    def test_record_and_load(self):
        recorder = SessionRecorder()
        with tempfile.TemporaryDirectory() as dump_dir:
            timing = self._make_timing(epoch=0, step=3)
            recorder.record_training_timing(timing, dump_dir)

            loaded = recorder.load_training_timing(0, 3, dump_dir)
            assert loaded is not None
            assert loaded.reward_start == timing.reward_start
            assert loaded.update_actor_end == timing.update_actor_end
            assert loaded.checkpoint_start == timing.checkpoint_start

    def test_list_returns_all_fields(self):
        recorder = SessionRecorder()
        with tempfile.TemporaryDirectory() as dump_dir:
            timing = self._make_timing(epoch=0, step=1)
            recorder.record_training_timing(timing, dump_dir)

            items = recorder.list_training_timings(dump_dir)
            assert len(items) == 1
            item = items[0]
            assert "reward_start" in item
            assert "update_actor_start" in item
            assert "checkpoint_end" in item
            assert item["reward_start"] == timing.reward_start

    def test_list_multiple_sorted(self):
        recorder = SessionRecorder()
        with tempfile.TemporaryDirectory() as dump_dir:
            recorder.record_training_timing(self._make_timing(epoch=0, step=2), dump_dir)
            recorder.record_training_timing(self._make_timing(epoch=0, step=1), dump_dir)
            recorder.record_training_timing(self._make_timing(epoch=1, step=1), dump_dir)

            items = recorder.list_training_timings(dump_dir)
            assert len(items) == 3
            keys = [(it["epoch"], it["global_step"]) for it in items]
            assert keys == [(0, 1), (0, 2), (1, 1)]

    def test_load_nonexistent(self):
        recorder = SessionRecorder()
        with tempfile.TemporaryDirectory() as dump_dir:
            assert recorder.load_training_timing(99, 99, dump_dir) is None


# ---------------------------------------------------------------------------
# Helpers to import TimedOneStepOffRayTrainer with mocked heavy deps
# ---------------------------------------------------------------------------


def _stub_verl_modules():
    """Insert lightweight stubs for verl.* into sys.modules so that
    ``recipe.agentic.timed_trainer`` can be imported without torch/ray/tensordict.

    Returns the list of module names that were injected (for cleanup).
    """
    injected = []

    class _FakeOneStepOffRayTrainer:
        def __init__(self, *a, **kw):
            pass

        def _fit_update_weights(self):
            pass

        def _fit_compute_reward(self, batch):
            return batch

        def _fit_compute_log_prob(self, batch):
            return batch

        def _fit_compute_ref_log_prob(self, batch):
            return batch

        def _fit_compute_critic(self, batch):
            return batch

        def _fit_compute_advantage(self, batch):
            return batch

        def _fit_update_critic(self, batch):
            return batch

        def _fit_update_actor(self, batch):
            return batch

        def _fit_save_checkpoint(self):
            pass

        async def _fit_generate(self, batch_data_future, continuous_iterator):
            return MagicMock(), None

        async def fit_step(self, batch_data_future, continuous_iterator):
            return None

        async def fit(self):
            pass

    stub_names = [
        "verl",
        "verl.protocol",
        "verl.experimental",
        "verl.experimental.one_step_off_policy",
        "verl.experimental.one_step_off_policy.ray_trainer",
        "verl.experimental.separation",
        "verl.experimental.separation.ray_trainer",
    ]
    for name in stub_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            injected.append(name)

    sys.modules["verl.experimental.one_step_off_policy.ray_trainer"].OneStepOffRayTrainer = _FakeOneStepOffRayTrainer

    return injected, _FakeOneStepOffRayTrainer


@pytest.fixture()
def timed_trainer_module():
    """Import ``recipe.agentic.timed_trainer`` with verl deps stubbed out."""
    injected, FakeBase = _stub_verl_modules()
    mod_key = "recipe.agentic.timed_trainer"
    saved = sys.modules.pop(mod_key, None)
    try:
        import importlib

        mod = importlib.import_module(mod_key)
        yield mod, FakeBase
    finally:
        if saved is not None:
            sys.modules[mod_key] = saved
        else:
            sys.modules.pop(mod_key, None)
        for name in injected:
            sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# TimedOneStepOffRayTrainer phase override tests
# ---------------------------------------------------------------------------


class TestTimedTrainerOverrides:
    """Verify that each _fit_* override records timestamps in _timing_data."""

    def _make_trainer(self, timed_trainer_module):
        mod, _ = timed_trainer_module
        trainer = mod.TimedOneStepOffRayTrainer(proxy_url="http://localhost:8080")
        trainer._timing_data = {}
        trainer.timing_raw = {}
        trainer.epoch = 0
        trainer.global_steps = 1
        return trainer

    def test_fit_compute_reward(self, timed_trainer_module):
        trainer = self._make_trainer(timed_trainer_module)
        batch = MagicMock()
        result = trainer._fit_compute_reward(batch)
        assert result is batch
        assert "reward_start" in trainer._timing_data
        assert "reward_end" in trainer._timing_data
        assert trainer._timing_data["reward_end"] >= trainer._timing_data["reward_start"]

    def test_fit_compute_log_prob(self, timed_trainer_module):
        trainer = self._make_trainer(timed_trainer_module)
        batch = MagicMock()
        result = trainer._fit_compute_log_prob(batch)
        assert result is batch
        assert "log_prob_start" in trainer._timing_data
        assert "log_prob_end" in trainer._timing_data

    def test_fit_compute_ref_log_prob(self, timed_trainer_module):
        trainer = self._make_trainer(timed_trainer_module)
        batch = MagicMock()
        result = trainer._fit_compute_ref_log_prob(batch)
        assert result is batch
        assert "ref_log_prob_start" in trainer._timing_data
        assert "ref_log_prob_end" in trainer._timing_data

    def test_fit_compute_critic(self, timed_trainer_module):
        trainer = self._make_trainer(timed_trainer_module)
        batch = MagicMock()
        result = trainer._fit_compute_critic(batch)
        assert result is batch
        assert "critic_start" in trainer._timing_data
        assert "critic_end" in trainer._timing_data

    def test_fit_compute_advantage(self, timed_trainer_module):
        trainer = self._make_trainer(timed_trainer_module)
        batch = MagicMock()
        result = trainer._fit_compute_advantage(batch)
        assert result is batch
        assert "advantage_start" in trainer._timing_data
        assert "advantage_end" in trainer._timing_data

    def test_fit_update_critic(self, timed_trainer_module):
        trainer = self._make_trainer(timed_trainer_module)
        batch = MagicMock()
        result = trainer._fit_update_critic(batch)
        assert result is batch
        assert "update_critic_start" in trainer._timing_data
        assert "update_critic_end" in trainer._timing_data

    def test_fit_update_actor(self, timed_trainer_module):
        trainer = self._make_trainer(timed_trainer_module)
        batch = MagicMock()
        result = trainer._fit_update_actor(batch)
        assert result is batch
        assert "update_actor_start" in trainer._timing_data
        assert "update_actor_end" in trainer._timing_data

    def test_fit_update_weights(self, timed_trainer_module):
        trainer = self._make_trainer(timed_trainer_module)
        trainer._fit_update_weights()
        assert "weight_sync_start" in trainer._timing_data
        assert "weight_sync_end" in trainer._timing_data

    def test_fit_save_checkpoint(self, timed_trainer_module):
        trainer = self._make_trainer(timed_trainer_module)
        trainer._fit_save_checkpoint()
        assert "checkpoint_start" in trainer._timing_data
        assert "checkpoint_end" in trainer._timing_data

    def test_timing_captured_on_exception(self, timed_trainer_module):
        """End timestamps should be recorded even if the parent method raises."""
        mod, FakeBase = timed_trainer_module
        trainer = self._make_trainer(timed_trainer_module)

        original = FakeBase._fit_compute_reward
        FakeBase._fit_compute_reward = lambda self, batch: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            with pytest.raises(RuntimeError, match="boom"):
                trainer._fit_compute_reward(MagicMock())
            assert "reward_start" in trainer._timing_data
            assert "reward_end" in trainer._timing_data
        finally:
            FakeBase._fit_compute_reward = original


# ---------------------------------------------------------------------------
# Timing record assembly test (fit_step)
# ---------------------------------------------------------------------------


class TestTimingRecordAssembly:
    """Verify that fit_step assembles all phase timestamps into the record."""

    def test_fit_step_sends_all_phases(self, timed_trainer_module):
        mod, FakeBase = timed_trainer_module
        trainer = mod.TimedOneStepOffRayTrainer(proxy_url="http://localhost:8080")
        trainer._timing_data = {}
        trainer._last_timing_task = None
        trainer.timing_raw = {"step": 10.0, "gen": 3.0}
        trainer.epoch = 0
        trainer.global_steps = 5

        sent_records = []

        async def fake_send(timing_data):
            sent_records.append(timing_data)

        trainer._send_training_timing = fake_send

        now = time.time()

        async def fake_parent_fit_step(self_arg, batch_data_future, continuous_iterator):
            trainer._timing_data["inference_start"] = now + 1
            trainer._timing_data["inference_end"] = now + 10
            trainer._timing_data["weight_sync_start"] = now + 2
            trainer._timing_data["weight_sync_end"] = now + 5
            trainer._timing_data["reward_start"] = now + 11
            trainer._timing_data["reward_end"] = now + 15
            trainer._timing_data["log_prob_start"] = now + 16
            trainer._timing_data["log_prob_end"] = now + 20
            trainer._timing_data["ref_log_prob_start"] = now + 21
            trainer._timing_data["ref_log_prob_end"] = now + 25
            trainer._timing_data["critic_start"] = now + 26
            trainer._timing_data["critic_end"] = now + 30
            trainer._timing_data["advantage_start"] = now + 31
            trainer._timing_data["advantage_end"] = now + 33
            trainer._timing_data["update_critic_start"] = now + 34
            trainer._timing_data["update_critic_end"] = now + 40
            trainer._timing_data["update_actor_start"] = now + 41
            trainer._timing_data["update_actor_end"] = now + 50
            trainer._timing_data["checkpoint_start"] = now + 51
            trainer._timing_data["checkpoint_end"] = now + 55
            return "next_future"

        original = FakeBase.fit_step
        FakeBase.fit_step = fake_parent_fit_step
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(trainer.fit_step("batch_future", "iterator"))
            loop.close()
        finally:
            FakeBase.fit_step = original

        assert result == "next_future"
        assert len(sent_records) == 1
        record = sent_records[0]

        assert record["epoch"] == 0
        assert record["global_step"] == 5
        assert record["step_start"] is not None
        assert record["step_end"] is not None

        expected_phase_keys = [
            "inference_start",
            "inference_end",
            "weight_sync_start",
            "weight_sync_end",
            "reward_start",
            "reward_end",
            "log_prob_start",
            "log_prob_end",
            "ref_log_prob_start",
            "ref_log_prob_end",
            "critic_start",
            "critic_end",
            "advantage_start",
            "advantage_end",
            "update_critic_start",
            "update_critic_end",
            "update_actor_start",
            "update_actor_end",
            "checkpoint_start",
            "checkpoint_end",
        ]
        for key in expected_phase_keys:
            assert key in record, f"Missing key: {key}"
            assert record[key] is not None, f"Key {key} is None"

        assert record["phase_durations"] == {"step": 10.0, "gen": 3.0}
        assert record["training_start"] == record["inference_end"]
