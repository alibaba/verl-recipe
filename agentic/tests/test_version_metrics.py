"""Recipe metric compatibility without changing verl or dropping samples."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = spec_from_file_location("version_metrics", Path(__file__).parents[1] / "version_metrics.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("bounds", [[(0, 0), (None, None)], [(None, None)], [(2, 4), (3, 3)]])
def test_unknown_versions_do_not_change_samples_or_other_metrics(bounds):
    tags = [dict(min_global_steps=a, max_global_steps=b) for a, b in bounds]
    tags.append(dict(is_padding=True, min_global_steps=None, max_global_steps=None))
    batch = SimpleNamespace(tags=tags, keys=list(range(len(tags))), rewards=[0] * len(tags))
    original = deepcopy(batch.__dict__)
    calls = []

    def native(view, metrics, timing_raw, global_steps, epoch):
        calls.append(view)
        # Reproduce native V1's cast-before-padding-selection contract.
        mask = np.array([not tag.get("is_padding", False) for tag in view.tags])
        low = np.array([tag["min_global_steps"] for tag in view.tags], dtype=int)[mask]
        high = np.array([tag["max_global_steps"] for tag in view.tags], dtype=int)[mask]
        metrics["training/off_policy/trajectory_spans/mean"] = (high - low + 1).mean()
        metrics["actor/loss"] = -0.1
        assert view.keys == batch.keys
        assert view.rewards == batch.rewards

    trainer = SimpleNamespace(_compute_metrics=native)
    module.install_version_metrics(trainer)
    metrics = {"training/off_policy/trajectory_spans/max": 999}
    trainer._compute_metrics(batch, metrics, {}, 6, 0)
    assert batch.__dict__ == original
    assert len(calls) == 1
    assert metrics["actor/loss"] == -0.1
    known = [(a, b) for a, b in bounds if a is not None and b is not None]
    assert metrics["training/off_policy/known_version_count"] == len(known)
    assert metrics["training/off_policy/unknown_version_count"] == len(bounds) - len(known)
    if known:
        assert metrics["training/off_policy/trajectory_spans/mean"] == sum(b - a + 1 for a, b in known) / len(known)
        assert metrics["training/off_policy/trajectory_staleness/max"] == max(5 - b for a, b in known)
        assert metrics["training/off_policy/trajectory_staleness_worst/min"] == min(5 - a for a, b in known)
    else:
        assert set(k for k in metrics if k.startswith("training/off_policy/")) == {
            "training/off_policy/known_version_count",
            "training/off_policy/unknown_version_count",
        }
