"""Recipe-local compatibility for native V1 behavior-policy metrics."""

from copy import copy
from functools import wraps


def install_version_metrics(trainer):
    """Keep unknown-version samples while reporting staleness only for known ones.

    Native V1 casts every tag to int, including padding, before selecting rows.
    Run its metric implementation against a disposable metadata view, then replace
    only its off-policy metrics using original, known versions. Compatibility zeros
    never enter TransferQueue, training, persisted tags, or published version metrics.
    """
    native_compute = trainer._compute_metrics

    @wraps(native_compute)
    def compute(batch, metrics, timing_raw, global_steps, epoch):
        known = [
            tag
            for tag in batch.tags
            if not tag.get("is_padding", False)
            and tag.get("min_global_steps") is not None
            and tag.get("max_global_steps") is not None
        ]
        count = sum(not tag.get("is_padding", False) for tag in batch.tags)
        metric_batch = copy(batch)
        metric_batch.tags = [
            {
                **tag,
                "min_global_steps": tag.get("min_global_steps") or 0,
                "max_global_steps": tag.get("max_global_steps") or 0,
            }
            for tag in batch.tags
        ]
        # Keep provisional metrics private, including if native computation fails.
        result = dict(metrics)
        native_compute(metric_batch, result, timing_raw, global_steps, epoch)
        prefix = "training/off_policy/"
        for key in list(result):
            if key.startswith(prefix):
                del result[key]
        result[prefix + "known_version_count"] = len(known)
        result[prefix + "unknown_version_count"] = count - len(known)
        if known:
            values = {
                "trajectory_spans": [tag["max_global_steps"] - tag["min_global_steps"] + 1 for tag in known],
                "trajectory_staleness": [global_steps - 1 - tag["max_global_steps"] for tag in known],
                "trajectory_staleness_worst": [global_steps - 1 - tag["min_global_steps"] for tag in known],
            }
            for name, items in values.items():
                result[f"{prefix}{name}/mean"] = sum(items) / len(items)
                result[f"{prefix}{name}/max"] = max(items)
                result[f"{prefix}{name}/min"] = min(items)
        metrics.update(result)
        # Remove stale off-policy keys if a caller reuses its metrics dictionary.
        for key in list(metrics):
            if key.startswith(prefix) and key not in result:
                del metrics[key]

    trainer._compute_metrics = compute
