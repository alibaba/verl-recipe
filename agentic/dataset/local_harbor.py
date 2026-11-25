"""Prepare recipe rows with the Alibaba Harbor 0.25 training SDK.

Harbor owns task discovery, schema validation, and first-step instructions.
This module only preserves the recipe's task identity/path columns and parquet
entry points. Imports stay lazy so other remote-agent runners need no Harbor.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


def _training_sdk():
    try:
        from harbor.training import TrainingDataset, TrainingExample, VerlAdapter
    except ImportError as exc:
        if getattr(exc, "name", None) not in {"harbor", "harbor.training"}:
            raise
        raise ImportError(
            "Harbor dataset preparation requires alibaba/harbor's training SDK. "
            "Install harbor[huggingface] from release-v0.25.0-fdba28fc. "
            "Add the oss extra only when reading a repository on OSS."
        ) from exc
    return TrainingDataset, TrainingExample, VerlAdapter


def _parse_example(dataset, example) -> dict[str, Any]:
    from harbor.models.task.config import TaskConfig

    local_dir = (dataset.directory / example.task_path).resolve()
    config = TaskConfig.model_validate_toml((local_dir / "task.toml").read_text(encoding="utf-8"))
    return {
        # Directory identity remains stable even if task.toml declares a
        # different display name; it is also the uploaded task archive root.
        "task_id": local_dir.name,
        "local_path": local_dir,
        "instruction": example.instruction,
        "task_config": config.model_dump(exclude_unset=True),
        "training_example": example,
    }


def iter_local_task_dirs(root: str | Path) -> Iterator[Path]:
    """Use Harbor's discovery/validation, including single and multi-step tasks."""
    TrainingDataset, _, _ = _training_sdk()
    dataset = TrainingDataset(root)
    for example in dataset.iter_examples():
        yield (dataset.directory / example.task_path).resolve()


def parse_local_task_dir(local_dir: Path) -> dict[str, Any]:
    """Keep the legacy task dictionary while delegating parsing to Harbor."""
    local_dir = Path(local_dir).expanduser()
    if not (local_dir / "task.toml").is_file():
        raise FileNotFoundError(f"Harbor task config not found: {local_dir / 'task.toml'}")
    TrainingDataset, _, _ = _training_sdk()
    dataset = TrainingDataset(local_dir)
    examples = dataset.iter_examples()
    example = next(examples)
    return _parse_example(dataset, example)


def task_to_verl_row(
    task: dict[str, Any],
    index: int,
    split: str,
    *,
    data_source: str = "harbor",
    ability: str = "agent",
    extra_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Add recipe identity/path columns to Harbor's native veRL row."""
    _, TrainingExample, VerlAdapter = _training_sdk()
    example = task.get("training_example")
    if example is None:
        # Preserve callers that construct the old dictionary themselves.
        example = TrainingExample(
            dataset=data_source,
            dataset_ref=None,
            index=index,
            task_name=task["task_id"],
            task_path=Path(task["local_path"]).name,
            instruction=task["instruction"],
            split=split,
        )
    example = replace(example, index=index, split=split)
    adapter = VerlAdapter(data_source=data_source, ability=ability, labeler=lambda _: task["task_id"])
    row = dict(adapter.to_row(example))
    config = task.get("task_config", {})
    row["extra_info"].update(
        {
            "task_id": task["task_id"],
            "local_path": str(task["local_path"]),
            "task_metadata": config.get("metadata", {}) if isinstance(config, dict) else {},
        }
    )
    if extra_info:
        row["extra_info"].update(extra_info)
    row["instance_id"] = task["task_id"]
    row["local_task_path"] = str(task["local_path"])
    return row


def build_rows_from_local(
    root: str | Path,
    *,
    split: str,
    data_source: str = "harbor",
    ability: str = "agent",
    extra_info: Optional[dict[str, Any]] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Read tasks through TrainingDataset and adapt only recipe-specific fields."""
    from itertools import islice

    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a non-negative integer or None")
    TrainingDataset, _, _ = _training_sdk()
    dataset = TrainingDataset(root, split=split)
    examples = dataset.iter_examples()
    if limit is not None:
        examples = islice(examples, limit)
    return [
        task_to_verl_row(
            _parse_example(dataset, example),
            index,
            split,
            data_source=data_source,
            ability=ability,
            extra_info=extra_info,
        )
        for index, example in enumerate(examples)
    ]


def _signature_for(root: Path, split: str, limit: Optional[int]) -> str:
    """Stable short hash describing the inputs that produced a parquet file."""
    raw = f"harbor-training-v1|{root.resolve()}|split={split}|limit={limit}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


def build_verl_parquet_from_local(
    root: str | Path,
    *,
    split: str,
    cache_dir: str | Path,
    data_source: str = "harbor",
    ability: str = "agent",
    extra_info: Optional[dict[str, Any]] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Produce a parquet file under ``cache_dir`` (or ``output_path``).

    The parquet path is deterministic: ``<cache_dir>/<split>-<sig>.parquet``
    where ``sig`` is a hash of (root, split, limit), so calling this function
    twice with the same arguments and ``overwrite=False`` is a no-op.

    Returns the absolute path of the parquet file.
    """
    try:
        from datasets import Dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Building verl parquet from a Harbor directory requires the "
            "'datasets' package. Install via: pip install datasets"
        ) from exc

    root_path = Path(root).expanduser()
    if output_path is not None:
        out = Path(output_path)
    else:
        cache_root = Path(cache_dir).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        out = cache_root / f"{split}-{_signature_for(root_path, split, limit)}.parquet"

    if out.exists() and not overwrite:
        logger.info("Reusing cached harbor parquet: %s", out)
        return out

    rows = build_rows_from_local(
        root_path,
        split=split,
        data_source=data_source,
        ability=ability,
        extra_info=extra_info,
        limit=limit,
    )
    if not rows:
        raise ValueError(f"no harbor tasks found under {root_path}; expected Harbor tasks containing task.toml")

    # A mapping with no keys is not a writable Parquet struct. Preserve
    # arbitrary task metadata as JSON at this file boundary (rows keep dicts).
    for row in rows:
        row["extra_info"]["task_metadata"] = json.dumps(row["extra_info"]["task_metadata"], sort_keys=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(out))
    logger.info("Wrote %d harbor task rows to %s", len(rows), out)
    return out


def build_verl_parquets(
    train_root: Optional[str | Path],
    val_root: Optional[str | Path],
    *,
    cache_dir: str | Path,
    data_source: str = "harbor",
    ability: str = "agent",
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    overwrite: bool = False,
) -> tuple[Optional[Path], Optional[Path]]:
    """Convenience helper that builds train/val parquet files in one call.

    Either root may be ``None`` — in that case the corresponding parquet path
    is returned as ``None`` and the caller should keep its existing dataset.
    """
    train_path = (
        build_verl_parquet_from_local(
            train_root,
            split="train",
            cache_dir=cache_dir,
            data_source=data_source,
            ability=ability,
            limit=train_limit,
            overwrite=overwrite,
        )
        if train_root
        else None
    )
    val_path = (
        build_verl_parquet_from_local(
            val_root,
            split="val",
            cache_dir=cache_dir,
            data_source=data_source,
            ability=ability,
            limit=val_limit,
            overwrite=overwrite,
        )
        if val_root
        else None
    )
    return train_path, val_path


__all__ = [
    "iter_local_task_dirs",
    "parse_local_task_dir",
    "task_to_verl_row",
    "build_rows_from_local",
    "build_verl_parquet_from_local",
    "build_verl_parquets",
]


def _main() -> None:
    """Prepare a parquet without importing verl, Ray, or any model runtime."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local-dir", type=Path, help="A task directory or a directory of tasks, including PVC mounts")
    source.add_argument("--dataset-ref", help="A published Harbor repository reference, e.g. team/benchmark@v1")
    parser.add_argument("--context", help="An existing Harbor dataset repository context")
    parser.add_argument("--download-dir", type=Path, help="New local directory for --dataset-ref")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset_ref and args.download_dir is None:
        parser.error("--dataset-ref requires --download-dir")
    if args.local_dir and (args.context or args.download_dir):
        parser.error("--context and --download-dir apply only to --dataset-ref")
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"output already exists: {args.output}")

    metadata = None
    local_dir = args.local_dir
    if args.dataset_ref:
        TrainingDataset, _, _ = _training_sdk()
        dataset = asyncio.run(
            TrainingDataset.from_repo(
                args.dataset_ref,
                args.download_dir,
                context=args.context,
                split=args.split,
            )
        )
        local_dir = dataset.directory
        metadata = {"dataset": dataset.name, "dataset_ref": dataset.reference}
    path = build_verl_parquet_from_local(
        local_dir,
        split=args.split,
        cache_dir=args.output.parent,
        output_path=args.output,
        extra_info=metadata,
    )
    print(path)


if __name__ == "__main__":
    _main()
