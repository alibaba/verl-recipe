"""Exercise both recipe adapters against the real Alibaba Harbor training SDK."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULES = ("agentic/dataset/local_harbor.py", "remote_agent/runner/harbor/dataset.py")


@pytest.fixture(params=MODULES)
def reader(request):
    spec = importlib.util.spec_from_file_location("harbor_dataset_under_test", ROOT / request.param)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task(root, name, instruction="Do the task", config=""):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "task.toml").write_text(config, encoding="utf-8")
    (directory / "instruction.md").write_text(instruction, encoding="utf-8")
    return directory


def test_sdk_first_step_and_legacy_task_identity(reader, tmp_path):
    directory = task(
        tmp_path,
        "folder-id",
        "Unused root instruction",
        '[task]\nname="demo/different-name"\n[[steps]]\nname="first"\n',
    )
    step = directory / "steps/first"
    step.mkdir(parents=True)
    (step / "instruction.md").write_text("First-step prompt", encoding="utf-8")
    parsed = reader.parse_local_task_dir(directory)
    row = reader.task_to_verl_row(parsed, 7, "val")
    assert row["prompt"] == [{"role": "user", "content": "First-step prompt"}]
    assert row["instance_id"] == row["reward_model"]["ground_truth"] == "folder-id"
    assert row["local_task_path"] == str(directory)
    assert row["extra_info"]["step_name"] == "first"
    assert row["extra_info"]["task_name"] == "demo/different-name"
    assert row["extra_info"]["index"] == 7
    assert row["extra_info"]["split"] == "val"


def test_sorted_tasks_single_directory_and_sdk_validation(reader, tmp_path):
    task(tmp_path, "b")
    first = task(tmp_path, "a")
    rows = reader.build_rows_from_local(tmp_path, split="train", limit=1)
    assert [r["instance_id"] for r in rows] == ["a"]
    assert reader.build_rows_from_local(first, split="train")[0]["local_task_path"] == str(first)
    (first / "instruction.md").write_text("  ", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty task instruction"):
        reader.build_rows_from_local(tmp_path, split="train")


def test_parquet_retains_sdk_provenance_and_empty_task_metadata(reader, tmp_path):
    directory = task(tmp_path / "data", "one")
    output = reader.build_verl_parquet_from_local(
        directory.parent,
        split="train",
        cache_dir=tmp_path / "cache",
        extra_info={"dataset_ref": "team/benchmark@v1"},
    )
    row = pq.read_table(output).to_pylist()[0]
    assert row["extra_info"]["dataset_ref"] == "team/benchmark@v1"
    assert row["extra_info"]["task_path"] == "one"
    assert row["extra_info"]["task_metadata"] == "{}"
    assert row["local_task_path"] == str(directory)
    assert row["instance_id"] == "one"


def test_repository_cli_materializes_with_sdk_and_preserves_reference(reader, tmp_path, monkeypatch):
    from harbor.training import TrainingDataset

    directory = task(tmp_path / "downloaded", "one")
    dataset = TrainingDataset(directory.parent, name="team/benchmark", reference="team/benchmark@v1")
    download = AsyncMock(return_value=dataset)
    monkeypatch.setattr(TrainingDataset, "from_repo", download)
    output = tmp_path / "train.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--dataset-ref",
            "team/benchmark@v1",
            "--context",
            "team",
            "--download-dir",
            str(tmp_path / "new-cache"),
            "--output",
            str(output),
        ],
    )
    reader._main()
    download.assert_awaited_once_with(
        "team/benchmark@v1",
        tmp_path / "new-cache",
        context="team",
        split="train",
    )
    row = pq.read_table(output).to_pylist()[0]
    assert row["extra_info"]["dataset"] == "team/benchmark"
    assert row["extra_info"]["dataset_ref"] == "team/benchmark@v1"
    assert Path(row["local_task_path"]).is_dir()


def test_local_cli_example_needs_no_model_runtime(reader, tmp_path):
    output = tmp_path / "example.parquet"
    subprocess.run(
        [
            sys.executable,
            reader.__file__,
            "--local-dir",
            str(ROOT / "tasks"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    row = pq.read_table(output).to_pylist()[0]
    assert row["instance_id"] == "hello-world"
    assert "Hello, Harbor!" in row["prompt"][0]["content"]
    assert json.loads(row["extra_info"]["task_metadata"])["difficulty"] == "easy"


def test_import_stays_independent_of_harbor_and_training_runtime(reader):
    code = (
        "import importlib.util,sys;"
        f"s=importlib.util.spec_from_file_location('adapter',{reader.__file__!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "assert not any(x in sys.modules for x in ('harbor','ray','torch'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_example_verifier_observes_real_output(tmp_path):
    example = ROOT / "tasks/hello-world"
    env = dict(os.environ, TASK_ROOT=str(tmp_path / "app"), VERIFIER_LOG_DIR=str(tmp_path / "logs"))
    verifier = ["sh", str(example / "tests/test.sh")]
    assert subprocess.run(verifier, env=env, check=False).returncode == 1
    assert (tmp_path / "logs/reward.txt").read_text() == "0\n"
    subprocess.run(["sh", str(example / "solution/solve.sh")], env=env, check=True)
    subprocess.run(verifier, env=env, check=True)
    assert (tmp_path / "logs/reward.txt").read_text() == "1\n"
