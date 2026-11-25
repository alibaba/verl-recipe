# Local Harbor task input

`hello-world/` is a complete, small Harbor task: instruction, task configuration,
Docker build context, reference solution, and verifier. Its verifier awards 1
only when `/app/answer.txt` contains exactly `Hello, Harbor!` and a newline.
This is a data-loading and execution example, not a model-quality benchmark.

## Prepare training data without a model or cluster

From the recipe repository root, use Python 3.12 and the **Alibaba fork** of
Harbor. The training SDK is available on `release-v0.25.0-fdba28fc`; the revision
below pins the API used by this example rather than relying on the PyPI version
number alone:

```bash
uv venv --python 3.12 .venv-harbor-data
uv pip install --python .venv-harbor-data/bin/python \
  'harbor[huggingface] @ git+https://github.com/alibaba/harbor.git@1b30a8f162a0e38de12a66652186a25005e91341'

.venv-harbor-data/bin/python agentic/dataset/local_harbor.py \
  --local-dir ./tasks --split train --output ./prepared/train.parquet
.venv-harbor-data/bin/python agentic/dataset/local_harbor.py \
  --local-dir ./tasks/hello-world --split val --output ./prepared/val.parquet
```

The `remote_agent` recipe has the same preparation interface:

```bash
.venv-harbor-data/bin/python remote_agent/runner/harbor/dataset.py \
  --local-dir ./tasks --split train --output ./prepared/remote-train.parquet
```

Both entry points delegate discovery, task validation, and instruction selection
to `harbor.training.TrainingDataset`, and row construction to `VerlAdapter`.
They preserve SDK provenance in `extra_info`, add the existing recipe fields
`instance_id` and absolute `local_task_path`, and keep the task directory name as
the execution ID. For multi-step tasks the initial prompt is the first declared
step's instruction; the task runtime still executes the task. Arbitrary
`task.toml` metadata is serialized as JSON in the parquet's
`extra_info.task_metadata` to avoid empty Arrow structs.

Pass the generated files to verl through `data.train_files` and
`data.val_files`. Set `data.train_harbor_dir=null data.val_harbor_dir=null` to
avoid rematerializing them. Alternatively, when the training driver's
environment includes this Harbor SDK, set `data.train_harbor_dir` and
`data.val_harbor_dir` to local task paths as before. No model/optimizer changes
are required for the data format.

Task files must remain readable by the process that uploads or executes each
trial. On Ray, mount or stage the data at the same absolute path in every
relevant worker; generating a parquet on the head does not distribute the task
directories. An ossfs/PVC mount is simply another local directory for this API.

Use separate real train/validation data for training-quality evaluation. Reusing
this one task in both files only demonstrates the two input paths. Existing
parquet output files are not overwritten by the CLI. The automatic recipe cache
uses a new SDK format namespace; after editing task content, set
`data.harbor_overwrite=true` where supported or choose a new cache directory.

## Read an existing versioned dataset repository

For OSS, install the same pinned Harbor revision with `[oss,huggingface]` extras,
and register an **existing initialized repository** using its real Bucket,
prefix, and region:

```bash
.venv-harbor-data/bin/harbor dataset repo context add team \
  oss://my-bucket/datasets --region cn-hongkong

.venv-harbor-data/bin/python agentic/dataset/local_harbor.py \
  --dataset-ref team/benchmark@v1 --context team \
  --download-dir ./prepared/tasks-v1 --split train \
  --output ./prepared/repository-train.parquet
```

This calls `TrainingDataset.from_repo()`, materializes the complete task tree to
a new local directory, and retains `dataset`/`dataset_ref` in parquet metadata.
The SDK validates the repository archive when downloading. A directory of loose
objects on OSS is not automatically a Harbor versioned repository; mount or
download that directory first and use `--local-dir`. The configured SDK backend
must have working OSS credentials. This change does not introduce RRSA credential
rotation or assume that an arbitrary CLI profile supplies it.

Data preparation can run in the dedicated environment above even when the
training/execution image uses an older Harbor runtime. Direct directory
materialization inside that image requires the pinned training SDK there too;
check runtime compatibility before upgrading, and do not copy old runtime
overrides over the new Harbor installation. The SDK integration tests do not
establish a new end-to-end GPU training compatibility matrix.

## Check the example's solution and verifier locally

The example scripts accept alternate paths so their actual behavior can be
checked without writing `/app` or `/logs` on the host:

```bash
demo_dir="$(mktemp -d)"
TASK_ROOT="$demo_dir/app" sh tasks/hello-world/solution/solve.sh
TASK_ROOT="$demo_dir/app" VERIFIER_LOG_DIR="$demo_dir/verifier" \
  sh tasks/hello-world/tests/test.sh
cat "$demo_dir/verifier/reward.txt"
```

For a full Harbor execution, use a configured environment backend and the normal
`harbor run -p ./tasks/hello-world ...` command. The default paths in the task
scripts remain `/app` and `/logs/verifier`.
