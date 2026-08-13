# remote_agent

RL-train verl against **any** external, OpenAI-compatible agent — without
reimplementing that agent inside verl. The agent runs unchanged and points its
LLM client at a proxy; the proxy records the agent's LLM traffic (token ids +
logprobs) and verl reconstructs a training trajectory from that recording.
[Harbor](https://pypi.org/project/harbor/) is one pluggable runner; you can add
your own to drive any framework or subprocess.

## Verified against

| Component | Version / commit |
|-----------|-----------------|
| verl (upstream `origin/main`) | `0.9.0.dev` @ `535c4779` |
| recipe submodule | `release-remote-agent-rl` |
| Python | 3.12 |
| Ray | 2.54.0 |
| SGLang | 0.5.12 |
| transformers | < 5.6.0 |
| TransferQueue | 0.1.8 |
| Harbor (`alibaba/harbor` fork, branch `release-dev-e5e46809-rollout-server`) | 0.21.0 |

> **No core verl changes required.** All compatibility gaps are handled inside
> the recipe (see [Cluster deployment notes](#cluster-deployment-notes)).
>
> (*) transformers ≥ 5.6.0 has a bug in
> `transformers/integrations/flash_attention.py` where `s_aux` (an optional
> "learnable attention sink" tensor) is unconditionally `.to()`-ed without a
> `None` check, crashing models that don't use it (e.g. Qwen2). The Dockerfile
> pins `transformers<5.6.0`; if you must use 5.6.0+, patch line 84 to
> `s_aux=s_aux.to(query.dtype) if s_aux is not None else None`.

## Relationship to `recipe/agentic`

This recipe is the **decoupled, upstream-oriented refactor** of the
`recipe/agentic` core path (proxyserver + RemoteAgentLoop, ported in commit
`264e6d7`). The two are **parallel recipes with no dependency either way**:

- **`agentic`** — the production full-feature variant: standalone proxy mode
  (external K8s proxy via `PROXY_SERVER_URL`), disaggregated training
  (`agentic_disagg_main`), per-step timing (`timed_trainer`), job-submission
  SDK (`serversdk`), deep Harbor/ACK integration; config is injected into Ray
  workers via env vars.
- **`remote_agent`** (this recipe) — framework-agnostic: agent frameworks plug
  in through the runner registry (Harbor is one runner among many), pure-yaml
  config, Ray-actor proxy mode only.

Choose `remote_agent` for upstream contributions or wiring a new agent
framework; choose `agentic` for production training (standalone proxy, disagg,
timing).

## Architecture

```
                        verl PPO trainer (async rollout)
                                   │
                        ┌──────────▼───────────┐
                        │   RemoteAgentLoop     │  framework-agnostic core:
                        │  (agent_loop/…)       │  proxy session lifecycle +
                        └──────┬─────────┬──────┘  trajectory reconstruction +
                               │         │         tokenization
                    delegates  │         │  records via HTTP
                               │         │
                    ┌──────────▼──┐  ┌───▼─────────────────────────┐
                    │ External    │  │ OpenAI-compatible proxy     │
                    │ AgentRunner │  │ (Ray named actor)           │
                    └──────┬──────┘  │ records token_ids/logprobs, │
                           │         │ bridges to verl SGLang rollout│
                    runs the agent   └─────────────────────────────┘
                           │                     ▲
                           └── agent's LLM calls ┘
```

`RemoteAgentLoop` owns everything framework-independent: it registers a proxy
session, hands the external agent an OpenAI `base_url`, and — after the agent
finishes — rebuilds an `AgentLoopOutput` (prompt/response token ids, response
mask, logprobs) from the proxy's recording. Actually running the agent is
delegated to a pluggable `ExternalAgentRunner`. Two runners ship:

- **`HarborRunner`** (`runner/harbor/`) — runs a Harbor `Trial` in a sandbox
  (Docker or ACK K8s).
- **`ExampleRunner`** (`runner/example_runner.py`) — a dependency-free
  bring-your-own runner that spawns any subprocess agent, handing it the proxy
  URL via the standard `OPENAI_BASE_URL` / `OPENAI_API_KEY` env vars.

## How it works

Per rollout, `RemoteAgentLoop.run()`:

1. Discovers the proxy URL (a Ray named actor) and registers a unique session
   keyed by `trial_id`.
2. Hands the runner the agent-facing URL
   `http://{advertised_host}:{port}/{trial_id}/v1` and calls
   `runner.run(...)`; the external agent makes its LLM calls against that URL
   and the proxy records each turn (with retry/reset between attempts).
3. Fetches the recorded session, completes and deletes it, and reconstructs an
   `AgentLoopOutput` — LLM completions get `mask=1`, inter-turn tool/user
   messages get `mask=0`.
4. **Fails open**: `run()` never raises. On any error (or an empty recording)
   it returns a minimal but valid EOS-only output so a single bad rollout can
   never abort the batch.

## Config

`config/remote_agent_trainer.yaml` layers on top of verl's `ppo_trainer`. The
recipe-specific block:

```yaml
actor_rollout_ref:
  rollout:
    name: sglang                    # rollout backend
    mode: async
    log_prob_micro_batch_size_per_gpu: 1
    agent:
      default_agent_loop: remote_agent
      agent_loop_config_path: remote_agent/config/agent_loop.yaml
    remote_agent:
      proxy:
        advertised_host: "0.0.0.0"   # override with the externally-reachable host
        port: 0                      # 0 = auto-pick a free port
        tool_format: hermes
      retry:
        max_retries: 3
        retry_base_delay: 1.0
        poll_interval: 2.0
      task_path:
        roots: []                    # dirs searched for <root>/<instance_id>
        template: "{instance_id}"    # fallback path template
      runner:
        name: harbor                 # selects the ExternalAgentRunner
        kwargs:                      # opaque, passed straight to the runner
          agent_name: swe-agent
          environment_import_path: "harbor.environments.docker.docker:DockerEnvironment"

  actor:
    ppo_micro_batch_size_per_gpu: 1
  ref:
    log_prob_micro_batch_size_per_gpu: 1

# GRPO does not use a critic, but validate_config still requires a model path.
critic:
  model:
    path: ${actor_rollout_ref.model.path}
  ppo_micro_batch_size_per_gpu: 1
```

Key fields:

- **`proxy.advertised_host`** — an IP the external agent can reach. `0.0.0.0`
  is a placeholder; set it to the trainer node's externally-reachable host, or
  export `REMOTE_AGENT_ADVERTISED_HOST` (the env var wins). **On Ray clusters,
  pass this via the Hydra CLI override** (`actor_rollout_ref.rollout.remote_agent.proxy.advertised_host=<ip>`)
  so Ray worker processes inherit it from the config, not just the env.
- **`agent.agent_loop_config_path`** — points at `config/agent_loop.yaml` which
  declares the `remote_agent` agent loop class. verl resolves this relative to
  the project root (the directory containing both `verl/` and `remote_agent/`).
  Every `AgentLoopWorker` loads it; no recipe import is needed in workers.
- **`runner.name`** — which registered runner to use (`harbor`, `example`, or
  your own).
- **`runner.kwargs`** — an opaque dict handed to the runner's constructor; the
  core never inspects it.

## Adding your own runner

Subclass `ExternalAgentRunner`, implement `async run(...)`, and register it by
name:

```python
from remote_agent.runner.base import (
    AgentRunResult, AgentTask, ExternalAgentRunner, register_runner,
)

@register_runner("myagent")
class MyRunner(ExternalAgentRunner):
    async def run(self, *, task: AgentTask, agent_base_url: str,
                  sampling_params: dict, **kwargs) -> AgentRunResult:
        # point your agent's OpenAI client at agent_base_url, run it,
        # and translate the outcome into an AgentRunResult.
        ok = await run_my_agent(task, agent_base_url)
        status = "completed" if ok else "error"
        return AgentRunResult(status=status, rewards={"success": float(ok)})

    # optional: materialize a dataset if your runner owns a task format
    def build_dataset(self, data_cfg) -> tuple[str, str] | None:
        return None
```

`run()` returns an `AgentRunResult(status, rewards, error)` where `status` is
`"completed" | "failed" | "error"`. Optionally override `build_dataset(data_cfg)`
to return `(train_files, val_files)` parquet paths — `main.py` calls it up
front so the entry point never imports your framework.

To make the name resolvable, either add it to `_LAZY_MODULES` in
`runner/base.py` (so it is imported only when selected) or import the module
before use. Runners never touch tokens — the proxy records LLM traffic
out-of-band.

---

## Runbook: cluster deployment (KubeRay + E2B/ACK sandbox)

This runbook covers a full end-to-end deployment on a K8s cluster with
KubeRay, using the Harbor + swe-agent runner with E2B or ACK sandbox pods.
Every `<placeholder>` should be replaced with your environment's value.

### 1. Prerequisites

- **verl** installed in editable mode (`pip install -e .[sglang]`) at the
  version listed in [Verified against](#verified-against).
- **remote_agent recipe** checked out under the verl project root (so that
  `verl/` and `remote_agent/` are siblings).
- **Harbor** installed from the `alibaba/harbor` fork, branch
  `release-dev-e5e46809-rollout-server` (version 0.21.0).
  Install from the fork:
  ```bash
  git clone -b release-dev-e5e46809-rollout-server https://github.com/alibaba/harbor.git
  pip install --ignore-installed ./harbor
  pip install kubernetes kubernetes_asyncio e2b   # ACK + E2B sandbox support
  ```
- **TransferQueue** (`pip install TransferQueue==0.1.8`) — required by verl's
  7.1 TaskRunner (async rollout pattern).
- **transformers < 5.6.0** — the Dockerfile pins this; 5.6.0+ has a
  `flash_attention s_aux` bug that crashes Qwen2.
- A Docker image containing verl + remote_agent + harbor + deps. The Dockerfile
  at `remote_agent/docker/Dockerfile` is a reference; it copies
  `sitecustomize.py` to `/workspace/` and sets `PYTHONPATH=/workspace`.
- A K8s cluster with:
  - KubeRay operator (RayCluster CRD)
  - GPU nodes with appropriate tolerations
  - A `PersistentVolumeClaim` for model weights (read-only mount)
  - A `PersistentVolumeClaim` for task data (read-write mount, RWX)
  - **For E2B mode**: sandbox-manager + sandbox-gateway deployed (e.g. in
    `sandbox-system` namespace), and a `SandboxSet` CRD with pre-provisioned
    sandbox pods matching the task images.
  - **For ACK mode**: an image-pull secret for the sandbox images.

### 2. RayCluster manifest

Save as `raycluster.yaml` and adjust the placeholders:

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: <raycluster-name>
  namespace: <namespace>
  labels:
    app: <raycluster-name>
spec:
  rayVersion: "2.54.0"
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
      num-cpus: "0"           # proxy actor uses num_cpus=0 (see deployment notes)
      num-gpus: "0"
    template:
      spec:
        imagePullSecrets:
        - name: <image-pull-secret>
        tolerations:           # cluster-specific GPU tolerations
        - key: <toleration-key>
          operator: Exists
          effect: NoSchedule
        containers:
        - name: ray-head
          image: <verl-image>
          imagePullPolicy: Always
          env:
          - name: NCCL_DEBUG
            value: "WARN"
          - name: PYTHONPATH     # so sitecustomize.py is auto-imported
            value: "/workspace:$(PYTHONPATH)"
          ports:
          - containerPort: 6379
          - containerPort: 8265
          - containerPort: 10001
          resources:
            requests: { cpu: "4", memory: "16Gi" }
            limits:   { cpu: "16", memory: "32Gi" }
          volumeMounts:
          - name: shm
            mountPath: /dev/shm
          - name: models
            mountPath: /mnt/models
          - name: data
            mountPath: /mnt/data
        volumes:
        - name: shm
          emptyDir: { medium: Memory, sizeLimit: "8Gi" }
        - name: models
          persistentVolumeClaim: { claimName: <models-pvc> }
        - name: data
          persistentVolumeClaim: { claimName: <data-pvc> }
  workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 1
    minReplicas: 1
    maxReplicas: 1
    rayStartParams:
      num-gpus: "<n-gpus>"     # e.g. "8"
    template:
      spec:
        imagePullSecrets:
        - name: <image-pull-secret>
        tolerations:
        - key: <toleration-key>
          operator: Exists
          effect: NoSchedule
        containers:
        - name: ray-worker
          image: <verl-image>
          imagePullPolicy: Always
          env:
          - name: GLOO_SOCKET_IFNAME
            value: "eth0"
          - name: PYTHONPATH
            value: "/workspace:$(PYTHONPATH)"
          resources:
            requests: { cpu: "32", memory: "256Gi", nvidia.com/gpu: "<n-gpus>" }
            limits:   { cpu: "64", memory: "512Gi", nvidia.com/gpu: "<n-gpus>" }
          volumeMounts:
          - name: shm
            mountPath: /dev/shm
          - name: models
            mountPath: /mnt/models
          - name: data
            mountPath: /mnt/data     # workers need task data for harbor runner
        volumes:
        - name: shm
          emptyDir: { medium: Memory, sizeLimit: "128Gi" }
        - name: models
          persistentVolumeClaim: { claimName: <models-pvc> }
        - name: data
          persistentVolumeClaim: { claimName: <data-pvc> }
---
# RBAC: allow the proxy / agent loop to create sandbox pods
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: <raycluster-name>
  namespace: <namespace>
rules:
- apiGroups: [""]
  resources: ["pods", "pods/exec", "pods/log"]
  verbs: ["get", "list", "create", "delete", "patch", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: <raycluster-name>
  namespace: <namespace>
subjects:
- kind: ServiceAccount
  name: default
  namespace: <namespace>
roleRef:
  kind: Role
  name: <raycluster-name>
  apiGroup: rbac.authorization.k8s.io
```

Key points:
- **`PYTHONPATH=/workspace:$(PYTHONPATH)`** — ensures `sitecustomize.py`
  (shipped at `remote_agent/compat/sitecustomize.py`) is auto-imported in every
  python process, installing the config-compat shim and registering the
  `remote_agent` agent loop. Alternatively, copy `sitecustomize.py` to
  `/workspace/sitecustomize.py` in the image build step.
- **`num-cpus: "0"`** on the head — the proxy actor requests `num_cpus=0` so
  it can be scheduled on the head even with zero CPU resources.
- **Data PVC on workers** — harbor's `ACKEnvironment` creates sandbox pods in
  the same namespace; task data must be visible on both head and workers.
- **RBAC** — the `default` service account needs pod create/delete/exec
  permissions for harbor to spin up sandbox pods.

### 3. Prepare task data

Task directories follow Harbor's layout: each `instance_id` is a subdirectory
containing `task.toml` + `instruction.md`.

```bash
# On a pod that can access the data PVC
mkdir -p /mnt/data/<task-set>/train /mnt/data/<task-set>/val
# Copy or symlink task subdirs into train/ and val/
# Example: SWE-bench verified
cp -r /mnt/data/swe-bench-verified/<instance-id> /mnt/data/<task-set>/train/
```

### 4. Launch training

The recipe ships a launch script that handles env-var-driven configuration.
Two sandbox modes are supported:

**E2B mode** (recommended — uses sandbox-manager/gateway HTTP API):

```bash
export MODEL_PATH=/mnt/models/<model-name>
export TRAIN_TASKS=/mnt/data/<task-set>/train
export VAL_TASKS=/mnt/data/<task-set>/val
export SANDBOX_MODE=e2b
export SANDBOX_SET=<sandboxset-name>          # pre-created SandboxSet
# E2B env vars (set as pod env in RayCluster yaml):
#   E2B_API_KEY=<admin-key>
#   E2B_API_URL=http://sandbox-manager.sandbox-system:8080
#   E2B_SANDBOX_URL=http://sandbox-gateway.sandbox-system:7788
#   E2B_VALIDATE_API_KEY=false

bash remote_agent/scripts/train_harbor_sweagent.sh [hydra overrides...]
```

**ACK mode** (direct K8s pod creation, needs RBAC):

```bash
export SANDBOX_MODE=ack
export SANDBOX_NAMESPACE=<namespace>
export SANDBOX_IMAGE_PULL_SECRET=<sandbox-image-pull-secret>
export SANDBOX_MEMORY_LIMIT_MULT=4

bash remote_agent/scripts/train_harbor_sweagent.sh [hydra overrides...]
```

The script:
- Sets `REMOTE_AGENT_ADVERTISED_HOST` to the head pod's IP (via `hostname -i`).
  **Note**: the V1 `_ProxyTaskRunner` actor overrides this at runtime with
  `ray.util.get_node_ip_address()` (the worker's IP), because the proxy runs
  inside the TaskRunner actor on a worker node, not on the head.
- Derives `MODEL_NAME` as `openai/$(basename $MODEL_PATH)` — the `openai/`
  prefix is required because swe-agent uses litellm.
- Passes `advertised_host` via Hydra override (not env var) so Ray worker
  processes inherit it from the serialized config.
- Passes `model_name` via `++` Hydra override (it's not in the yaml struct).

**V1 TaskRunner**: upstream verl (≥ `535c4779`) uses `@ray.remote
TaskRunnerV1` — the recipe defines `_ProxyTaskRunner` (a `@ray.remote` actor
that replicates `TaskRunnerV1.run()` but injects `start_proxy_server`
between `trainer.init()` and `trainer.fit()`). The recipe calls
`run_ppo(config, task_runner_class=_ProxyTaskRunner)`. A `ray.init(address="auto")`
is called before `run_ppo` to connect to the existing KubeRay cluster.

**Shared PVC for harbor cache**: the V1 TaskRunner actor runs on a worker
node, so `data.harbor_cache_dir` must point to a shared PVC path (not local
filesystem). Example: `data.harbor_cache_dir=/mnt/data/.harbor_cache`.

Common Hydra overrides for smoke tests:

```bash
  actor_rollout_ref.rollout.agent.num_workers=1    # serial sandbox (less memory pressure)
  data.train_batch_size=<n-tasks>                  # must be divisible by num_workers × n_gpus
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.actor.ppo_mini_batch_size=<n-tasks>
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.max_model_len=10240
  data.max_prompt_length=8192
  data.max_response_length=1024
  trainer.total_training_steps=1
  trainer.logger=[console]
  trainer.val_before_train=False
  trainer.test_freq=-1
  trainer.save_freq=-1
```

### 5. Verify the data flow

After launch, check the training log at each stage:

| Stage | Log pattern | Meaning |
|-------|-------------|---------|
| Dataset materialized | `materialized dataset: train=...parquet val=...parquet` | Harbor runner built parquets from task dirs |
| FSDP weights loaded | `Loading weights: 100%` | Actor model loaded on GPUs |
| SGLang server ready | `Capturing num tokens` / `SGLang http server` | Rollout engine initialized |
| Proxy started | `Proxy server actor created at http://<ip>:<port>` | OpenAI-compatible proxy is live (inside the TaskRunner actor) |
| Rollout started | `Training Progress: 0%` | First step's rollout phase began |
| LLM traffic | `[GENERATE] prompt_ids length=<n>` | swe-agent called the proxy; real LLM generation |
| Trajectory rebuilt | `Successfully converted trajectory to ATIF format` | Harbor converted swe-agent's `.traj` to training format |
| Training step | `Training Progress: 100%` | Step completed (advantage + actor update) |

### 6. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `TypeError: RolloutConfig.__init__() got an unexpected keyword argument 'remote_agent'` | `sitecustomize.py` not auto-imported | Ensure `PYTHONPATH` includes the directory containing `sitecustomize.py` |
| `Agent loop remote_agent not registered` | `agent_loop_config_path` not resolved | Check that `config/agent_loop.yaml` exists and the path is relative to the verl project root |
| `ActorUnschedulableError` for proxy actor | Head node has `num-cpus: 0` | Proxy actor requests `num_cpus=0` (already in recipe); ensure head `rayStartParams.num-cpus` is `"0"` |
| `ValueError: Model name must be specified for SWE-agent` | `model_name` not set | Script auto-derives it; if overriding, use `++...model_name=openai/<name>` |
| `BadRequestError: LLM Provider NOT provided` | litellm needs `openai/` prefix | Script prepends `openai/`; don't override with a bare model name |
| `NonZeroAgentExitCodeError: exit 100/137` | Sandbox apt-get failed | Check sandbox pod resources; use `SANDBOX_MEMORY_LIMIT_MULT` to increase memory; verify sandbox image's apt sources are reachable |
| `AttributeError: 'NoneType' object has no attribute 'to'` in flash_attention | transformers ≥ 5.6.0 bug | Pin `transformers<5.6.0` or patch `flash_attention.py` line 84 |
| `ValueError: Total available GPUs 0` | `run_ppo` didn't connect to KubeRay cluster | Recipe calls `ray.init(address="auto")` before `run_ppo`; ensure head pod has `RAY_ADDRESS=127.0.0.1:6379` |
| `MissingExtraError: The 'e2b' package is required` | `e2b` SDK not installed | `pip install e2b` in the image or on the pod |
| `OSError: [Errno 107] Transport endpoint is not connected` | OSS FUSE mount dropped | Restart the pod or use a more stable storage backend |
| `Connection error` from swe-agent (litellm) | `advertised_host` points to head, not worker | V1 TaskRunner overrides `advertised_host` at runtime; ensure the override is in `_ProxyTaskRunner.run()` |
| `ReadTimeout` from E2B sandbox claim | SandboxSet replicas < concurrent tasks | Scale `SandboxSet.spec.replicas` to match `train_batch_size` |
| `FileNotFoundError: ...harbor_cache/train-*.parquet` | V1 actor on different pod than driver | Set `data.harbor_cache_dir` to a shared PVC path |

---

## Cluster deployment notes (zero core-verl changes)

Core verl's `RolloutConfig` does not know about the recipe-added
`rollout.remote_agent` section, and Ray worker processes start without
importing this recipe. Both gaps are bridged **inside the recipe**, no core
verl edits required:

- **Agent-loop registration** uses verl's own mechanism:
  `config/agent_loop.yaml` declares `name: remote_agent` + its `_target_`, and
  `remote_agent_trainer.yaml` defaults `rollout.agent.agent_loop_config_path`
  to it. Every `AgentLoopWorker` loads that file and resolves the class lazily
  via hydra — no recipe import needed in the worker beforehand.
- **Config compatibility**: `remote_agent/compat/` ships a shim (`install()`)
  that makes `verl.utils.config.omega_conf_to_dataclass` drop recipe-added keys
  before conversion (the original config object is never mutated).
  `remote_agent/main.py` installs it in the driver and injects
  `worker_process_setup_hooks=["remote_agent.compat:setup_worker"]` into the
  Ray runtime env.
- On KubeRay clusters verl overrides the per-actor runtime env when creating
  WorkerDicts (`verl/single_controller/ray/base.py`), which drops that
  job-level hook; `remote_agent/compat/sitecustomize.py` handles this by
  auto-installing the shim when `verl.utils.config` is first imported in any
  python process. Copy it to a `PYTHONPATH` directory in the Ray pods (e.g.
  `/workspace/sitecustomize.py`) or set `PYTHONPATH` to include the
  `remote_agent/compat/` directory.

Also note: the proxy actor is pinned to the head node with `num_cpus=0` so it
schedules even when the head runs with `num-cpus: 0`.

## v1 limitations

- **Proxy mode**: Ray-actor proxy only — the proxy runs as a named actor
  inside the `_ProxyTaskRunner` on a worker node. There is no
  standalone/external proxy mode.
- **Harbor runner**: local Trials only. A remote Harbor-HTTP runner (submitting
  trials to a separate Harbor service) is a planned follow-up.
- **transformers compatibility**: requires `transformers < 5.6.0` (Dockerfile
  pins this) or a one-line patch to `flash_attention.py`
  (see [Verified against](#verified-against)).
- **V1 TaskRunner**: upstream verl ≥ `535c4779` uses `@ray.remote TaskRunnerV1`;
  the recipe's `_ProxyTaskRunner` adapts to this API. If upstream changes the
  V1 `run()` method signature, the recipe must be updated accordingly.
