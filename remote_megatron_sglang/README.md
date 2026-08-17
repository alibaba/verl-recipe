# Remote backend: external Megatron (train) + external SGLang (inference)

Run verl RL training where **both** engines live outside verl's Ray cluster:

- **Training** — Megatron-LM in a Kubeflow **PyTorchJob**.
- **Inference** — SGLang in a **RoleBasedGroup** (RBG).
- **verl** — a **CPU-only orchestrator** (PPO/GRPO, data, advantage, reward,
  metrics). It owns no GPUs.

This is built on the generic `RemoteBackend` abstraction ported from upstream
PR #6422 (`verl/remote_backend/`). See `DESIGN.md` for the full design and the
rationale behind each decision.

> **Status: e2e-validated on ACK (H20) with the V1-trainer port in
> `v1_trainer.py`** — Qwen2.5-0.5B / GRPO / 2 training steps: external SGLang
> generation → forwarder → external Megatron `compute_log_prob` / `update_actor`
> → `nccl_http` weight sync (290 tensors / ~0.8 s per step) all verified; rollout
> vs training log-prob Pearson ≈ 0.999. Runs on both the internal branch and
> **stock upstream verl main (verified on 0.10.0.dev via the
> `remote_backend_compat` fallback — same e2e, same metrics)**. Zero verl-core
> changes beyond that. Cluster/version-specific spots are still marked
> `# VALIDATE`.

## Architecture

```
verl driver (CPU, 0-GPU pool) ── RemoteBackendTrainer
   │ 1 CPU forwarder worker (MegatronSGLangForwarderWorker)
   ▼
MegatronSGLangBackend (adapter)
   ├── HTTP ─► Megatron PyTorchJob  (train_server wraps verl MegatronEngine)
   │            compute_log_prob / update_actor / save_checkpoint
   └── HTTP ─► SGLang RoleBasedGroup (generation via gateway_url;
                weight-update endpoints for sync)
        └───── weight sync: Megatron ──(pluggable transport)──► SGLang
```

## Pieces

| Path | Role |
|------|------|
| `backend.py` | `MegatronSGLangBackend(RemoteBackend)` — registered `megatron_sglang`. |
| `v1_trainer.py` | **V1 integration**: `RemoteMegatronSGLangTrainer` registered as trainer mode `remote_megatron_sglang`; CPU-only pool + bound forwarder worker. |
| `forwarder_worker.py` | CPU-only forwarder; the verl-side actor/rollout worker. |
| `train_client.py` / `infer_client.py` | HTTP clients (train server / SGLang weight ops). |
| `weight_sync/` | Pluggable transports: `nccl_http` (default), `store`, `mooncake`. |
| `checkpoint_engine.py` | verl `CheckpointEngine` for **external-SGLang-only** weight sync (former `recipe/external_sglang`). |
| `server/train_server.py` | Runs **inside** the PyTorchJob; wraps verl `MegatronEngine`. |
| `server/protocol.py` | The stable HTTP contract (endpoints + tensor codec). |
| `config/megatron_sglang.yaml` | Backend Hydra config. |
| `k8s/` | `pytorchjob-megatron.yaml`, `rbg-sglang.yaml`, `sglang-external*.yaml`. |
| `test/` | Registry + protocol + backend tests; `test/external_sglang/` holds the external-SGLang validation suite (2×TP4 / PD-disagg / NCCL-HTTP / benchmarks). |
| `train_agentic_ack_external.sh` / `train_gsm8k_external.sh` | External-SGLang training drivers (agentic + one-step-off-policy). |

## V1 integration (how it actually runs)

Only the `RemoteBackend` scaffolding is present in verl core — the swap points
this recipe originally assumed were **never landed**:

| Piece | State |
|-------|-------|
| `verl/remote_backend/{base,__init__,trainer,worker_utils}.py` | **present** |
| `verl/single_controller/ray/base.py` — CPU-only (0-GPU) resource pools | **present** (`use_gpu` plumbed natively) |
| `verl/trainer/main_ppo.py` — swap worker + trainer when `remote_backend` set | **missing** |
| `verl/trainer/config/ppo_trainer.yaml` — `trainer.remote_backend` field + `remote_backend` config group | **missing** |
| `verl/trainer/ppo/ray_trainer.py` — `wg_kwargs` / `use_gpu` / `max(n_gpus,1)` | **missing** |

`main_ppo` dispatches through `TaskRunnerV1` →
`get_trainer_cls(config.trainer.v1.trainer_mode)`, and `RemoteBackendTrainer`
subclasses the deprecated `RayPPOTrainer`. So the integration lives in
`v1_trainer.py` (zero verl-core changes):

* `@register_trainer("remote_megatron_sglang")` — selected with
  `trainer.v1.trainer_mode=remote_megatron_sglang`. Subclasses `PPOTrainerSync`
  so weight sync fires at the stock hook points (V1 pins
  `checkpoint_engine.backend="naive"`, and that path reduces to
  `actor_wg.update_weights(...)` — i.e. the forwarder's transport).
* `CpuOnlyResourcePoolManager` — one `use_gpu=False` bundle for the single
  CPU forwarder process (the stock manager hardcodes `use_gpu=True`).
* `bind_forwarder_worker()` — closures `main_config` / `backend_handle` into a
  forwarder subclass because V1's `_setup` builds the actor
  `RayClassWithInitArgs` with a fixed kwarg set.
* The `remote_backend` Hydra group is *appended* (`+remote_backend=...`), not
  declared in `ppo_trainer.yaml`'s defaults.

The registration rides `VERL_USE_EXTERNAL_MODULES=recipe.remote_megatron_sglang.register`
(imported by `import verl` in the driver, TaskRunnerV1 actor and every Ray
worker; single-node local Ray inherits the env natively).

The only verl-core symbol this recipe still needs beyond stock upstream main
is `verl.remote_backend` (the PR #6422 abstraction, carried on our internal
branch). `remote_backend_compat.py` is an inline fallback with the same two
classes; every import site prefers `verl.remote_backend` when present and
falls back otherwise, so the recipe runs on both trees unchanged.

## Migrating the V1 integration to other trainers

`RemoteMegatronSGLangTrainer` rides `trainer.v1.trainer_mode=sync`. Everything
outside `v1_trainer.py` (backend / forwarder / HTTP clients / weight-sync
transports / `train_server.py` / the external rollout proxy) is
trainer-agnostic — a port only touches a copy of `v1_trainer.py`. Work
through the steps below in order; each has a **Check** (how to tell you're
affected), a **Fix** (what to change), and a **Verify** (how to confirm).

### Step 0 — fork the trainer (required for every port)

1. Copy `v1_trainer.py` → `v1_trainer_<mode>.py`; rename the class and the
   `@register_trainer("<new-mode-name>")` string.
2. Swap the base class to the target trainer
   (`PPOTrainerColocateAsync` / `PPOTrainerSeparateAsync`).
3. Keep `__init__`, `_validate_config`, `_init_resource_pool_mgr` and
   `on_train_end` unchanged — they don't depend on the mode.
4. In `register.py`, import the new module next to `_v1_trainer` so it
   self-registers.
5. Verify it resolves:

```bash
VERL_USE_EXTERNAL_MODULES=recipe.remote_megatron_sglang.register python -c \
  "import verl; from verl.trainer.ppo.v1 import get_trainer_cls; print(get_trainer_cls('<new-mode-name>'))"
```

### Step 1 — port to `colocate_async`

Do Step 0 with base `PPOTrainerColocateAsync`, then:

- [ ] **Hooks — no override needed.**
      Check: the base already calls `checkpoint_manager.update_weights` in
      `on_init_end`/`on_step_end` and `abort_replicas` + `sleep_replicas` in
      `on_sample_end`.
      Fix: none. Against the external proxy abort/sleep are no-ops (the
      external SGLang never frees GPUs); the transport's trailing
      `/flush_cache` is the only cleanup.
- [ ] **`get_llm_client`** now returns `FullyAsyncLLMServerClient` over the
      proxy actor.
      Check: interface-compatible with the proxy.
      Fix: none, but re-run the e2e suite — partial-rollout / abort paths
      were only exercised in sync mode.
- [ ] **Off-policy bound.**
      Fix: set `trainer.v1.sampler.max_off_policy_threshold`; staleness is
      bounded by the replay buffer, not by weight-sync timing.
- [ ] **Verify:** driver log shows `timing_s/update_weights` every step and
      `rollout_actor_probs_pearson_corr ≈ 0.999`.

### Step 2 — port to `separate_async`

Do Step 0 with base `PPOTrainerSeparateAsync`, then fix three conflicts in
this order:

**2.1 The `naive`-backend assert.**
- Check: `__init__` asserts
  `rollout.checkpoint_engine.backend != "naive"` → startup fails.
- Fix: override `__init__` with a copy of the base minus that one assert
  (keep the `train_batch_size == parameter_sync_step * ppo_mini_batch_size`
  and rollout nnodes/gpus asserts — they still apply). The pinned `naive`
  backend is exactly the channel that forwards `update_weights` to the
  external transport; keep it `naive`.

**2.2 The standalone rollout pool.**
- Check: `_setup` builds `standalone_server_manager` +
  `standalone_checkpoint_manager`, and `get_llm_client()` would serve from a
  *second* proxy replica. With an external SGLang there is nothing to stand
  up.
- Fix: skip the middle `_setup` and neuter the mode-switch hooks:

```python
def _setup(self):
    super(PPOTrainerSeparateAsync, self)._setup()  # skip the standalone block
    self.current_mode = None

def get_llm_client(self):
    return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

def on_init_end(self):
    self.checkpoint_manager.update_weights(self.global_steps)

def on_step_end(self):
    self._pending_sync_metrics = self.checkpoint_manager.update_weights(self.global_steps)

def on_validate_begin(self): pass
def on_sample_begin(self): pass
def on_sample_end(self): pass
```

**2.3 Decoupled PPO (`parameter_sync_step > 1`).**
- Check: does the run use `parameter_sync_step > 1`? The base
  `_compute_old_log_prob` then calls `save_model_to_cpu` /
  `restore_model_from_cpu` / `clear_cpu_model`, which
  `MegatronSGLangForwarderWorker` does not implement.
- Fix (bypass mode, `parameter_sync_step=1`): nothing to do — rollout
  log-probs are reused.
- Fix (decoupled mode): add the three methods to the forwarder, forwarding
  to the train server's `/save_weights` and `/load_checkpoint` endpoints.
  **UNVALIDATED** — run the e2e suite before trusting it.

**2.4 Verify:** same checklist as Step 1.

### Step 3 — port to a non-V1 trainer

1. Check how the trainer is selected: `register_trainer` exists only on the
   V1 path (`TaskRunnerV1` → `get_trainer_cls`).
2. If `trainer.use_v1=false`: the only shipped adapter is
   `verl/remote_backend/trainer.RemoteBackendTrainer` — internal branch only,
   subclasses the deprecated `RayPPOTrainer`, and is NOT part of the compat
   fallback, so it does not exist on upstream main.
3. Recommendation: stay on V1 and reuse this recipe's trainer-agnostic
   pieces (the train server needs no change either way).

### Invariants — check after any port

- [ ] exactly ONE CPU forwarder (`trainer.n_gpus_per_node=1`,
      `trainer.nnodes=1`) — the external Megatron owns the training
      parallelism;
- [ ] verl-side rollout parallelism 1×1×1 and `rollout.name=megatron_sglang`,
      so `LLMServerManager` builds exactly one external proxy;
- [ ] `MEGATRON_SGLANG_ENDPOINTS` set wherever the driver runs;
- [ ] the `remote_backend` Hydra group *appended* (`+remote_backend=...`);
- [ ] the trainer-side worker group stays CPU-only (`use_gpu=False` pools —
      never let a port re-introduce GPU pools);
- [ ] weight sync still reaches `actor_wg.update_weights(...)` — confirm via
      `timing_s/update_weights` in the step metrics;
- [ ] the train server is untouched.

## Deploy

```bash
# 1. Install operators (once):
#    - Kubeflow training-operator: https://github.com/kubeflow/training-operator
#    - RoleBasedGroup: https://github.com/sgl-project/rbg
#      kubectl apply --server-side -f \
#        https://raw.githubusercontent.com/sgl-project/rbg/main/deploy/kubectl/manifests.yaml
kubectl apply -f k8s/pytorchjob-megatron.yaml
kubectl apply -f k8s/rbg-sglang.yaml

# 2. Launch verl (CPU-only driver):
TRAIN_ENDPOINT=http://megatron-train-master:8000 \
SGLANG_ENDPOINT=http://sglang-rbg-leader:30000 \
WEIGHT_SYNC_TRANSPORT=nccl_http \
bash train_gsm8k_remote.sh
```

## Runbook (validated end-to-end)

Everything below was executed against a real cluster (ACK, 2×8×H20,
verl 0.9.0.dev internal branch AND stock upstream main 0.10.0.dev — identical
results). Substitute the placeholders:

| Placeholder | Meaning |
|-------------|---------|
| `<verl-image>` | image with verl + megatron.core + mbridge + sglang (e.g. `your-registry/verl:<tag>`); must also ship this recipe as `recipe.remote_megatron_sglang` on `PYTHONPATH` |
| `<image-pull-secret>` | Secret with registry credentials |
| `<models-pvc>` / `<dataset-pvc>` | PVCs holding the HF model (e.g. `Qwen2.5-0.5B-Instruct`) and the RL parquet dataset |
| `<gpu-node-a>` / `<gpu-node-b>` | GPU node names |
| `<model>` | model directory name under the models PVC mount |

### 0. Prerequisites

```bash
kubectl get crd | grep -E 'pytorchjobs.kubeflow.org|rolebasedgroups.workloads.x-k8s.io'
# missing RBG? install the operator (see Deploy), then:
kubectl wait deploy/rbgs-controller-manager -n rbgs-system --for=condition=available --timeout=5m
```

### 1. External SGLang (RoleBasedGroup, TP=1)

One `leader` role, one GPU. See `k8s/rbg-sglang.yaml` for the general shape;
the validated minimal instance:

```yaml
apiVersion: workloads.x-k8s.io/v1alpha1
kind: RoleBasedGroup
metadata: { name: sglang-rbg, namespace: default }
spec:
  roles:
    - name: leader
      replicas: 1
      workload: { apiVersion: apps/v1, kind: StatefulSet }
      template:
        spec:
          imagePullSecrets: [{ name: <image-pull-secret> }]
          containers:
            - name: sglang
              image: <verl-image>
              command: ["/bin/bash", "-c"]
              args: ["python3 -m sglang.launch_server --model-path /mnt/models/<model> --tp 1 --host 0.0.0.0 --port 30000 --mem-fraction-static 0.6 --trust-remote-code"]
              ports: [{ containerPort: 30000, name: http }]
              env:
                - { name: NCCL_SOCKET_IFNAME, value: eth0 }
                - { name: GLOO_SOCKET_IFNAME, value: eth0 }
              resources: { limits: { nvidia.com/gpu: "1" }, requests: { nvidia.com/gpu: "1" } }
              volumeMounts: [{ mountPath: /mnt/models, name: models }]
          volumes: [{ name: models, persistentVolumeClaim: { claimName: <models-pvc> } }]
---
apiVersion: v1
kind: Service
metadata: { name: sglang-rbg-leader, namespace: default }
spec:
  selector: { rbg.workloads.x-k8s.io/group-name: sglang-rbg, rbg.workloads.x-k8s.io/role-name: leader }
  ports: [{ name: http, port: 30000, targetPort: 30000 }]
```

Wait until `curl http://sglang-rbg-leader:30000/health_generate` (from any pod)
succeeds, then sanity-check generation with `/generate`.

### 2. Megatron train server (PyTorchJob, 1 GPU)

Export a full verl PPO config first — the server reads only
`actor_rollout_ref`, but the export keeps it consistent with the driver:

```bash
python -m verl.trainer.main_ppo --cfg job --resolve \
  model_engine=megatron \
  actor_rollout_ref.model.path=/mnt/models/<model> \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
  actor_rollout_ref.actor.megatron.vanilla_mbridge=true \
  data.train_files=/mnt/data/train.parquet data.val_files=/mnt/data/test.parquet \
  algorithm.adv_estimator=grpo > train_config.yaml
kubectl create configmap megatron-train-config --from-file=train_config.yaml=train_config.yaml
```

Then the PyTorchJob (key points: a **custom pod label** for the Service
selector — operator label styles drift between `job-name`/`training.kubeflow.org/*`;
`POD_IP` for the NCCL rendezvous; the config as a plain file mount):

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata: { name: megatron-train, namespace: default }
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never
      template:
        metadata: { labels: { app: megatron-train } }   # stable selector
        spec:
          imagePullSecrets: [{ name: <image-pull-secret> }]
          nodeSelector: { kubernetes.io/hostname: <gpu-node-a> }
          containers:
            - name: pytorch
              image: <verl-image>
              command: ["/bin/bash", "-c"]
              args: ["python -m recipe.remote_megatron_sglang.server.train_server --config /config/train_config.yaml --port 8000"]
              env:
                - { name: PYTHONPATH, value: /workspace }
                - { name: POD_IP, valueFrom: { fieldRef: { fieldPath: status.podIP } } }
                - { name: NCCL_SOCKET_IFNAME, value: eth0 }
                - { name: GLOO_SOCKET_IFNAME, value: eth0 }
              ports: [{ containerPort: 8000, name: http-rl }]
              resources: { limits: { nvidia.com/gpu: "1" }, requests: { nvidia.com/gpu: "1" } }
              volumeMounts:
                - { mountPath: /config, name: cfg }
                - { mountPath: /mnt/models, name: models }
                - { mountPath: /dev/shm, name: shm }
          volumes:
            - { name: cfg, configMap: { name: megatron-train-config } }
            - { name: models, persistentVolumeClaim: { claimName: <models-pvc> } }
            - { name: shm, emptyDir: { medium: Memory, sizeLimit: 24Gi } }
---
apiVersion: v1
kind: Service
metadata: { name: megatron-train-master, namespace: default }
spec:
  selector: { app: megatron-train }        # the custom label, not operator labels
  ports: [{ name: http-rl, port: 8000, targetPort: 8000 }]
```

Wait for `curl http://megatron-train-master:8000/health` →
`{"status": "ok", ...}` (model load takes ~1 min for 0.5B).

### 3. CPU-only verl driver

Any 0-GPU pod with `<verl-image>` (add tolerations/nodeSelector to land it on
`<gpu-node-b>` if plain CPU nodes are full), PVCs mounted, then:

```bash
export VERL_USE_EXTERNAL_MODULES=recipe.remote_megatron_sglang.register
export MEGATRON_SGLANG_ENDPOINTS=http://sglang-rbg-leader:30000
TRAIN_ENDPOINT=http://megatron-train-master:8000 \
SGLANG_ENDPOINT=http://sglang-rbg-leader:30000 \
WEIGHT_SYNC_TRANSPORT=nccl_http \
MODEL_PATH=/mnt/models/<model> \
TRAIN_FILES=/mnt/data/train.parquet VAL_FILES=/mnt/data/test.parquet \
bash recipe/remote_megatron_sglang/train_gsm8k_remote.sh
```

`trainer.n_gpus_per_node=1 trainer.nnodes=1` in the script is a *forwarder
process count*, not GPUs — the driver's Ray pool is CPU-only.

### 4. Verify

1. **Config layer**: the driver resolves `+remote_backend=megatron_sglang`
   and logs the trainer mode `remote_megatron_sglang`.
2. **Generation**: `timing_s/gen` in the step metrics (external SGLang served
   the rollout).
3. **HTTP contract**: `timing_s/old_log_prob` / `timing_s/update_actor` — the
   forwarder reached the train server. For a direct probe, POST a *packed*
   (nested THD) batch to `/compute_log_prob`; dense `[B, L]` batches fail the
   megatron forward when `use_remove_padding=True`.
4. **Weight sync**: `timing_s/update_weights` ≈ 1 s/step for 0.5B; the train
   server log shows `weight-sync NCCL group formed: world_size=2`, and the
   SGLang log shows `POST /update_weights_from_distributed ... 200 OK` per
   chunk plus a trailing `flush_cache` 200.
5. **Policy consistency**: `rollout_actor_probs_pearson_corr ≈ 0.999` — SGLang
   generated with the same weights Megatron trains.

### Troubleshooting

| Symptom | Cause / fix |
|---------|------------|
| `Could not override 'remote_backend'` | missing `+` prefix — the group is appended, not declared in `ppo_trainer.yaml` |
| train server: `assert lr_decay_steps > 0` | config exported without the driver-side `optim.total_training_steps` injection — re-export with a current verl, or set `actor.optim.total_training_steps` explicitly (the server also self-injects a fallback) |
| train server: `ModuleNotFoundError: megatron.bridge` | image lacks megatron-bridge — set `actor.megatron.vanilla_mbridge=true` (uses verl's built-in bridge) |
| `KeyError: 'loss_mask'` / `nested_tensor must be nested` | posted batch is dense; send packed THD (`agent_loop_tq` does this on the driver path) |
| Service endpoints `<none>` | (a) `kubectl patch` merges selectors and keeps stale keys — delete+recreate the Service; (b) operator label style drifted — select on a custom pod label instead |
| PyTorchJob pod stuck `Error` forever, new job creates nothing | orphaned pod (no ownerRef) blocks the recreate — `kubectl delete pod` it |
| configMap file not found at the mount | the key IS the filename (`--from-file=rms-pkg.tgz=...` → `/pkg/rms-pkg.tgz`) |
| driver dies at *final* validation with `DataLoader worker killed` | CPU pod OOM during teardown — harmless for training; give the driver pod more memory or set `trainer.val_before_train=false` and skip final val |
| SGLang crash on weight update (`is_fully_idle` assert) | never let flush_cache run inside the update — the transport already posts `flush_cache: false` and issues a separate `/flush_cache` after the broadcast |

### Cleanup

```bash
kubectl delete pytorchjob megatron-train
kubectl delete svc megatron-train-master sglang-rbg-leader
kubectl delete rbg sglang-rbg
kubectl delete configmap megatron-train-config
kubectl delete pod <driver-pod>
```

Validated topology (ACK, 2×8xH20): driver pod (0 GPU, local Ray) + PyTorchJob
Master (1 GPU, `train_server.py` + `Qwen2.5-0.5B-Instruct`, mcore TP=1,
`vanilla_mbridge=true`) + RBG leader (1 GPU, SGLang 0.5.12, TP=1). The train
server's `--config` is a full verl PPO config exported via
`python -m verl.trainer.main_ppo --cfg job --resolve model_engine=megatron ...`
(only `actor_rollout_ref` is read). Driver-side batch knobs that the forwarder
never uses (`actor.ppo_{mini,micro}_batch_size*`,
`rollout.log_prob_micro_batch_size_per_gpu`) still must satisfy
`validate_config`.

## Weight sync (pluggable)

Select with `remote_backend.megatron_sglang.weight_sync.transport`:

- **`nccl_http`** (default) — Megatron rank 0 broadcasts unsharded HF weights
  over a cross-cluster NCCL group; SGLang pulls via
  `/update_weights_from_distributed`. Fast; needs NCCL/RDMA reachability.
- **`store`** — Megatron writes an HF checkpoint to a shared store; SGLang
  reloads via `/update_weights_from_disk`. Reachability-free fallback.
- **`mooncake`** — RDMA point-to-point via Mooncake. Fastest; most infra.

Add a transport = one file in `weight_sync/` + one registry decorator.

## External-SGLang-only mode (former `recipe/external_sglang`)

When only **inference** is external and verl still trains FSDP in-process (the
remote backend above externalizes *both*), use `checkpoint_engine.py` instead
of the `RemoteBackend` path. It was consolidated here from the former
`recipe/external_sglang` recipe.

- `ExternalSGLangNCCLEngine` (backend `external_sglang_nccl`): trainer rank 0
  pushes weights to external SGLang endpoints over HTTP + NCCL
  (`/init_weights_update_group`, `/update_weights_from_distributed`,
  `/flush_cache`). No colocated receiver, no CUDA IPC, SGLang need not be in Ray.
- `ExternalSGLangCheckpointManager`: drives only the trainer group and skips
  verl-replica orchestration.

```bash
python -m verl.trainer.main_ppo \
  ... \
  actor_rollout_ref.rollout.checkpoint_engine.backend=external_sglang_nccl \
  actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=recipe.remote_megatron_sglang.checkpoint_engine \
  '+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.external_sglang_nccl.sglang_endpoints=["http://sgl-0:30000"]' \
  actor_rollout_ref.rollout.checkpoint_manager_class=recipe.remote_megatron_sglang.checkpoint_engine.ExternalSGLangCheckpointManager \
  '+actor_rollout_ref.rollout.gateway_url=http://your-sglang-router:8080'
```

Related assets (moved from `recipe/external_sglang`):

- Validation suite: `test/external_sglang/` — NCCL-HTTP correctness, 2×TP4
  host-memory, PD-disagg, benchmarks (`weight_sync_benchmark_results.md`).
- Docs: `docs/external-engine-comparison.md`, `docs/autoscaling-patent-design.md`.
- Deployment: `k8s/sglang-external.yaml` (1 pod), `sglang-external-8gpu.yaml`,
  `sglang-2pod-tp4.yaml`.
- Drivers: `train_agentic_ack_external.sh` (mooncake|nccl switch over
  `recipe/agentic/train_agentic_disagg.sh`), `train_gsm8k_external.sh`.

## Issues found & fixed during e2e validation

0. ~~Core wiring is absent~~ **Fixed by `v1_trainer.py`** (see "V1 integration").
1. ~~Rollout replica selection~~ **Fixed in `train_gsm8k_remote.sh`**: it now
   sets `actor_rollout_ref.rollout.name=megatron_sglang`, pins the verl-side
   rollout parallelism to 1×1×1 and exports `MEGATRON_SGLANG_ENDPOINTS`.
2. ~~Weight-sync trigger wiring~~ **Works as-is**: V1's
   `CheckpointEngineManager.update_weights()` with the pinned `naive` backend
   reduces to `actor_wg.update_weights(...)`, which reaches the forwarder's
   transport. `PPOTrainerSync.on_step_end` fires it every step (verified:
   `timing_s/update_weights ≈ 0.8 s/step`).
3. **Train server engine ops** — verified against this build:
   `TrainingWorker.{reset,set_loss_fn,infer_batch,train_mini_batch,save_checkpoint,load_checkpoint}`
   and `engine.get_per_tensor_param()` all exist
   (`verl/workers/engine_workers.py`), so `train_server.py`'s calls line up.
4. **Cross-pod NCCL group for `nccl_http`** — verified pod-network TCP NCCL
   (no RDMA): the train server forms a world_size=2 group (rank 0 + SGLang TP
   ranks) and broadcast 290 tensors per sync. RDMA/HCA still recommended for
   large models; `store` remains the reachability-free fallback.
5. **RBG CRD shape** — verified against `sgl-project/rbg` v1alpha1
   (`workloads.x-k8s.io`): `spec.roles[].{name,replicas,workload,template}` and
   pod labels `rbg.workloads.x-k8s.io/{group-name,role-name}` match
   `rbg-sglang.yaml`'s Service selector. ACK clusters ship
   `pytorchjobs.kubeflow.org` / `rayclusters.ray.io` but **not**
   `rolebasedgroups` — install the operator first (see Deploy).

Additional fixes landed during validation (all in `server/train_server.py`):
the standalone server must self-inject `actor.optim.total_training_steps`
(the V1 driver normally does), derive `loss_mask` from `response_mask` when a
batch omits it, and inject `global_token_num` as a per-sequence list (an int
crashes `FlopsCounter.estimate_flops`). Batches must be packed (nested THD)
when `use_remove_padding=True`; the `torch.save`-based codec round-trips
nested tensors faithfully, so the driver path needs no extra handling.

## Test

```bash
pytest recipe/remote_megatron_sglang/test/test_remote_backend.py
# the weight-sync registry test runs without torch; the rest need the verl env.
```
