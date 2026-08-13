# Remote backend: external Megatron (train) + external SGLang (inference)

Run verl RL training where **both** engines live outside verl's Ray cluster:

- **Training** — Megatron-LM in a Kubeflow **PyTorchJob**.
- **Inference** — SGLang in a **RoleBasedGroup** (RBG).
- **verl** — a **CPU-only orchestrator** (PPO/GRPO, data, advantage, reward,
  metrics). It owns no GPUs.

This is built on the generic `RemoteBackend` abstraction ported from upstream
PR #6422 (`verl/remote_backend/`). See `DESIGN.md` for the full design and the
rationale behind each decision.

> **Status: prototype / scaffold.** All code compiles and the extension points
> are wired against the verl + PR #6422 source, but it has **not** been run
> end-to-end on a cluster. Cluster/version-specific spots are marked
> `# VALIDATE`. Treat it as a concrete starting point, not a drop-in feature.

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

## Core changes (gated, zero-intrusion)

The `RemoteBackend` scaffolding + a few gated edits live in core, all behind
`trainer.remote_backend` (default `null` → standard in-process path unchanged):

- `verl/remote_backend/{base,__init__,trainer,worker_utils}.py`
- `verl/single_controller/ray/base.py` — CPU-only (0-GPU) resource pools
- `verl/trainer/main_ppo.py` — swap worker + trainer when `remote_backend` set
- `verl/trainer/ppo/ray_trainer.py` — `wg_kwargs` / `use_gpu` / `max(n_gpus,1)`
- `verl/trainer/config/ppo_trainer.yaml` — `trainer.remote_backend` field

## Deploy

```bash
# 1. Install operators (once):
#    - Kubeflow training-operator: https://github.com/kubeflow/training-operator
#    - RoleBasedGroup:            https://github.com/AliyunContainerService/rolebasedgroup
kubectl apply -f k8s/pytorchjob-megatron.yaml
kubectl apply -f k8s/rbg-sglang.yaml

# 2. Launch verl (CPU-only driver):
TRAIN_ENDPOINT=http://megatron-train-master:8000 \
SGLANG_ENDPOINT=http://sglang-rbg-leader:30000 \
WEIGHT_SYNC_TRANSPORT=nccl_http \
bash train_gsm8k_remote.sh
```

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

## Open issues to resolve before it runs (important)

1. **Weight-sync trigger wiring.** In this verl version weight sync is normally
   driven by `CheckpointEngineManager` + rollout replicas. With an external
   SGLang (generation via `gateway_url`) there is no verl-managed replica, so
   `forwarder_worker.update_weights()` must be invoked from the
   rollout/checkpoint path each step. See the `# VALIDATE` note there.
2. **Train server engine ops.** `train_server.py` calls verl engine methods
   (`infer_batch` / `train_batch` / `get_per_tensor_param` / `save_hf_weights`);
   confirm the exact names/signatures against your verl build (marked
   `# VALIDATE`).
3. **Cross-cluster NCCL group** for `nccl_http`: both pods need RDMA/HCA and a
   reachable rendezvous (`master_addr`); `init_broadcast_group` must create the
   shared group. Use `store` if the clusters can't form a collective.
4. **RBG CRD shape** (`apiVersion`, role selectors) — validate against the
   installed RoleBasedGroup CRD version.

## Test

```bash
pytest recipe/remote_megatron_sglang/test/test_remote_backend.py
# the weight-sync registry test runs without torch; the rest need the verl env.
```
