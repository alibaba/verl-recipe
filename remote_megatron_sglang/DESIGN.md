# Design: RemoteBackend for external Megatron (train) + SGLang (inference)

**Branch:** `feat/remote-backend-megatron-sglang`
**Status:** design (pre-implementation)

## Context

verl today owns its compute: it launches training engines (FSDP/Megatron) and
inference engines (vLLM/SGLang) *inside its own Ray cluster on its own GPUs*.
This design makes **both** engines external and separately deployed on
Kubernetes, with verl reduced to a **CPU-only orchestrator** that runs the RL
algorithm (PPO/GRPO), data pipeline, advantage, reward, and metrics:

- **Training** = Megatron-LM in a **PyTorchJob** (Kubeflow `kubeflow.org/v1`).
- **Inference** = SGLang in a **RoleBasedGroup** (`AliyunContainerService/rolebasedgroup`,
  multi-role orchestration + service discovery, leader/worker or prefill/decode).

This is distinct from the two existing references:
- Upstream RemoteBackend RFC (#6422 / Arctic RL): *one* external backend does
  train **and** infer.
- `checkpoint_engine.py` (from the former `recipe/external_sglang`): only inference is external; verl still trains FSDP
  in-process.

Here training and inference are **two separate external clusters**, and verl
only triggers the weight sync between them.

### Prerequisite finding

The `RemoteBackend` ABC / registry / `RemoteBackendTrainer` / CPU-pool support
**do not exist in this repo** — they are only in the unmerged upstream PR #6422.
So the work has two layers:

1. **Port the RemoteBackend core scaffolding** (small, gated, zero-intrusion).
2. **Implement the `megatron_sglang` backend** on top.

## Decisions (confirmed)

| Fork | Decision |
|------|----------|
| Weight-sync transport | **Pluggable** strategy interface (nccl_http / store / mooncake), config-selected |
| Control plane (verl ↔ clusters) | **HTTP/REST** (`httpx`), reuse `gateway_url` / `GatewayLLMServerClient` |
| Megatron train server | **Reuse verl `MegatronEngine`** wrapped in an HTTP server, behind a stable HTTP contract (Option A) |

## Architecture

```
                    ┌───────────────────────────────────────┐
                    │  verl driver  (CPU-only, 0-GPU pool)    │
                    │  RemoteBackendTrainer: PPO/GRPO loop,   │
                    │  data pipeline, advantage, metrics      │
                    │        │  1 CPU forwarder worker         │
                    │        ▼                                │
                    │  MegatronSGLangBackend (adapter)        │
                    └────┬───────────────────────┬───────────┘
                  HTTP control            HTTP control
                         │                        │
          ┌──────────────▼─────────┐   ┌──────────▼──────────────┐
          │ TRAIN cluster          │   │ INFERENCE cluster        │
          │ PyTorchJob (Kubeflow)  │   │ RoleBasedGroup (rbgs)    │
          │  Master + Worker pods  │   │  leader + worker roles   │
          │  verl-train-server     │   │  SGLang TP servers        │
          │  wraps MegatronEngine  │   │  OpenAI/SGLang HTTP API   │
          └────────────┬───────────┘   └──────────▲──────────────┘
                       └──── weight sync (direct, pluggable) ──┘
```

**Division of labor** (matches the ABC — verl owns orchestration + weight-sync
trigger; the backend owns compute):
- **verl driver:** rollout scheduling, reward, advantage (GAE/GRPO), KL,
  metrics, checkpoint trigger.
- **Megatron PyTorchJob:** `compute_log_prob`, `update_actor` (fwd+bwd+opt),
  ref-model forward, holds master weights.
- **SGLang RBG:** `generate`.
- **`update_weights`:** Megatron → SGLang directly; verl drives the handshake.

## Layer 1 — core scaffolding (ported, gated behind `trainer.remote_backend`)

| File | Change |
|------|--------|
| `verl/remote_backend/base.py` | `RemoteBackend` ABC: `from_config` / `reconnect_handle` / `destroy` / `update_weights` / `save_checkpoint` / `requires_single_forwarder`. |
| `verl/remote_backend/__init__.py` | `RemoteBackendRegistry` (explicit registration). |
| `verl/remote_backend/trainer.py` | `RemoteBackendTrainer(RayPPOTrainer)`: `use_gpu=False`, put `reconnect_handle()` in `wg_kwargs`, `destroy()` in `finally`. |
| `verl/single_controller/ray/base.py` | `create_resource_pool(use_gpu=…)` + `gpu_resource_pool_dict` + `get_n_gpus()` counts GPU pools only → enables **0-GPU pools**. |
| `verl/trainer/main_ppo.py` | branch on `config.trainer.remote_backend`: swap worker cls + trainer cls; wrap `fit()` in `try/finally: destroy()`. |
| `verl/trainer/ppo/ray_trainer.py` | add `self.wg_kwargs` / `self.use_gpu`; thread into resource-pool + worker-group creation; `max(n_gpus,1)` for throughput metric. |
| `verl/trainer/config/ppo_trainer.yaml` | add `trainer.remote_backend: null` + optional Hydra default `remote_backend@remote_backend`. |

Default (`remote_backend=null`) leaves the FSDP/Megatron path byte-for-byte
unchanged.

## Layer 2 — the `megatron_sglang` backend (recipe, out of core)

```
recipe/remote_megatron_sglang/
  __init__.py
  backend.py            MegatronSGLangBackend(RemoteBackend)  → registered "megatron_sglang"
  train_client.py       httpx client → PyTorchJob Service (compute_log_prob/update_actor/push_weights/save_checkpoint)
  infer_client.py       wraps GatewayLLMServerClient (generate) + SGLang weight-update endpoints
  forwarder_worker.py   CPU-only forwarder; @register(ONE_TO_ALL) for update_weights/save_checkpoint, mesh dispatch for compute
  weight_sync/
    base.py             WeightSyncTransport ABC: setup() / sync(step) / teardown()  + registry
    nccl_http.py        Megatron rank0 torch.distributed.broadcast + SGLang HTTP (default)
    store.py            HDFS/S3 write → SGLang /update_weights_from_disk
    mooncake.py         RDMA P2P (Mooncake transfer engine)
  server/
    train_server.py     runs INSIDE the PyTorchJob; wraps verl MegatronEngine behind HTTP
    protocol.py         shared request/response schemas (the stable HTTP contract)
  config/
    megatron_sglang.yaml  backend Hydra config (endpoints, parallelism, weight_sync.transport)
  k8s/
    pytorchjob-megatron.yaml   kind: PyTorchJob (Master+Worker, runs train_server, exposes Service)
    rbg-sglang.yaml            kind: RoleBasedGroup (SGLang leader+worker roles + Service)
  train_gsm8k_remote.sh
  README.md
```

**Reuse, not rewrite:**
- Train server wraps `verl.workers.engine.megatron.MegatronEngine` → reuses the
  mcore build, `AutoBridge` HF↔Megatron conversion, `get_per_tensor_param()`
  weight extraction, checkpoint manager, and fwd/bwd verbatim.
- Generation reuses `GatewayLLMServerClient` (set `rollout.gateway_url`).
- `nccl_http` transport reuses the `external_sglang` NCCL-over-HTTP logic
  (`/init_weights_update_group`, `/update_weights_from_distributed`,
  `/flush_cache`).

## HTTP contract (train server, `server/protocol.py`)

Stable API so Option-B (standalone Megatron) can later slot in unchanged:

| Endpoint | Body → Response |
|----------|-----------------|
| `POST /compute_log_prob` | token ids + attn → per-token logprobs |
| `POST /update_actor` | batch + advantages/returns → loss/grad-norm metrics |
| `POST /push_weights` | `{transport, step}` → triggers weight export/broadcast |
| `POST /save_checkpoint` | `{path, step}` → ack |
| `GET  /health` | readiness (PyTorchJob rank0 only) |

Tensors serialized as safetensors/np over HTTP; large logprob payloads chunked.

## Weight-sync (pluggable) flow — `nccl_http` default

1. verl → SGLang: `POST /init_weights_update_group` (once) — forms an NCCL group
   spanning Megatron ranks + SGLang TP workers.
2. verl → Megatron: `POST /push_weights` — rank 0 broadcasts unsharded HF
   weights (`get_per_tensor_param()`).
3. verl → SGLang: `POST /update_weights_from_distributed` (SGLang
   `model.load_weights()` re-shards per TP) + `POST /flush_cache`.

`store` and `mooncake` implement the same `WeightSyncTransport` interface; the
backend's `update_weights()` just calls `transport.sync(step)`.

## Deployment

- **`pytorchjob-megatron.yaml`** — `apiVersion: kubeflow.org/v1`, `kind: PyTorchJob`;
  `Master: 1` + `Worker: N`; container runs `python -m recipe.remote_megatron_sglang.server.train_server`;
  Kubeflow injects `MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE`; rank-0 Service exposes the RL HTTP API.
- **`rbg-sglang.yaml`** — `kind: RoleBasedGroup`; SGLang leader + TP-worker roles
  (role startup ordering + cross-role service discovery via RBG); Service exposes
  OpenAI/SGLang generate + weight-update endpoints.

## Config example

```bash
python -m verl.trainer.main_ppo \
  trainer.remote_backend=megatron_sglang \
  remote_backend=megatron_sglang \
  hydra.searchpath='[file://recipe/remote_megatron_sglang/config]' \
  remote_backend.megatron_sglang.train_endpoint=http://megatron-master:8000 \
  remote_backend.megatron_sglang.weight_sync.transport=nccl_http \
  actor_rollout_ref.rollout.gateway_url=http://sglang-rbg-router:8080 \
  algorithm.adv_estimator=grpo data.train_files=... trainer.n_gpus_per_node=0
```

## Verification

- **Unit:** registry resolves `megatron_sglang`; `requires_single_forwarder`
  assert; mocked-HTTP clients round-trip; each `WeightSyncTransport` selectable.
- **Local integration:** docker-compose — fake train-server + one real
  single-GPU SGLang — run 2 GRPO steps on GSM8K; assert logprob/reward shapes
  and one successful weight-sync round.
- **Cluster:** apply both manifests; `train_gsm8k_remote.sh`; confirm 4-step
  training + generation + weight-sync; reward curve sane vs in-process baseline.

## Risks / phasing

- **Coupling to `MegatronEngine`** (not a stable API; the RFC itself refactors
  it) — isolated inside `server/`; HTTP contract stays stable.
- **Cross-cluster NCCL reachability** for `nccl_http` — needs RDMA/HCA on both
  pods (cf. prior external-sglang weight-sync work); `store` transport is the
  reachability-free fallback.
- **Phasing:** (1) core scaffolding + registry, (2) HTTP contract + train server
  reusing MegatronEngine, (3) `nccl_http` transport + infer client, (4) k8s
  manifests + e2e, (5) `store` / `mooncake` transports.
```
