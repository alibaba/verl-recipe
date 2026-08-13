# Step-by-Step: Disaggregated GRPO Training on Wulan Cluster

Tested and verified end-to-end on 2026-06-12.

**What:** Fully disaggregated GRPO RL training with Qwen3.6-27B using verl + SGLang + Mooncake checkpoint engine.

**Cluster:** Wulan (ACK), 2 GPU nodes (8xH20 96GB each), lingjun nodes, RDMA-capable (4 HCAs per node).

**Outcome:** 30 training steps completed in 1h29m. Mooncake weight sync at 3.3 GB/s over RDMA. Full pipeline validated: FSDP training, SGLang rollout, cross-node P2P weight sync, GRPO with rule-based reward.

## Prerequisites

- `kubeconfig`: `~/.kube/config-wulan`
- KubeRay operator installed on the cluster
- PVCs: `ym-dataset` (for GSM8K data), `ym-models` (for model weights, e.g. `/mnt/models/Qwen3.6-27B`)
- Image registry: `registry-cn-hangzhou.ack.aliyuncs.com/dev/verl`
- `imagePullSecrets`: `regcred-cn-hangzhou`

## Step 1: Build and Push Image

The Dockerfile is at `examples/k8s-ray-mooncake/Dockerfile`. Base image: `verlai/verl:sgl0512.dev1`.

```bash
# Ensure README.md is in the COPY line (setup.py reads it)
# The line should be: COPY verl /tmp/verl-src/verl
#                      COPY setup.py pyproject.toml README.md /tmp/verl-src/

# Build via Chorus CI: push to the inner remote on branch ack-test
git push inner ack-test
# Chorus suite ID: ba784e86-2ca1-42d3-a9ec-653cd0c138f8
# Image tag format: registry-cn-hangzhou.ack.aliyuncs.com/dev/verl:2.47.1.sglmooncake.<commit-hash>
```

## Step 2: Deploy RayCluster

```bash
kubectl --kubeconfig ~/.kube/config-wulan apply -f examples/k8s-ray-mooncake/ray-cluster-wulan.yaml
```

Key adaptations in `ray-cluster-wulan.yaml` (vs the generic `ray-cluster-verl-mooncake.yaml`):

| Setting | Value | Why |
|---------|-------|-----|
| replicas | 1 training + 1 rollout | 2-node cluster |
| nodeSelector | removed | lingjun nodes lack `verl.io/pool` labels |
| memory limits | 1024Gi | H20 nodes have large host RAM |
| shm | 128Gi | Needed for NCCL and model loading |
| `RAY_memory_usage_threshold` | 0.99 | Prevent Ray OOM killer from killing workers prematurely |
| tolerations | `nvidia.com/gpu` + `node-role.alibabacloud.com/lingjun` | Required for lingjun GPU nodes |
| `rdma/hca: 1` | on workers + mooncake master | RDMA access for Mooncake |

Verify pods are running:
```bash
kubectl --kubeconfig ~/.kube/config-wulan get pods -l app=verl-disagg-mooncake
# Expected: head, training-workers-worker-*, rollout-workers-worker-*
```

## Step 3: Prepare Dataset

```bash
# On head pod
kubectl --kubeconfig ~/.kube/config-wulan exec <head-pod> -c ray-head -- \
  python3 /workspace/examples/data_preprocess/gsm8k.py --local_dir /mnt/data/gsm8k

# Output: /mnt/data/gsm8k/train.parquet (7.5k examples)
#         /mnt/data/gsm8k/test.parquet (1.3k examples)
```

For quick validation runs, create smaller datasets:
```python
import pandas as pd
df = pd.read_parquet('/mnt/data/gsm8k/train.parquet')
df.head(256).to_parquet('/mnt/data/gsm8k/train_small.parquet')
df_test = pd.read_parquet('/mnt/data/gsm8k/test.parquet')
df_test.head(64).to_parquet('/mnt/data/gsm8k/test_small.parquet')
```

## Step 4: In-Pod Setup

Install verl from source and apply flashinfer workaround on **both** worker pods:

```bash
# On each worker pod (training + rollout)
kubectl --kubeconfig ~/.kube/config-wulan exec <worker-pod> -c ray-worker -- bash -c '
pip install /workspace/verl -i https://mirrors.aliyun.com/pypi/simple/
pip install flashinfer-python==0.6.3 flashinfer-cubin==0.6.3 -i https://mirrors.aliyun.com/pypi/simple/
'
```

## Step 5: Launch Training

Run on the **head pod**:

```bash
kubectl --kubeconfig ~/.kube/config-wulan exec <head-pod> -c ray-head -- bash -c '
export TRAIN_NNODES=1 ROLLOUT_NNODES=1 NGPUS_PER_NODE=8
export MODEL_PATH=/mnt/models/Qwen3.6-27B
export TRAIN_FILE=/mnt/data/gsm8k/train_small.parquet TEST_FILE=/mnt/data/gsm8k/test_small.parquet
export TRAIN_BATCH_SIZE=32 PPO_MINI_BATCH_SIZE=8
export ROLLOUT_TP=8 PD_ENABLED=false ENABLE_HICACHE=false
export ROLLOUT_N=2 MAX_RESPONSE_LENGTH=2048
export PPO_MAX_TOKEN_LEN_PER_GPU=8192 ROLLOUT_GPU_MEM_UTIL=0.85
export CHECKPOINT_ENGINE_BACKEND=mooncake ACTOR_LR=1e-6
export TOTAL_EPOCHS=2 SAVE_FREQ=50 TEST_FREQ=10
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export FLASHINFER_DISABLE_VERSION_CHECK=1

nohup bash /workspace/run_grpo_disagg_sglang_mooncake.sh \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  trainer.val_before_train=False \
  trainer.logger='\''["console"]'\'' \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
  > /tmp/train.log 2>&1 &
'
```

### Critical Parameters for Qwen3.6-27B

| Parameter | Value | Constraint |
|-----------|-------|------------|
| `PPO_MINI_BATCH_SIZE` | 8 | Must be >= data_parallel_size (= number of training GPUs) |
| `update_weights_bucket_megabytes` | 3072 | Must be > largest single weight tensor. Qwen3.6-27B `embed_tokens` = 248320 x 5120 x bf16 = 2425 MB. Default 2048 is too small. |
| `param_offload` | True | 27B model needs CPU offloading on 8xH20 |
| `optimizer_offload` | True | Optimizer states for 27B are ~100 GB |
| `ROLLOUT_TP` | 8 | Full node for 27B model (no PD split) |
| `PD_ENABLED` | false | TP=8 uses all GPUs, no room for PD split |
| `TRAIN_BATCH_SIZE` | 32 | Reduced for small validation dataset |

## Step 6: Monitor

```bash
# Follow training log
kubectl --kubeconfig ~/.kube/config-wulan exec <head-pod> -c ray-head -- tail -f /tmp/train.log

# Ray dashboard
kubectl --kubeconfig ~/.kube/config-wulan port-forward svc/verl-disagg-mooncake-head-svc 8265:8265

# Check step progress
kubectl --kubeconfig ~/.kube/config-wulan exec <head-pod> -c ray-head -- \
  grep "OneStepTaskRunner.*global_step" /tmp/train.log | grep -oP "global_step:\d+" | tail -5
```

### Milestones to Watch

1. `Loading weights: 100%` — FSDP model loaded on training node (~1.5 min)
2. `Topology discovery complete. Found 4 HCAs` — Mooncake RDMA initialized
3. `Rank 0 send weights done, bandwidth: 3.3 GB/s` — First weight sync successful
4. `Multi-thread loading shards: 100%` — SGLang model loaded on rollout node
5. `step:1 - training/global_step:1` — First training step completed
6. `Training Progress: 100%` — Training complete

## Troubleshooting

### 1. `ValueError: Total available GPUs 0`

**Cause:** Stale Ray placement groups from previous failed runs hold all GPU reservations.

**Fix:** Remove stale PGs using Ray internal API:
```python
import ray
from ray._raylet import PlacementGroupID
import binascii

ray.init(address='auto')
pgs = ray.util.placement_group_table()
for pg_id_hex, info in pgs.items():
    if info['state'] == 'CREATED':
        worker = ray._private.worker.global_worker
        worker.core_worker.remove_placement_group(
            PlacementGroupID(binascii.unhexlify(pg_id_hex)))
        print(f'Removed: {info["name"]}')
print(f'Available GPUs: {ray.available_resources().get("GPU", 0)}')
```

**Warning:** Only remove PGs when no training is running. Removing PGs of a live run causes `ActorDiedError`.

### 2. `AssertionError: mini_batch_size=4 < data_parallel_size=8`

**Cause:** `PPO_MINI_BATCH_SIZE` must be >= the number of training GPUs (data parallel size).

**Fix:** Set `PPO_MINI_BATCH_SIZE=8` (or higher, must be >= number of GPUs).

**Location:** `verl/trainer/ppo/ray_trainer.py`, `_balance_batch()` -> `_get_dp_size()`.

### 3. `AssertionError: Weight embed_tokens too large for bucket`

**Cause:** Mooncake checkpoint engine sends weights in buckets via RDMA. Each tensor must fit entirely in one bucket. Qwen3.6-27B has vocab_size=248320, hidden_size=5120 -> `embed_tokens` = 248320 x 5120 x 2 bytes (bf16) = 2425 MB. Default bucket is 2048 MB.

**Fix:** Add override: `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072`

**Formula:** `bucket_size_mb >= vocab_size * hidden_size * bytes_per_param / 1024 / 1024 + 100`

**Location:** `verl/checkpoint_engine/mooncake_checkpoint_engine.py:198`

### 4. Dockerfile build fails: `FileNotFoundError: README.md`

**Cause:** `setup.py` reads `README.md` for the long description, but it wasn't included in the COPY line.

**Fix:** Ensure the Dockerfile has:
```dockerfile
COPY verl /tmp/verl-src/verl
COPY setup.py pyproject.toml README.md /tmp/verl-src/
```

### 5. SGLang server crashes silently after CUDA graph capture

**Possible causes:** OOM on the rollout node, NCCL timeout, or RDMA issues.

**Debug:**
```bash
# Check kernel OOM killer
kubectl --kubeconfig ~/.kube/config-wulan exec <rollout-pod> -c ray-worker -- dmesg | grep -i oom

# Check SGLang stderr
kubectl --kubeconfig ~/.kube/config-wulan exec <head-pod> -c ray-head -- \
  grep "SGLangHttpServer" /tmp/train.log | tail -20
```

**Mitigations:** Increase `shm` to 128Gi, set `RAY_memory_usage_threshold=0.99`, reduce `ROLLOUT_GPU_MEM_UTIL`.

## Performance Results (Qwen3.6-27B, 2x8xH20)

| Metric | Value |
|--------|-------|
| Total steps | 30 |
| Total time | 1h29m |
| Time per step (after warmup) | ~110s |
| Mooncake weight sync bandwidth | 3.3 GB/s (RDMA) |
| Weight sync time per step | ~16s (52 GB model) |
| Ref model compute per step | ~13s |
| Actor update per step | ~75s |
| SGLang generation per step | ~95s (async, overlapped) |
| Throughput | ~150 tokens/s |
| GPU memory allocated | 76.4 GB / 96 GB per GPU |
| GPU memory reserved | 93.7 GB / 96 GB per GPU |
| CPU memory (with offload) | ~820 GB per training node |
| Checkpoint save (model + optimizer) | ~24 min at final step |

## File Reference

| File | Purpose |
|------|---------|
| `ray-cluster-wulan.yaml` | RayCluster CRD adapted for wulan (2 nodes, lingjun tolerations) |
| `ray-cluster-verl-mooncake.yaml` | Generic RayCluster CRD (template) |
| `Dockerfile` | Image with Mooncake + RDMA + verl from source |
| `run_grpo_disagg_sglang_mooncake.sh` | Training launch script (all env vars documented in header) |
| `README.md` | Architecture overview and configuration reference |
