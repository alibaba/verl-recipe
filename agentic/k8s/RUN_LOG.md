# Agentic RL Training Run Log

## Run #19 (PID 8656) — 2026-06-13 12:38

**Script:** `bash /workspace/run_agentic_disagg.sh` + CLI overrides `data.train_batch_size=4 data.filter_overlong_prompts=False actor_rollout_ref.rollout.gpu_memory_utilization=0.7`

**Result:** Hydra config parse error — `remote_agent.environment_kwargs` missing `++` prefix.

**Root cause:** On-pod script was outdated (from old image build). Had bare `remote_agent.environment_kwargs=...` instead of `++remote_agent.environment_kwargs=...`.

**Fix:** `kubectl cp` the updated local script to the pod.

---

## Run #19b (PID 9295) — 2026-06-13 12:39

**Script:** Same overrides, updated script with `++` prefix copied to pod.

**Result:** Config OK → dataset materialized (4 tasks) → model loaded → SGLang started → Mooncake weight sync 3.05 GB/s → **FAILED at training step:**
```
AssertionError: 1 % 2 != 0
```

**Root cause:** `ppo_mini_batch_size=8` is multiplied by `rollout.n=2` internally (ray_trainer.py:1311), giving global mini_batch=16. Divided by dp_size=8 → 2 per GPU. But per-GPU batch is `(4*2)/8 = 1`. So `1 % 2 != 0`.

**Fix:** Set `ppo_mini_batch_size=4` so `4*2/8=1` per GPU matches the per-GPU batch.

---

## Run #20 (PID 13747) — 2026-06-13 12:54

**Script:** Updated script with `ppo_mini_batch_size=4`.

**Result:** Assertion passed → weight sync 3.00 GB/s → **FAILED at training step:**
```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 11
(input tensor's size at dimension 1), but got split_sizes=[4]
```

**Root cause (initially assumed):** PyTorch bug with 3D jagged NestedTensors (mRoPE position_ids). Documented in `chunk_tensordict` (tensordict_utils.py:321-339) but workaround missing from `index_select_tensor_dict` (line 495).

**Root cause (actual):** Agent execution FAILED on all 8 trials — `AgentLoopWorker` landed on training worker (10.8.0.25) which doesn't have `kubernetes` package. All trials returned empty/minimal responses (11 tokens total). The mRoPE bug would still need fixing after agent execution succeeds, but the immediate cause of the short data was failed agents.

---

## Run #21 (PID 16729) — 2026-06-13 13:06

**Script:** Added `actor_rollout_ref.model.use_remove_padding=False`.

**Result:** Same `split_with_sizes` error. `use_remove_padding=False` didn't help because `left_right_2_no_padding()` is called unconditionally in `_update_actor`.

**Conclusion:** Wrong fix direction.

---

## Run #22 (PID 19777) — 2026-06-13 13:14

**Script:** Added `trainer.balance_batch=False` on top of run #21.

**Result:** Same error. Detailed log analysis revealed the REAL problem:
```
AgentLoopWorker pid=21700, ip=10.8.0.25
Trial.create failed: ValueError: Failed to import module 'harbor.environments.ack': No module named 'kubernetes'
```

ALL 8 trials failed because `AgentLoopWorker` ran on training worker (10.8.0.25), not rollout worker (10.8.0.42). We only installed `kubernetes` + Harbor patches on the rollout worker.

**Key finding:** `AgentLoopWorker` is a Ray actor that can land on ANY node, just like `AgenticDisaggTaskRunner`. The `kubernetes` package, Harbor patches (ack.py, swe_agent.py, trial.py), and PVC mount must be available on ALL worker nodes, not just the rollout worker.

**Fix needed:** Install `kubernetes` on training worker (10.8.0.25). Long-term, bake it into the Docker image (already done in updated Dockerfile).

---

## Run #23 (PID 26852) — 2026-06-13 13:38

**Script:** `bash /workspace/run_agentic_disagg.sh` + `data.train_batch_size=4` (no `use_remove_padding` or `balance_batch` overrides — those were wrong directions in #21/#22).

**Changes:** Installed `kubernetes` on training worker. Copied patched `ack.py` + `swe_agent.py` from rollout worker to training worker.

**Result:** `kubernetes` import OK, but ALL 8 trials failed with:
```
unexpected indent (swe_agent.py, line 266)
```

**Root cause:** The `clone_cmd` patch in `swe_agent.py` replaced only the first line of a Python ternary expression:
```python
# Original:
clone_cmd = (
    f"git clone ... {self._version}"
    if self._version
    else "git clone ..."
)
# Patched (broken):
clone_cmd = "cp -r /mnt/data/sweagent-repo /opt/sweagent-repo"
    if self._version      # <-- dangling, causes IndentationError
    else "git clone ..."
)
```
Both rollout AND training workers had this error. It didn't surface before because `AgentLoopWorker` landed on the training worker (10.8.0.25), not the rollout worker.

**Fix:** Remove the `if self._version ... else ...` part, leaving just `clone_cmd = "cp -r ..."`. Applied on both workers.

---

## Run #24 (PID 30208) — 2026-06-13 13:49

**Script:** `bash /workspace/run_agentic_disagg.sh` + `data.train_harbor_dir=/mnt/data/swe-bench-quick-4 data.val_harbor_dir=/mnt/data/swe-bench-quick-4 data.train_batch_size=4`

**Result:** All 8 trials created on training worker (10.8.0.25). All 8 trials FAILED with:
```
ValueError: Image 'yueming-acr-registry.cn-hongkong.cr.aliyuncs.com/swebench-verified/django-django-10554:20260601' not found and no registry is configured.
```
Cleanup also failed with RBAC 403:
```
pods "django--django-10554-xxjtr2b5kz2v3mntrneaff" is forbidden: User "system:serviceaccount:default:default" cannot delete resource "pods"
```
Then proceeded to training step → same `split_with_sizes` mRoPE error (11 vs [4]) because all responses were empty.

**Root cause:** `LOCAL_TEST=true` env var is NOT set on the training worker. Verified: `os.getenv("LOCAL_TEST")` returns `None` on training worker (10.8.0.25). Without `LOCAL_TEST=true`, `_image_exists()` in `ack.py:517` tries `docker manifest inspect` / `crane` — neither installed → returns False → raises ValueError.

`LOCAL_TEST=true` was only patched in `agentic_disagg_main.py` on the HEAD pod (lines 139/145), which only affects the `AgenticDisaggTaskRunner` process. The `AgentLoopWorker` is a separate Ray actor and does NOT inherit this env var.

**Fix needed:** Set `LOCAL_TEST=true` on the training worker so `AgentLoopWorker` can access it. Options:
- Patch `ack.py` on training worker to hardcode `return True` in `_image_exists()`
- Inject via Ray runtime_env so all actors receive it

---

## Run #25 (PID 35640) — 2026-06-13 14:09

**Script:** `export LOCAL_TEST=true && bash /workspace/run_agentic_disagg.sh` + same overrides as #24.

**Changes attempted:**
1. Patched `agentic_disagg_main.py` on head pod — added `LOCAL_TEST=true` to Ray `runtime_env["env_vars"]` (inserted before `OmegaConf.merge`)
2. Exported `LOCAL_TEST=true` in shell before launch

**Result:** Same `split_with_sizes` error (11 vs [4]). Agent trials still failed — `LOCAL_TEST=true` was not propagated to the `AgentLoopWorker` on the training worker.

**Root cause:** Same as #24. The `export LOCAL_TEST=true` only affects the head pod shell. The runtime_env patch in `agentic_disagg_main.py` should propagate it, but was not verified. The fundamental issue remains: `LOCAL_TEST=true` must reach the `AgentLoopWorker` process on the training worker.

---

## All on-pod patches — verified status (as of run #28, 2026-06-13 15:03)

### HEAD pod (verl-disagg-mooncake-head-4gc4s, 10.8.0.36)

HEAD pod `num-cpus: "0"`, Ray 不会在上面调度 AgentLoopWorker，所以 Harbor patch 不影响运行。

1. `agentic_disagg_main.py` line 149-152: `LOCAL_TEST=true` 注入 `runtime_env_kwargs["env_vars"]` ✅ (run #26 确认传播)
2. `swe_agent.py`: **未 patch** (不影响运行)
3. `ack.py`: **未 patch** (不影响运行)
4. `trial.py`: **未 patch** (不影响运行)

### Training worker (verl-disagg-mooncake-training-workers-worker-5sqsh, 10.8.0.25)
1. `pip install kubernetes` ✅
2. `ack.py`: thread-safety fix — 6 处 `CoreV1Api()` 替换 ✅
3. `ack.py`: PVC volume mount (`ym-dataset` at `/mnt/data`) ✅
4. `trial.py`: timeout 1200s ✅
5. `swe_agent.py`: clone_cmd = `"cp -r /mnt/data/sweagent-repo /opt/sweagent-repo"` ✅
6. `swe_agent.py`: `--python /opt/miniconda3/bin/python3` (替换 `--python 3.12`) ✅
7. `swe_agent.py`: 删除 `uv python install 3.12` 行 ✅
8. `swe_agent.py`: `{clone_cmd} 2>/dev/null || true &&` (cp 错误处理) ✅
9. `swe_agent.py`: `uv pip install` 加阿里云镜像 ✅

### Rollout worker (verl-disagg-mooncake-rollout-workers-worker-wfvvf, 10.8.0.42)
1. `pip install kubernetes` ✅
2. `ack.py`: thread-safety fix — 6 处 ✅
3. `ack.py`: PVC volume mount ✅
4. `trial.py`: timeout 1200s ✅
5. `swe_agent.py`: clone_cmd cp from PVC ✅
6. `swe_agent.py`: miniconda python3.11 ✅
7. `swe_agent.py`: 删除 `uv python install 3.12` 行 ✅
8. `swe_agent.py`: cp 错误处理 `|| true` ✅
9. `swe_agent.py`: 阿里云 pip 镜像 ✅

### RBAC (live cluster)
- `verl-agentic` Role: pod CRUD + configmap 权限 ✅
- `verl-agentic` RoleBinding: 绑定 `verl-agentic` SA + `default` SA ✅
- Head/Training/Rollout pod 均使用 `serviceAccountName: verl-agentic` ✅

### run_agentic_disagg.sh (head pod /workspace/)
- tolerations 在 `++remote_agent.environment_kwargs` 中 ✅
- `image_pull_secret: acr-pro-registry` ✅
- `ppo_mini_batch_size=4`, `gpu_memory_utilization=0.7` ✅
- `data.filter_overlong_prompts=False` ✅
- 非 export 的 agentic env vars ✅

---

## 本地文件 vs on-pod patch 对应关系

| On-pod patch | 本地文件 | 位置 |
|---|---|---|
| ack.py thread-safety | `Dockerfile` patch a | line 87-89 |
| trial.py timeout 1200s | `Dockerfile` patch b | line 91-92 |
| swe_agent.py miniconda python | `Dockerfile` patch c | line 97-98 |
| swe_agent.py 删除 uv python install | `Dockerfile` patch c2 | line 99-100 |
| swe_agent.py clone_cmd cp | `Dockerfile` patch d | line 101-106 |
| swe_agent.py cp 错误处理 | `Dockerfile` patch e | line 108-109 |
| swe_agent.py 阿里云镜像 | `Dockerfile` patch f | line 111-112 |
| ack.py PVC mount | `Dockerfile` patch g | line 114-130 |
| LOCAL_TEST=true | `Dockerfile` ENV | line 141 |
| LOCAL_TEST runtime_env inject | on-pod only (agentic_disagg_main.py) | recipe git 不含此改动 |
| RBAC verl-agentic | `ray-cluster-wulan.yaml` | line 401-445 |
| SA binding (head/training/rollout) | `ray-cluster-wulan.yaml` | line 38/139/218 |
| tolerations in environment_kwargs | `run_agentic_disagg.sh` | line 132 |
| vllm_provider.py BatchEncoding fix | `Dockerfile` patch h | line 142-153 |

---

## Run #29 — 2026-06-13 ~15:30

**Fix applied:** `vllm_provider.py:_generate` — `apply_chat_template(tokenize=True)` 在 transformers 5.6.0 下返回 `BatchEncoding`（dict-like，有 `input_ids` + `attention_mask`），不是 `list[int]`。直接传给 SGLang 的 `GenerateReqInput(input_ids=BatchEncoding)` 导致 `_determine_batch_size` 误判：`len(BatchEncoding)=2`（key 数量），`isinstance(BatchEncoding[0], int)=False`（`Encoding` 对象），`is_single=False` → `_expand_inputs` → `ValueError: input_ids should be a list of lists`。

**修改:**
```python
# Before:
prompt_ids: list[int] = self.tokenizer.apply_chat_template(...)

# After:
encoded = self.tokenizer.apply_chat_template(...)
prompt_ids: list[int] = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
```

**Applied on:** training worker (10.8.0.25) + rollout worker (10.8.0.42)
**Baked into:** `Dockerfile` patch h

**Result:** Inference 0 errors ✅. 2/8 agent trajectories succeeded, 6 failed (SWE-agent). Training never started — AgentLoopWorker (PID 67188, training worker 10.8.0.25) was **OOM-killed** by Linux cgroup memory killer.

dmesg:
```
Memory cgroup out of memory: Killed process 3204229 (ray::WorkerDict) total-vm:379GB, anon-rss:15GB, shmem-rss:50GB
```

**Root cause:** FSDP param_offload + optimizer_offload for Qwen3.6-27B uses ~520 GB CPU memory across 8 WorkerDict processes (each ~65 GB RSS). Combined with AgentLoopWorker and Ray overhead, total exceeded training worker's 1024Gi memory limit.

All pods restarted (restart count = 2). Image patches survived (baked into Dockerfile), but `LOCAL_TEST=true` was NOT in the image (only in uncommitted local Dockerfile).

---

## Run #30 — 2026-06-14

**Fixes:**
1. **Training worker memory limit**: 1024Gi → 1500Gi (request: 768Gi → 1200Gi) in `ray-cluster-wulan.yaml`
2. **LOCAL_TEST=true env var**: Added to both training and rollout worker env vars in `ray-cluster-wulan.yaml` (persistent across restarts, no on-pod patch needed)

**Result:** (pending)

---

## Pending issues

1. **mRoPE nested tensor bug:** `index_select_tensor_dict` (tensordict_utils.py:495) lacks the try/except fallback that `chunk_tensordict` already has. Fix is prepared locally but not deployed yet. Will only matter once agents produce real multi-turn responses.

2. **LOCAL_TEST 已通过 YAML env var 解决:** 不再依赖 Dockerfile ENV 或 agentic_disagg_main.py on-pod patch。

---

## Patch code details

### Patch 1: ack.py — thread-safety fix (6 occurrences)

原始代码用 `self._core_api.connect_get_namespaced_pod_exec`，多线程并发时 `stream()` 会 monkey-patch `api_client.request` 导致冲突。

```bash
sed -i 's/self\._core_api\.connect_get_namespaced_pod_exec/k8s_client.CoreV1Api().connect_get_namespaced_pod_exec/g' \
  /usr/local/lib/python3.12/dist-packages/harbor/environments/ack.py
```

改后（共 6 处，line 1447/1543/1604/1662/1706/1754）：
```python
# 每次 stream() 调用用新的 CoreV1Api()，隔离 websocket session
k8s_client.CoreV1Api().connect_get_namespaced_pod_exec,
```

### Patch 2: ack.py — PVC volume mount for sandbox pods

在 `_create_pod_manifest` 或等效方法中，给 sandbox pod 添加 PVC 挂载，使 sandbox 内可以访问 `/mnt/data/sweagent-repo`：

```python
# container volume_mounts 中添加:
volume_mounts=[
    k8s_client.V1VolumeMount(
        name="ym-dataset",
        mount_path="/mnt/data",
        read_only=True,
    ),
],

# pod spec volumes 中添加:
volumes=[
    k8s_client.V1Volume(
        name="ym-dataset",
        persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
            claim_name="ym-dataset",
            read_only=True,
        ),
    ),
],
```

### Patch 3: trial.py — agent setup timeout 360s → 1200s

```bash
sed -i 's/_AGENT_SETUP_TIMEOUT_SEC = 360/_AGENT_SETUP_TIMEOUT_SEC = 1200/' \
  /usr/local/lib/python3.12/dist-packages/harbor/trial/trial.py
```

改后（line 138）：
```python
_AGENT_SETUP_TIMEOUT_SEC = 1200
```

### Patch 4: swe_agent.py — clone_cmd + pip mirror + miniconda python + 删除 uv python install

原始代码 `clone_cmd` 用 `git clone`，改为从 PVC 拷贝；`uv pip install` 加阿里云镜像；`uv venv` 用 miniconda python3.11；删除 `uv python install 3.12`（不需要，直接用 miniconda）：

```python
# line 265 — 原始: clone_cmd = f"git clone {repo_url}" (含 ternary)
# 改后:
clone_cmd = "cp -r /mnt/data/sweagent-repo /opt/sweagent-repo"

# 原始: "uv python install 3.12 && "
# 改后: 删除整行（用 miniconda python 不需要 uv 下载 python）

# line 272 — uv venv 用 miniconda python3.11 (原始: --python 3.12)
"uv venv /opt/sweagent-venv --python /opt/miniconda3/bin/python3 --clear && "

# line 275 — cp 错误用 || true (不能用 ; true，因为 set -euo pipefail)
f"{clone_cmd} 2>/dev/null || true && "

# line 276 — pip install 加阿里云镜像
"uv pip install /opt/sweagent-repo --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && "
```

对应 Dockerfile 中的 4 条 sed/python 命令 (patch c, c2, d, e, f)。

### Patch 5: agentic_disagg_main.py — LOCAL_TEST 注入 Ray runtime_env

在 head pod 的 `/workspace/recipe/agentic/agentic_disagg_main.py` 中，`OmegaConf.merge` 之前添加：

```python
        # Ensure LOCAL_TEST propagates to all Ray actors (for Harbor _image_exists bypass)
        _rt_env_vars = dict(runtime_env_kwargs.get("env_vars", {}))
        _rt_env_vars["LOCAL_TEST"] = "true"
        runtime_env_kwargs["env_vars"] = _rt_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
```

**注意:** 此 patch 在 run #25 中未生效，需要验证 Ray runtime_env 是否真正传播到 AgentLoopWorker。

### Patch 6 (本地未部署): tensordict_utils.py — mRoPE nested tensor unbind fix

`verl/utils/tensordict_utils.py` line 494-509，`index_select_tensor_dict` 函数中，给 nested tensor unbind 加 try/except fallback（与 `chunk_tensordict` 中已有的 workaround 一致）：

```python
            elif isinstance(tensor, torch.Tensor) and tensor.is_nested:
                ragged_idx = getattr(tensor, "_ragged_idx", tensor.dim() - 1)
                try:
                    tensor_lst = tensor.unbind()  # for performance
                except RuntimeError:
                    # Workaround for PyTorch bug with 3D+ jagged NestedTensors
                    # (e.g. mRoPE position_ids). See chunk_tensordict docstring
                    # and https://github.com/pytorch/pytorch/issues/153238
                    padded = tensor.to_padded_tensor(0)
                    offsets = tensor.offsets()
                    lengths = offsets.diff().tolist()
                    tensor_lst = [padded[j, :seq_len] for j, seq_len in enumerate(lengths)]
                selected_tensors = [tensor_lst[idx] for idx in indices]
                data_dict[key] = nested_tensor_from_tensor_list(
                    selected_tensors, ragged_idx=ragged_idx
                )
```

原始代码只有 `tensor_lst = tensor.unbind()`，没有 try/except。对 mRoPE 3D position_ids（shape `[batch, seq, 3]`），PyTorch 的 `unbind` 会在 `dim=ragged_idx-1` 上 split，维度不匹配导致 `split_with_sizes` 报错。

**状态:** 本地已改，未部署到集群。等 agent 执行成功后再部署。

---

## Run #30c — 2026-06-14 00:46 (继续)

**状态:** Step 1 training 成功完成。Step 2 agent trials 进入验证阶段时 `uv run parser.py` 卡在从 PyPI 下载包（pyarrow 46.6MB, numpy, pandas 等），sandbox pod 没有 `UV_INDEX_URL` 环境变量。

**Root cause:** Patch i (UV_INDEX_URL env var) 已 apply 到两个 worker 的 ack.py 源文件，但 AgentLoopWorker PID 18094 的 Python 进程缓存了修改前的模块。新创建的 sandbox pod 不会有 env var。

**Action:** Kill run #30c，清理 sandbox pod，重启以加载 patched ack.py。

---

## Run #31 (PID 38203) — 2026-06-14 02:13

**Script:** `bash /workspace/run_agentic_disagg.sh` + CLI overrides (同 run #30c)

**Changes:** Patch i (UV_INDEX_URL) 首次生效——新的 AgentLoopWorker PID 45445 加载了 patched ack.py。

**Result:** ✅ **完整训练成功完成！**

### Timeline
| Event | Time | Details |
|-------|------|---------|
| 启动 | 02:13 | PID 38203/38205 |
| Model loaded | 02:17 | 1184 shards, 1min16s |
| SGLang ready | 02:19 | CUDA graphs captured |
| Weight sync #1 | 02:20:29 | 3.57 GB/s (rollout→training) |
| Step 1 agents started | 02:20:33 | 8 trials (4 instances × rollout.n=2) |
| Sandbox pods created | 02:20:35 | UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ ✅ |
| Step 1 agents completed | 02:25:51 | All 8 ATIF trajectories |
| Step 1 verifiers completed | 02:30:07 | ~4min (vs 20min+ without mirror) |
| Weight sync #2 | 02:30:22 | 3.71 GB/s (training→rollout) |
| Step 2 agents started | 02:31:10 | New sandbox pods created |
| Step 2 agents completed | 02:35:22 | All 4 ATIF trajectories |
| Step 2 verifiers completed | 02:38:46 | |
| Checkpoint model save | 02:39:03 | 8×13GB model shards |
| Checkpoint optim save | 02:52:48 | 8×21GB optimizer shards |
| Validation | 02:52:48 | val-core/harbor/reward/mean@1: 0.0 |
| Training 100% | 02:52:51 | 1/1 step, 32:21 total |
| Process exit | ~02:53 | Clean exit |

### Key metrics
```
actor/ppo_kl: 1.21
actor/pg_loss: 0.0
critic/score/mean: 0.0
response_length/mean: 4096 (max)
num_turns/mean: 4
val-core/harbor/reward/mean@1: 0.0
timing_s/step: 639.98s
timing_s/gen: 577.57s (agent loop)
timing_s/update_actor: 47.17s
timing_s/save_checkpoint: 844.30s
timing_s/testing: 456.91s
perf/throughput: 7.54 tokens/s
```

### Checkpoint
`/workspace/checkpoints/agentic_swe/qwen3.6_27b_agentic_20260614_run31/global_step_1/actor/`
- 8 model shards (13GB each, 104GB total)
- 8 optimizer shards (21GB each, 168GB total)
- HuggingFace config + tokenizer

### 确认
- ✅ UV_INDEX_URL patch 生效: sandbox pod 验证器从阿里云镜像下载，~4min 完成（之前从 PyPI 20min+ 卡死）
- ✅ 无 OOM
- ✅ 无 split_with_sizes mRoPE 错误（agent 响应足够长）
- ✅ 无 DownloadVerifierDirError
- ✅ 8/8 agent trajectories 成功
- ✅ 训练步完成，梯度更新，checkpoint 保存

---

## All on-pod patches — 更新 (run #31, 2026-06-14 02:13)

在 run #30c 基础上新增:

### Training worker + Rollout worker (共同)
10. `ack.py`: UV_INDEX_URL + PIP_INDEX_URL env vars for sandbox pods ✅ (patch i)

### Patch i: ack.py — UV_INDEX_URL for sandbox container
在 sandbox pod 的 `V1Container` spec 中添加 `env` 列表：
```python
command=["sleep", "infinity"],
env=[
    k8s_client.V1EnvVar(name="UV_INDEX_URL", value="https://mirrors.aliyun.com/pypi/simple/"),
    k8s_client.V1EnvVar(name="UV_INSECURE_HOST", value="mirrors.aliyun.com"),
    k8s_client.V1EnvVar(name="PIP_INDEX_URL", value="https://mirrors.aliyun.com/pypi/simple/"),
    k8s_client.V1EnvVar(name="PIP_TRUSTED_HOST", value="mirrors.aliyun.com"),
],
```
**位置:** `ack.py` ~line 1249, sandbox container（`command=["sleep", "infinity"]` 的那个 `V1Container`）

---

## Dockerfile patch status (2026-06-14)

All patches a-i are now in the Dockerfile. Next image build will include all patches.

## Patch j: tensordict_utils.py — mRoPE nested tensor fix (deployed 2026-06-14)

在 `index_select_tensor_dict` 中，为 3D+ jagged NestedTensor（如 mRoPE position_ids）添加 try/except fallback：
- `tensor.unbind()` 对 3D nested tensor 会抛 RuntimeError
- Fallback: `to_padded_tensor(0)` + `offsets().diff()` 手动切片
- 已部署到两个 worker 节点（rollout-86ccv, training-4hn5g）
- 已在 codebase 的 `verl/utils/tensordict_utils.py` 中（commit b52c19f8），下次 image build 会包含

## Pending issues (更新 2026-06-14)

1. ~~**mRoPE nested tensor bug:**~~ ✅ 已部署到两个 worker 节点。
2. **Reward 全为 0:** `critic/score/mean: 0.0`，所有 agent trial 的 verifier 结果都是 0 分。可能因为 SWE-Agent 未能正确修复 bug。需要调查 verifier 的评分逻辑。
3. **response_length 全为 4096:** 所有响应都达到了 max_response_length，说明 agent 没有正常结束对话。
4. **Convention artifacts dir not found:** 验证器中的 best-effort 下载失败，不影响流程但值得调查。

---

## Run #40–#42 — 2026-06-15 (nested tensor ragged_idx 深度修复)

### 背景

Pod 重建后重新 apply 所有 patch，再次遇到 `split_with_sizes` 错误。此前的 try/except fallback (Patch j) 不够——需要找到根因。

### Run #40 (PID 5613) — 09:00

**结果:** `split_with_sizes expects split_sizes to sum exactly to 11, got split_sizes=[4]`

**Debug 输出:**
```
position_ids: shape=[1, j1, 11] offsets=[0, 4] values_shape=[4, 11]
```
- `offsets=[0, 4]` 表示 ragged dim 追踪 dim 0 (mRoPE=4)
- `_ragged_idx=2` 告诉 unbind 在 dim 1 (seq_len=11) 上 split
- **不匹配**: split [4] 到 size=11 的维度 → crash

### 根因分析

mRoPE position_ids shape 为 `(4, seq_len)` per sample。问题出在两处：

1. **`torch.nested.as_nested_tensor()`** 对 2D tensor 默认 `ragged_idx=1`，offsets 追踪 dim 0 (mRoPE=4)
2. **`maybe_fix_3d_position_ids()`** (tensordict_utils.py:910) 在 pickle/unpickle 后只设 `_ragged_idx=2`，但**不重建 tensor**，导致 offsets (追踪 mRoPE dim) 和 ragged_idx=2 (期望追踪 seq_len dim) 不匹配

两个 bug 叠加：
```
as_nested_tensor → offsets=[0,4], ragged_idx=1  (追踪 mRoPE dim)
    ↓ pickle/unpickle (Ray 序列化)
offsets=[0,4], ragged_idx=1 (丢失 _ragged_idx)
    ↓ maybe_fix_3d_position_ids
offsets=[0,4], ragged_idx=2  ← 不匹配！
    ↓ unbind (split along dim 1 of values)
split(values_shape=(4,11), [4], dim=1) → 4≠11 → CRASH
```

### Run #41 — 09:12

尝试只在 `list_of_dict_to_tensordict` 修复（用 `nested_tensor_from_tensor_list` 替代 `as_nested_tensor` for 2D+ tensors）。但该修复不生效，因为 batch_size=1 时走了 `torch.stack` 分支（all shapes same），不创建 nested tensor。Nested tensor 是在后续 data pipeline 中创建的。

**结果:** 同样的 crash。

### Run #42 — 09:29

**修复方案:** 修改 `maybe_fix_3d_position_ids` 从只设属性改为**重建 tensor**：

```python
# 修复前 (只设属性，不改 offsets):
def maybe_fix_3d_position_ids(data: TensorDict):
    if "position_ids" in data.keys() and data["position_ids"].dim() == 3 and data["position_ids"].is_nested:
        data["position_ids"]._ragged_idx = 2

# 修复后 (重建 tensor，offsets 和 ragged_idx 一致):
def maybe_fix_3d_position_ids(data: TensorDict):
    if "position_ids" in data.keys() and data["position_ids"].dim() == 3 and data["position_ids"].is_nested:
        pos = data["position_ids"]
        if getattr(pos, "_ragged_idx", 1) != 2:
            offsets = pos.offsets()
            values = pos.values()
            splits = (offsets[1:] - offsets[:-1]).tolist()
            individual_tensors = list(torch.split(values, splits, dim=0))
            data["position_ids"] = nested_tensor_from_tensor_list(individual_tensors, ragged_idx=2)
```

**验证:**
```python
# Bug 复现
t = torch.randn(4, 11)
nt = torch.nested.as_nested_tensor([t], layout=torch.jagged)
# shape=[1, j1, 11], offsets=[0, 4], ragged_idx=1
nt._ragged_idx = 2
nt.unbind()  # CRASH: split_with_sizes expects sum 11, got [4]

# 修复后
nt2 = nested_tensor_from_tensor_list([t], ragged_idx=2)
# shape=[1, j2, 11], offsets=[0, 11], ragged_idx=2
nt2.unbind()  # OK: split_with_sizes([11], dim=1) on values(4,11)
```

**结果:** ✅ **nested tensor 错误消失。** 训练步骤通过，checkpoint 保存成功。

但 agent sessions 失败: training worker 缺少 `kubernetes` 包（容器重建后丢失）。

### Run #43 — 09:42

安装 `kubernetes` 后重新启动。但因前一次 run 的 Ray actors 占用 GPU，报 `Total available GPUs 0 < desired 8`。

尝试 `ray stop --force` 清理，但导致 training worker 容器重启（RESTARTS: 4），rollout worker 的 Ray 进程断连。需要重新连接 Ray cluster + 重新 apply 所有 patch。

### Patch k: maybe_fix_3d_position_ids 重建 nested tensor

**文件:** `verl/utils/tensordict_utils.py`，`maybe_fix_3d_position_ids` 函数

**描述:** 对 3D nested tensor position_ids (mRoPE)，不能只设 `_ragged_idx=2`（会导致 offsets 和 ragged_idx 不匹配）。必须用 `nested_tensor_from_tensor_list` 重建 tensor，使 offsets 和 ragged_idx 一致。

**调用点:** `engine_workers.py:243` — `train_mini_batch` 开头调用，在 Ray 序列化反序列化后、数据迭代前。

---

## NCCL Backend 推理正确性验证 — 2026-06-15

### 测试环境
- **Backend:** `checkpoint_engine.backend=nccl` (NCCLCheckpointEngine)
- **Rollout pod:** `verl-disagg-mooncake-rollout-workers-worker-fkdkt` (10.8.0.92)
- **Training pod:** `verl-disagg-mooncake-training-workers-worker-fgmps` (10.8.2.174)
- **Model:** Qwen3.6-27B, TP=8
- **Config 确认:** 训练日志中 `'backend': 'nccl'`，`ProcessGroupNCCL.cpp` 警告可见

### Weight Sync 详情
- 10 个 bucket，总计 ~27GB
- Bucket sizes: 879MB + 2986MB + 7×2903MB + 2342MB + 2425MB
- 所有 TP rank 的 `load_weights` 调用: **0 skipped params**

### 推理测试结果 (weight sync 后)

| Test | 结果 | unique_chars | 内容摘要 |
|------|------|-------------|---------|
| 数学 "2+3" | ✅ PASS | 37 | 包含 thinking 过程 + 正确答案 |
| 中文自我介绍 | ✅ PASS | 43 | "我是 Qwen (通义千问)，由阿里云开发…" |
| 代码生成 | ✅ PASS | 31 | has_code=True, `def add(a, b): return a + b` |

### 结论

**NCCL weight sync 产生正确的模型输出。** Bug 仅存在于 Mooncake checkpoint engine。

以下组件已排除嫌疑:
- FSDP 权重提取 (`get_per_tensor_param` / `convert_weight_keys`)
- SGLang 的 `update_weights_from_tensor` / `model.load_weights()`
- `/dev/shm` bypass 序列化
- 参数名映射 (`model.language_model.` → `model.`, `.self_attn.` removal)

---

## Run #44 (PID 297031) — 2026-06-16 02:55 (NCCL + sandbox + 4 sample)

**Script:** `bash /tmp/run_nccl_test.sh`
- `checkpoint_engine.backend=nccl`
- `data.train_harbor_dir=/mnt/data/swe-bench-quick-4` (4 django tasks)
- NCCL env vars patched on both workers
- Training pod: `fgmps` (10.8.2.174), packages fixed (tokenizers 0.22.2, kubernetes, cupy-cuda13x)

**NCCL 环境变量（Wulan 集群 IPv6 RDMA 必需）:**
```bash
export NCCL_IB_ADDR_RANGE='2001:db8:80f:e000::/60'
export NCCL_IB_ADDR_FAMILY='AF_INET6'
export GLOO_SOCKET_IFNAME='eth0'
export NCCL_DEBUG='INFO'
export NCCL_NET_PLUGIN='none'
export CUDA_DEVICE_MAX_CONNECTIONS='1'
```
这些变量通过 `nccl_checkpoint_engine.py` 头部 `os.environ.setdefault()` 注入，需 patch 到两个 worker pod。
注意: `os.environ.setdefault` 不会覆盖已有变量（如 YAML 中的 `NCCL_DEBUG=WARN`）。

**额外依赖:** `cupy-cuda13x`（NCCL backend 必需，`pip install cupy-cuda13x -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com`）

**Result:** ✅ **训练完成**

```
Training Progress: 100%|██████████| 1/1 [15:11<00:00, 911.59s/it]
```

### Key metrics
| 指标 | 值 |
|------|-----|
| Weight sync | 10 buckets, ~27GB, all TP ranks skipped=0 |
| response_length/mean | 463.25 (min=314, max=567) |
| response/aborted_ratio | 0.0 |
| actor/ppo_kl | 0.625 |
| actor/pg_loss | 0.0 |
| critic/score/mean | 0.0 |
| val-core/harbor/reward/mean@1 | 0.0 |
| num_turns/mean | 4.0 |
| timing_s/gen | 373s (agent loop) |
| timing_s/update_weights | 162s (NCCL broadcast) |
| timing_s/update_actor | 47s |
| timing_s/save_checkpoint | 50s |
| timing_s/step | 582s |
| perf/throughput | 2.05 tokens/s |

### Checkpoint
`/workspace/checkpoints/agentic_swe/nccl_weight_sync_test/global_step_1/actor/`

### 确认
- ✅ NCCL weight sync 正常，0 skipped params
- ✅ 推理输出正确（非乱码），response_length 314-567（非打满 4096）
- ✅ Agent 执行: 12 个 trial（4 task × 3），全部有 trajectory
- ✅ 训练步完成，checkpoint 保存成功
- ✅ 无 OOM，无 mRoPE 错误
- **结论: NCCL backend 在实际训练中也正常工作，进一步确认 Mooncake-specific bug**

### 与 Run #31 (Mooncake) 对比
| | Run #31 (Mooncake) | Run #44 (NCCL) |
|---|---|---|
| Backend | mooncake | nccl |
| Weight sync 速度 | 3.57 GB/s | ~2.7 GB/s |
| 推理输出 | 乱码 | ✅ 正确 |
| response_length/mean | 4096（打满） | 463（正常） |
| reward | 0.0 | 0.0 |

**轨迹下载:** 12 个 NCCL run trial 已下载到本地 `run44-nccl-trials/trials/`（仅保留 11 点后的正常输出轨迹）
- django-10097 ×3, django-10554 ×3, django-10880 ×3, django-10914 ×3

---

## Mooncake 乱码根因深度排查 — 2026-06-16/17

### 背景

Run #44 (NCCL) 确认 NCCL backend 推理正确后，问题被隔离到 **Mooncake checkpoint engine 特有**。但 Mooncake 乱码的根因仍未知。本次调试目标是逐层排除 weight sync 路径上的每个环节。

### 排查方法

1. **CE dump patch**：在 rollout worker 的 `mooncake_checkpoint_engine.py` 上部署诊断 patch
   - `_dump_ranks = {1}`, `_dump_steps = None`, `_dump_stats_only = True`
   - 在 `receive_weights` 中记录每个 weight 的 name/shape/dtype/norm/mean/first5
   - dump 保存在 `/tmp/ce_weight_dumps/mooncake_1_step0/metadata.json`

2. **SGLang server log 分析**：从 SGLang 进程的 stdout/stderr 中提取 `LM-NORM-*` 和 `load_weights` 诊断信息

3. **磁盘原始模型对比**：用 Python 脚本从 `/mnt/models/Qwen3.6-27B` 加载权重，计算对应 shard 的 norm

### 排查结果

#### 1. Mooncake RDMA 数据传输 ✅ 正确

- 所有 bucket 的 `[CKSUM-RECV]` 均为 OK — **无校验和不匹配**
- `[IPC-FIX-v6]` 日志显示每个 bucket 成功写入 `/dev/shm`（~2903MB/bucket）
- Rank 1 receive weights 完成: 159.10s, bandwidth 0.32 GB/s
- dump metadata 包含 1184 个权重，total_params = 27,356,728,560

#### 2. get_named_tensor_buckets ✅ 安全

代码路径 `verl/workers/rollout/sglang_rollout/utils.py:99-108`：
```python
# 每个 tensor 都经过 .clone()，脱离 CE buffer 生命周期
current_bucket.append((name, tensor.clone()))
```
- Mooncake CE yield 的 tensor 是 RDMA buffer 的 view
- `tensor.clone()` 确保数据拷贝到独立 CUDA tensor
- NCCL 和 Mooncake 使用相同的 `get_named_tensor_buckets` 代码，NCCL 正常 → 此函数无问题

#### 3. SGLang load_weights ✅ 正确

从 SGLang server log 中提取：
- **初始加载 (call #0):** `total=1199, stacked=368, matched=816, skipped=0`
- **Weight sync (calls #1-#19):** 19 次 load_weights 调用，所有调用 `skipped=0`
- **最后一次 (call #19):** `total=1, stacked=0, matched=1, skipped=0`
- **结论:** 所有参数都被 SGLang 正确接收和加载

#### 4. SGLang 内部模型权重 ✅ 正确（关键验证）

**Replicated weights（所有 TP rank 相同）：**

| 权重 | 原始模型 norm | SGLang (all ranks) norm | 匹配 |
|------|-------------|----------------------|------|
| layers.0.input_layernorm [5120] | 3.513089 | 3.513090 | ✅ |
| layers.15.input_layernorm [5120] | 6.044116 | 6.044119 | ✅ |
| layers.30.input_layernorm [5120] | 11.868532 | 11.868538 | ✅ |
| model.norm [5120] | 69.513885 | 69.513969 | ✅ |

**Sharded weights（TP=8 分片）：**

| 权重 | 原始 shard norms | SGLang shard norms | 匹配 |
|------|-----------------|-------------------|------|
| embed_tokens [248320,5120] | [172.00, 159.88, 151.36, 166.21, 153.14, 138.23, 131.28, 123.76] | [175.54, 163.45, 154.88, 169.72, 156.66, 141.26, 134.04, 126.24] | ✅ (bf16 精度内) |
| lm_head [248320,5120] | [170.28, 169.96, 171.05, 174.43, 174.15, 160.17, 156.64, 157.88] | [173.80, 173.41, 174.52, 178.03, 177.57, 163.97, 160.14, 161.42] | ✅ (bf16 精度内) |

- Replicated weights: 所有 8 个 TP rank 完全一致，与原始模型仅差 bf16 rounding
- Sharded weights: 每个 rank 的 shard norm 与原始模型对应 shard 精确匹配
- **结论: Mooncake weight sync 后 SGLang 内部模型参数完全正确**

#### 5. flush_cache / KV cache 处理 ✅ 无差异

- `flush_cache`: HTTP GET 请求到 SGLang server，NCCL 和 Mooncake 使用相同代码路径
- `free_cache_engine=False` 时:
  - `release_kv_cache()` → 直接 return（no-op）
  - `resume_kv_cache()` → 直接 return（no-op）
  - 仅依赖 `flush_cache` 清理 KV cache entries
- NCCL 和 Mooncake 的 cache 处理完全相同 → 不是差异来源

### 排除清单

| 组件 | 状态 | 理由 |
|------|------|------|
| FSDP 权重提取 (`get_per_tensor_param`) | ✅ 排除 | NCCL 共享代码，NCCL 正常 |
| Mooncake RDMA 传输 | ✅ 排除 | [CKSUM-RECV] 全部 OK |
| `get_named_tensor_buckets` (clone) | ✅ 排除 | 共享代码 + clone 确保数据独立 |
| `convert_weight_keys` 参数名映射 | ✅ 排除 | 共享代码 |
| SGLang `load_weights` | ✅ 排除 | skipped=0, NCCL 正常 |
| `/dev/shm` IPC 传输 | ✅ 排除 | IPC-FIX-v6 写入成功, NCCL 正常 |
| `flush_cache` | ✅ 排除 | 共享代码 |
| KV cache release/resume | ✅ 排除 | `free_cache_engine=False` 时都是 no-op |

### 剩余可疑方向

排除了所有 weight sync 路径上的组件后，剩余方向：

1. **Mooncake TransferEngine RDMA 内存注册干扰 GPU 状态**
   - `batch_register_memory` 将 GPU buffer 注册为 RDMA pinned memory
   - 可能影响 CUDA context 或 SGLang 的 GPU 内存分配
   - 验证方法: 在 TransferEngine 初始化后、weight sync 前测试推理

2. **KV Cache 内部状态（block 索引 / position metadata）被破坏**
   - `flush_cache` 只清理 cache entries，不重置 block allocator
   - Mooncake RDMA 操作可能通过共享 GPU 内存干扰 block 索引
   - 验证方法: weight sync 后重建 cache engine（`free_cache_engine=True`）

3. **SGLang attention kernel 与 Mooncake RDMA 的 GPU 交互**
   - Mooncake 使用 CUDA stream 做异步传输
   - 可能与 SGLang 的 attention kernel 有 stream 同步问题
   - 验证方法: 在 weight sync 后添加 `torch.cuda.synchronize()` 测试

4. **Mooncake TransferEngine 初始化本身的问题**
   - 可能不需要 weight sync，仅 TransferEngine 初始化就导致问题
   - 验证方法: 启动时初始化 TransferEngine 但不做 weight sync，测试推理

### 建议的下一步实验

**实验 A: weight sync 前后对比推理**
1. 在第一次 weight sync **之前**发送测试 prompt 到 SGLang server
2. 在 weight sync + flush_cache **之后**发送相同 prompt
3. 对比两次输出:
   - sync 前就乱码 → 问题是 Mooncake 初始化（TransferEngine RDMA 注册）
   - sync 后才乱码 → 问题在 weight sync 的副作用

**实验 B: free_cache_engine=True 对比**
- 设置 `free_cache_engine=True`，使 weight sync 完整释放和重建 KV cache
- 如果正常 → 确认是 KV cache 状态问题

**实验 C: 最小复现**
- 只初始化 Mooncake TransferEngine，不做 weight sync
- 如果乱码 → 确认是 TransferEngine 的 RDMA 注册干扰

---

## DIAG-GEN / DIAG-POST 实验记录 — 2026-06-17

### 实验背景

上一轮排查已确认 Mooncake RDMA 数据路径完全正确（checksum 匹配），但 weight sync 后 generation 输出乱码。为定位根因，设计了在 weight sync 前后发送测试推理请求的实验。

### 实验 1: DIAG-GEN（在 ServerAdapter.update_weights 内部）

**方法：** 在 `sglang_rollout.py` 的 `ServerAdapter.update_weights()` 中，weight sync 循环前后各插入 `_diag_test_generate("BEFORE_SYNC")` 和 `_diag_test_generate("AFTER_SYNC")` 调用，通过 HTTP 向 SGLang server 发送 `/v1/chat/completions` 请求。

**结果：** BEFORE_SYNC 和 AFTER_SYNC **全部 TimeoutError (60s)**。

**根因：** `ServerAdapter.update_weights()` 运行在 SGLang server 的 asyncio 事件循环内部（通过 `load_weights` HTTP 端点调用）。事件循环被 weight sync 完全阻塞，无法处理任何新的 HTTP 请求。这是方法论缺陷，不是 bug。

**时间线 (experiment mooncake_diag_gen):**
| 时间 | 事件 |
|------|------|
| 04:40:20 | init_process_group 完成 |
| 04:40:21 | BEFORE_SYNC 开始 |
| 04:41:21 | BEFORE_SYNC 超时 (60s) |
| 04:41:21~04:43:57 | Mooncake weight sync (216.64s, 0.24 GB/s) |
| 04:44:05 | AFTER_SYNC 开始 |
| 04:45:05 | AFTER_SYNC 超时 (60s) |

### 实验 2: DIAG-POST（在 CheckpointEngineWorker.update_weights 中）

**方法：** 将测试移到 `CheckpointEngineWorker.update_weights()`（独立 Ray actor，不在 SGLang 事件循环内）：
- `EARLY` 测试：在 `receive_weights()` 之前
- `LATE` 测试：在 `server_adapter.update_weights()` 之后

**修改文件：**
- `/workspace/verl/checkpoint_engine/base.py` — 添加 `_diag_post_sync_test()` 方法 + logging import
- `/workspace/verl/workers/rollout/sglang_rollout/sglang_rollout.py` — 注释掉旧 DIAG-GEN 调用

**结果 (experiment mooncake_early_late):**

1. **EARLY 测试 (05:44:22)：所有 8 个 worker 全部 "no _engine, skip"**
   - 原因：`_init_server_adapter()` 是在 `server_adapter.update_weights()` 内部懒加载的
   - EARLY 测试在 `update_weights()` 调用之前执行，此时 `_engine` 尚未初始化

2. **LATE 测试 (05:47:10~05:48:11)：TP leader (rank 0) 仍然 TimeoutError**
   - rank 0 的 `_engine` 已初始化（由 `server_adapter.update_weights()` 触发）
   - HTTP 请求发出后 60s 内无响应
   - 非 leader rank 全部 "no _engine, skip"

3. **关键发现：`[DEBUG-INIT-GENERATE]` 日志**
   ```
   [2026-06-17 05:44:22] [DEBUG-INIT-GENERATE] After model load, test output:
   "Here's a thinking process:\n\n1.  **Analyze User Input:**\n   - Question"
   ```
   - **SGLang server 在 weight sync 之前可以正常生成连贯英文文本**
   - 证明 SGLang server 初始状态是正确的

4. **Mooncake RDMA checksum 全部 OK** — 数据传输无误

### 核心发现

```
SGLang server 生命周期:
  启动 → 加载模型 → 正常生成 ✅ → Mooncake weight sync → 完全无响应 ❌
```

**SGLang server 在 Mooncake weight sync 后对所有 HTTP 请求无响应（60s+ 超时）。**

可能的根因：
1. SGLang server 的事件循环被 weight update 的 `load_weights` 操作永久阻塞（deadlock）
2. Mooncake TransferEngine 的 RDMA 内存注册 (`batch_register_memory`) 干扰了 SGLang server 的 GPU 状态
3. `update_weights_from_tensor` 内部的异步操作未完成，事件循环被挂起
4. NCCL 模式下 `load_weights` 可能快速完成（cupy broadcast 直接更新 GPU 参数），而 Mooncake 模式下 HTTP-based weight update 触发了不同的代码路径

### 实验 D: DIRECT_RAY（绕过 HTTP 层直接调用 SGLangHttpServer.generate）

**方法：** 在 `CheckpointEngineWorker.update_weights()` 的 LATE 测试之后，增加 `_diag_direct_ray_test()`：
- 通过 `ray.get_actor("sglang_server_0_0")` 获取 SGLangHttpServer actor handle
- 调用 `server_actor.generate.remote(prompt_ids, sampling_params, request_id)` 直接执行推理
- 绕过 HTTP 层，直接与 SGLang 内部 scheduler 通信

**修改文件：**
- `/workspace/verl/checkpoint_engine/base.py` — 添加 `_diag_direct_ray_test()` + v2 修复 (unique request_id, fallback tokenizer)

**结果 (experiment mooncake_direct_ray_v2, 2026-06-17 06:49-06:53):**

| 测试 | Worker | 结果 |
|------|--------|------|
| EARLY_s0 | 全部 8 workers | "no _engine, skip" (懒加载，预期行为) |
| LATE_s0 | 非 TP leader (7 workers) | "no _engine, skip" (预期行为) |
| LATE_s0 | TP leader (pid=1118118) | **TimeoutError** (HTTP 超时 60s，复现之前发现) |
| DIRECT_s0 | 全部 8 workers | 找到 actor `sglang_server_0_0` ✅, tokenized 13 tokens ✅ |
| DIRECT_s0 | TP leader (pid=1118123) | **`TIMEOUT: SGLang scheduler stuck (45s)! Confirms scheduler is blocked`** |
| DIRECT_s0 | 其他 7 workers | "Duplicate request ID" (竞争同一 request_id，非关键) |

**关键发现：**

1. **SGLang scheduler 本身卡死** — 不是 HTTP 层问题
   - HTTP 请求超时 → 可能是 HTTP 层被阻塞
   - Ray actor method 直接调用也超时 → **SGLang scheduler 进程本身无法处理新请求**

2. **GPU 利用率 0%** — scheduler 在 CPU 侧空转
   ```
   GPU 0-7: 0% utilization, 74GB memory used
   SGLang scheduler processes: 56-72% CPU each (8 processes)
   ```
   - GPU 完全空闲，说明不是 CUDA 操作卡住
   - Scheduler 进程 CPU 占用高但无法调度推理 → CPU 侧死锁或忙等

3. **结论更新：**
   ```
   SGLang server 生命周期:
     启动 → 加载模型 → 正常生成 ✅
     → Mooncake weight sync (load_weights HTTP + flush_cache)
     → Scheduler 进程 CPU 侧死锁 ❌ (GPU 0%, HTTP 和 Ray 均无响应)
   ```

### 更新后的排查方向

排除了 HTTP 层因素后，根因缩小到 SGLang scheduler 子进程：

1. **Scheduler 与 Tokenizer Manager 之间的 ZMQ 通信死锁**
   - `update_weights` 可能在 tokenizer manager 侧持有锁
   - Scheduler 发送的请求无法到达 tokenizer manager → 死锁
   - 验证: 在 `update_weights` 完成后检查 ZMQ socket 状态

2. **`flush_cache` 操作的副作用**
   - `flush_cache` 可能发送一个需要 scheduler 确认的请求
   - 如果 scheduler 正在处理 weight update 的内部回调 → 无法响应 flush_cache → 死锁
   - 验证: 移除 `flush_cache` 调用，测试推理是否恢复

3. **SGLang scheduler 子进程的内部状态**
   - `load_weights` 可能在 scheduler 子进程内部触发了 CUDA graph 重建
   - 重建过程可能在 Mooncake RDMA 注册后失败（CUDA IPC 冲突）
   - 验证: 对比 NCCL 模式下 scheduler 状态

### 建议的下一步实验

**实验 E: 移除 flush_cache 测试**
- 在 `ServerAdapter.update_weights()` 中注释掉 `flush_cache` 调用
- 如果 scheduler 恢复响应 → 确认是 flush_cache 导致死锁

**实验 F: 对比 NCCL 模式的 scheduler 状态**
- 在 NCCL weight sync 后检查 scheduler 是否仍然响应
- 如果 NCCL 模式下 scheduler 正常 → 确认是 Mooncake 的 weight update 路径问题

---

## Pending issues (更新 2026-06-18)

1. ~~**mRoPE nested tensor bug:**~~ ✅ 根因修复 (Patch k)，已验证 Run #42 训练步通过。
2. **Reward 全为 0:** 待调查。verifier 评分逻辑可能有问题，或 SWE-agent patch 质量不足。
3. **Pod 重建后 patch 丢失:** 每次容器重启都需要重新 apply patch。已在 Dockerfile 中 bake 了大部分 patch（override 文件方式）。`tensordict_utils.py` 的 Patch k 需要合入 verl 源码。
4. **kubernetes 包:** 需要加入 Dockerfile（training worker 的 harbor 环境依赖）。
5. **Ray cluster 清理:** 多次 run 之间需要清理 Ray actors 释放 GPU。`ray stop --force` 可能导致容器重启。
6. **Mooncake weight sync 乱码: ❌ 未解决。** `flush_cache=False` + `resume_generation` 重排序仅解决了 scheduler CPU 死锁（scheduler 恢复响应），但乱码问题仍然存在——`response_length=4096` 打满 max 就是乱码未解决的直接证据（对比 NCCL: response_length=463，generation 正常）。权重数据已验证正确（norm 精确匹配、RDMA checksum OK），问题在 generation 侧：可能是 Mooncake TransferEngine RDMA 内存注册干扰 KV cache / GPU 状态。建议实验: `free_cache_engine=True` 对比、仅初始化 TransferEngine 不做 weight sync 测试推理。
7. **`maybe_fix_3d_position_ids` Case B:** Run #42 修复只覆盖了 offsets 追踪 mRoPE dim 的情况（Case A）。当 offsets 追踪 seq_len dim 时（Case B），tensor 已经是 `ragged_idx=2` 正确状态，只需恢复 `_ragged_idx` 属性。已修复并部署。
8. **Sandbox pod 清理:** 每次启动训练前需要清理上一轮 agent loop 创建的 sandbox pod（Django 环境 pod）。
9. **Tool parser 配置不匹配:** `tool_format=hermes` 与 Qwen3-Coder 的 XML tool call 格式不兼容，需改为 `qwen3_coder`（见下方 Patches）。

---

## Patches — 2026-06-22 (tool_format + NCCL env + 调试日志)

### 背景

Run #44 验证 NCCL weight sync 正常后，重新部署运行 agentic 训练时发现两个问题：
1. **`Failed to decode tool call: Expecting value`** — 模型生成 SWE-agent XML tool call 格式（`function=bash / parameter=command`），但 `hermes` parser 期望 JSON 格式，`json.loads()` 失败。
2. **NCCL IPv6 RDMA 环境变量缺失** — Wulan 集群使用 IPv6 RDMA，需要 `NCCL_IB_ADDR_RANGE` 等环境变量，否则 NCCL 通信失败。

### Patch 1: NCCL IPv6 RDMA 环境变量

**文件:** `ray-cluster-wulan.yaml`
- training-workers 和 rollout-workers 的 env 中都添加了 6 个 NCCL 变量：
  - `NCCL_DEBUG=INFO`（从 WARN 改为 INFO）
  - `NCCL_IB_ADDR_RANGE=2001:db8:80f:e000::/60`
  - `NCCL_IB_ADDR_FAMILY=AF_INET6`
  - `GLOO_SOCKET_IFNAME=eth0`
  - `NCCL_NET_PLUGIN=none`
  - `CUDA_DEVICE_MAX_CONNECTIONS=1`

**文件:** `verl/checkpoint_engine/nccl_checkpoint_engine.py`
- 头部添加 `os.environ.setdefault()` 注入同样的 6 个变量（作为代码级兜底，不覆盖 YAML 已有值）
- 遵守 `NCCL_SOCKET_IFNAME` 禁止代码修改的约束

### Patch 2: tool_format 从 hermes 改为 qwen3_coder

**根因:** Qwen3-Coder 模型生成 XML tool call 格式，但 `hermes` parser 只能解析 JSON 格式。通过调试日志确认 match 内容后定位。

**修改文件（4处）:**

| 文件 | 修改 |
|------|------|
| `recipe/agentic/run_agentic_disagg.sh` L54 | `TOOL_FORMAT` 默认值 `hermes` -> `qwen3_coder` |
| `recipe/agentic/config/agentic_trainer_disagg.yaml` L77 | `tool_format: "hermes"` -> `"qwen3_coder"` |
| `recipe/agentic/config/agentic_trainer_k8s.yaml` L24 | `tool_format: "hermes"` -> `"qwen3_coder"` |
| `verl/trainer/config/rollout/rollout.yaml` L218 | `format: hermes` -> `format: qwen3_coder` |

`run_agentic_disagg.sh` 中 `TOOL_FORMAT` 同时控制 `actor_rollout_ref.rollout.multi_turn.format` 和 `proxy_server.tool_format`，保证两端 parser 一致。

### Patch 3: InferenceWorkerClient tool_format 传参

**文件:** `recipe/agentic/agent_loop/remote_agent_loop.py`
- `_ensure_inference_worker()` 中 `InferenceWorkerClient` 初始化时未传入 `tool_format`，导致内部默认回退到 `hermes`
- 修复：从 `self.config.actor_rollout_ref.rollout.multi_turn.format` 读取并传入

### Patch 4: tool_parser.py 调试日志

**文件:** `verl/experimental/agent_loop/tool_parser.py`
- `HermesToolParser.extract_tool_calls()` 添加 debug 日志输出 raw match 内容
- error 日志追加 `| match=...` 显示导致解析失败的具体内容
- 用于确认模型实际输出格式（最终确认是 XML 而非 JSON）

### 部署方式

所有修改通过 `kubectl cp` 拷贝到两个 worker pod：
- `verl-disagg-mooncake-training-workers-worker-jph7j`
- `verl-disagg-mooncake-rollout-workers-worker-bn2kc`

目标路径：
- `/workspace/verl/checkpoint_engine/nccl_checkpoint_engine.py`
- `/workspace/recipe/agentic/agent_loop/remote_agent_loop.py`
- `/workspace/verl/experimental/agent_loop/tool_parser.py`

Python 模块更新后需重启训练进程才生效。

### 验证

在 training pod 上运行 `test_tool_parser.py` 验证：
- `qwen3_coder` parser 能正确找到 XML function calls 并解析（`tools=None` 时会 crash，但实际训练流程中 tools 由 agent loop 传入）
- `hermes` parser 对 XML 内容 FAIL（json parse error），对 JSON 内容 PASS
- 确认问题根因是配置错误，不是模块版本问题

### 拷贝脚本更新

`recipe/agentic/copy_proxy_logs_to_pods.sh` 增强：
- 添加 `nccl_checkpoint_engine.py` 和 `mooncake_checkpoint_engine.py` 到 `CKPT_FILES`
- 重构为 `copy_group()` 函数，支持多组文件（proxyserver + checkpoint_engine）
- 修复 `local find_hint="$1" shift` 缺少分号的 bash 语法 bug

---

## Patches — 2026-06-22 (tool_parser + vllm_provider 修复)

### Patch 5: Qwen3XMLToolParser tools=None 防护

**文件:** `verl/experimental/agent_loop/tool_parser.py`
- `get_arguments_config()` 添加 `if not tools: return {}` 防护
- 根因: `vllm_provider.py` 调用 `extract_tool_calls(token_ids)` 时未传 `tools` 参数，导致 `for config in tools` 崩溃 (`TypeError: 'NoneType' object is not iterable`)
- 修复后 `tools=None` 时跳过参数类型转换，直接返回原始字符串值

### Patch 6: vllm_provider.py tools 参数传递链路修复

**文件:** `recipe/agentic/proxyserver/vllm_provider.py`

**问题:** `acompletion()` 中 `tools` 从 `optional_params.get("tools")` 取出后，只传给了 `_generate()`（用于 vLLM generation），但 `_parse_tool_calls()` 方法签名中根本没有 `tools` 参数，导致 tool parser 收不到 tools schema。

**修复（3处）:**
1. `_parse_tool_calls` 方法签名添加 `tools: list[dict] | None = None`
2. 内部调用 `extract_tool_calls(token_ids, tools=tools)` 传入 tools
3. `acompletion` 调用 `_parse_tool_calls(token_ids, completion_text, tools=tools)`

**完整链路修复后:**
```
acompletion(tools=...) -> _parse_tool_calls(tools=...) -> extract_tool_calls(tools=...) -> Qwen3XMLToolParser
```

### Patch 7: vllm_provider.py tools schema 防御性清理

**文件:** `recipe/agentic/proxyserver/vllm_provider.py`

**问题:** `apply_chat_template` 的 Qwen3 Jinja 模板对 `tool.function.parameters.properties` 执行 `|items` 遍历，要求必须是 dict。SWE-agent 传入的 tools schema 中某些 tool 的 `properties` 类型不是 dict，导致 `TypeError: Can only get item pairs from a mapping.`

**修复:** 添加 `_sanitize_tools()` 静态方法，在 `apply_chat_template` 前清理 tools schema：
- `properties` 不是 dict → 强制转为 `{}`
- `parameters` 不是 dict → 强制转为 `{}`
- 每次转换打 `[SANITIZE]` warning 日志

### 部署与运行

- 所有 patch 通过 `kubectl cp` 部署到 training/rollout 两个 worker pod
- pod 中旧版 `/workspace/run_agentic_disagg.sh` 仍为 `hermes` 默认值，需通过环境变量覆盖:
  ```bash
  HARBOR_DATA_DIR=/mnt/data/swe-bench-quick-4 \
  TOOL_FORMAT=qwen3_coder \
  CHECKPOINT_ENGINE_BACKEND=nccl \
  bash run_agentic_disagg.sh
  ```
- NCCL backend 已验证启动正常，`tool_format=qwen3_coder` 配置正确传入
