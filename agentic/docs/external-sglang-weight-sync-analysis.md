# VeRL 对接外部 SGLang 权重同步方案对比分析

## 1. 背景与目标

当前 VeRL 的权重同步架构依赖 Ray 管理 SGLang 的生命周期：训练端通过 Checkpoint Engine (CE) 将权重发给 CE Worker（与 SGLang 同 GPU），CE Worker 再通过 CUDA IPC 推送到 SGLang。

```
当前架构（两跳传输）：

Trainer (FSDP rank 0)
    │  get_per_tensor_param() → 全量未分片权重
    │
    ▼  NCCL broadcast / Mooncake chain
CE Worker (与 SGLang 同 GPU)
    │  receive_weights() → (name, tensor) generator
    │
    ▼  CUDA IPC (sgl_update_weights)
SGLang Server
    │  model.load_weights() → TP 自动分片
    ▼
推理就绪
```

目标：将 SGLang 解耦为独立部署的推理服务，由外部控制器管理（不在 VeRL Ray 集群内），VeRL 只负责训练和权重同步。

本文对比两种权重同步方案：
- **方案 A**：NCCL（利用 SGLang 已有的 HTTP API）
- **方案 B**：Mooncake TransferEngine（RDMA 直传）

---

## 2. 关键前置分析：权重格式兼容性

### 2.1 VeRL 训练端输出格式

VeRL 通过 `get_per_tensor_param()` 提取权重：

```
文件：verl/workers/engine/fsdp/transformer_impl.py:838-846

(name, param.full_tensor().to(torch.bfloat16))
    ↑           ↑
    │           └─ DTensor.full_tensor()：FSDP all-gather，还原完整参数
    └─ HuggingFace 格式的参数名（如 model.layers.0.self_attn.q_proj.weight）
```

**结论：VeRL 始终输出全量未分片的 HF 格式权重。**

### 2.2 SGLang 的权重加载方式

SGLang 有两条权重加载路径，它们对输入格式的要求不同：

| 路径 | 输入格式 | TP 分片方式 | 用途 |
|------|---------|------------|------|
| `model.load_weights()` | **全量未分片** | 内部通过 `weight_loader` 按 TP rank 自动切片 | RLHF 权重更新、初始化加载 |
| `batch_transfer_sync_read` (Mooncake) | **已 TP 分片** | 直接拷贝，不做分片 | SGLang 实例间权重克隆 |

`load_weights()` 内部处理：
- `ColumnParallelLinear`：按 `output_dim` 切片，`start_idx = tp_rank * shard_size`
- `RowParallelLinear`：按 `input_dim` 切片
- QKV 融合：`q_proj + k_proj + v_proj → qkv_proj`（名称映射 + 偏移计算）
- `gate_proj + up_proj → gate_up_proj`

### 2.3 核心约束

**VeRL 输出全量权重 → 必须经过 `model.load_weights()` 才能正确加载到 SGLang。**

这意味着任何方案的最后一步都必须调用 `load_weights()` 来处理 TP 分片和参数名映射。直接写入 SGLang 的 GPU 内存（跳过 `load_weights()`）是不可行的，除非 VeRL 侧重新实现 SGLang 的全部 `weight_loader` 逻辑。

---

## 3. 方案 A：NCCL（利用 SGLang 已有 HTTP API）

### 3.1 架构

```
目标架构：

Trainer (FSDP rank 0)
    │  get_per_tensor_param() → 全量权重
    │
    │  ┌──────── HTTP ────────┐
    │  │ /init_weights_update_group     SGLang TP rank 0,1,...,N-1
    │  │ (master_addr, port,            各自加入 NCCL group
    │  │  rank_offset, world_size)      rank = rank_offset + tp_rank
    │  └──────────────────────┘
    │
    ▼  torch.distributed.broadcast (NCCL group, src=0)
SGLang 各 TP Worker
    │  接收全量权重
    │  model.load_weights() → TP 自动分片
    │
    │  ┌──────── HTTP ────────┐
    │  │ /update_weights_from_distributed
    │  │ (names, dtypes, shapes,
    │  │  group_name, flush_cache)
    │  └──────────────────────┘
    ▼
推理就绪
```

### 3.2 可行性验证

**结论：可行。SGLang 已有完整的 HTTP API 链路，且有端到端测试验证。**

依据：

1. **SGLang 已有全部所需 HTTP 端点：**
   - `POST /init_weights_update_group` — 创建 NCCL group（`io_struct.py:1601`）
   - `POST /update_weights_from_distributed` — 触发 NCCL 接收 + `load_weights()`（`io_struct.py:1476`）
   - `POST /destroy_weights_update_group` — 销毁 NCCL group（`io_struct.py:1623`）
   - `POST /abort_request` — 中止推理请求
   - `GET|POST /release_memory_occupation` — 释放 KV cache
   - `GET|POST /resume_memory_occupation` — 恢复 KV cache
   - `GET|POST /flush_cache` — 刷新缓存

2. **NCCL group 创建协议已标准化：**
   - SGLang 使用 `init_custom_process_group()`（`utils/common.py:1926`），底层是 `torch.distributed` TCP rendezvous
   - 训练端使用相同的 `init_custom_process_group()` 即可加入同一 group
   - 测试文件 `test_update_weights_from_distributed.py:229-235` 展示了完整的 rank 0 训练端创建方式

3. **权重格式完全兼容：**
   - 训练端 broadcast 全量 HF 格式权重
   - SGLang 的 `update_weights_from_distributed()`（`model_runner.py:1884`）接收后调用 `model.load_weights()` 自动处理 TP 分片
   - 支持 `flattened_bucket` 格式批量传输（`model_runner.py:1943`）

4. **端到端测试已验证：**
   - `test_update_weights_from_distributed.py` 覆盖了：训练进程创建 NCCL group → SGLang 通过 HTTP 加入 → broadcast 权重 → 验证更新正确

### 3.3 VeRL 侧改动

#### 新增文件（1 个）

| 文件 | 说明 |
|------|------|
| `verl/checkpoint_engine/sglang_direct_nccl_engine.py` | 新的 CE 后端，直接通过 NCCL 与外部 SGLang 通信 |

核心实现：

```python
# 伪代码 - 关键逻辑
class SGLangDirectNCCLEngine(CheckpointEngine):
    def prepare(self):
        # 分配 TCP rendezvous 端口
        return {"addr": hostname, "port": free_port}

    def build_topology(self, trainer_world_size, rollout_world_size, metadatas):
        # trainer rank 0 = NCCL rank 0
        # SGLang TP workers = NCCL rank 1..N (通过 HTTP 通知)
        ...

    def init_process_group(self, rank, world_size, metadata):
        if rank == 0:  # 训练端
            self.group = init_custom_process_group(
                backend="nccl", init_method=f"tcp://{addr}:{port}",
                world_size=world_size, rank=0, group_name=self.group_name)
        # SGLang 端通过 HTTP /init_weights_update_group 已自行加入

    def send_weights(self, weights, global_steps):
        # 方式一：逐参数 broadcast
        for name, tensor in weights:
            torch.distributed.broadcast(tensor, src=0, group=self.group)
        # 方式二：flattened_bucket 批量 broadcast（性能更好）
        ...

    def finalize(self):
        torch.distributed.destroy_process_group(self.group)
        # HTTP POST /destroy_weights_update_group
```

#### 修改文件（2-3 个）

| 文件 | 改动 | 说明 |
|------|------|------|
| `verl/checkpoint_engine/base.py` | `CheckpointEngineManager` | 新增 `external_sglang` 模式：跳过 CE Worker 创建，用 HTTP 代替 Ray RPC 做生命周期管理（abort/release_kv/resume_kv） |
| `verl/workers/config/rollout.py` | `CheckpointEngineConfig` | 新增 `sglang_direct_nccl` 后端选项，增加 `sglang_endpoints` 配置字段（SGLang 各实例 URL） |
| `verl/trainer/ppo/ray_trainer.py` | `update_weights` 调用处 | 适配新的 `CheckpointEngineManager` 模式（如果 manager 接口不变则无需改动） |

#### 不需要改动的文件

- `sglang_rollout.py` — 不再使用 `ServerAdapter`（CUDA IPC 路径被跳过）
- `nccl_checkpoint_engine.py` — 保持不变，原有路径不受影响
- `mooncake_checkpoint_engine.py` — 保持不变

### 3.4 SGLang 侧改动

**无需改动。** 所有需要的 HTTP 端点、NCCL group 管理、`load_weights()` TP 分片逻辑均已存在。

### 3.5 影响面

| 维度 | 影响 |
|------|------|
| 现有 colocated 路径 | 无影响，新增独立后端 |
| 现有 NCCL/Mooncake CE | 无影响，不修改已有代码 |
| 配置 | 新增 `checkpoint_engine.backend = "sglang_direct_nccl"` + `sglang_endpoints` |
| 依赖 | 无新依赖，使用已有的 `torch.distributed` + `aiohttp`/`requests` |
| 测试 | 可复用 SGLang 已有的 `test_update_weights_from_distributed.py` 模式 |

### 3.6 风险

| 风险 | 严重程度 | 说明与缓解 |
|------|---------|-----------|
| NCCL group 创建失败 | 高 | 跨网络 TCP rendezvous 可能因防火墙/网络分区失败。**缓解**：增加重试 + 超时配置，提供诊断日志 |
| NCCL group 静态成员约束 | 中 | group 创建后不能动态增减成员。SGLang 扩缩容需 destroy/rebuild group。**缓解**：VeRL 已有 `rebuild_group` 模式（`finalize` 时销毁，下次 `build_process_group` 重建）|
| 广播冗余数据 | 低 | 全量权重 broadcast 到每个 TP rank，每个 rank 只用 1/tp_size。TP=8 时浪费 7/8 带宽。**缓解**：对大规模部署可考虑 scatter 优化，但 MVP 阶段可接受 |
| 生命周期协调复杂性 | 中 | abort → release_kv → broadcast → flush_cache → resume_kv 需要通过 HTTP 而非 Ray RPC 协调。**缓解**：SGLang 已有所有对应 HTTP 端点，可逐步实现 |
| 训练端 NCCL 与 FSDP NCCL 冲突 | 中 | 训练端可能有多个 NCCL communicator。**缓解**：使用独立的 group_name 隔离，SGLang 的 `init_custom_process_group` 已处理此场景 |
| NCCL 端口占用 | 低 | TCP rendezvous 需要可达端口。**缓解**：使用动态端口分配 |

---

## 4. 方案 B：Mooncake TransferEngine（RDMA 直传）

### 4.1 架构

```
目标架构：

Trainer (FSDP rank 0)
    │  get_per_tensor_param() → 全量权重
    │  打包到 registered bucket buffer
    │
    │  ┌──────── HTTP ────────┐
    │  │ 新端点：/init_mooncake_weight_update
    │  │ 返回 SGLang 的 session_id +
    │  │ receive_buffer 地址              SGLang 各 TP Worker
    │  └──────────────────────┘           分配 receive_buffer
    │                                     register_memory()
    │
    ▼  transfer_sync_write (RDMA P2P)
SGLang receive_buffer (已注册的 GPU 内存)
    │  收到全量权重
    │  反序列化 bucket → (name, tensor) 列表
    │  model.load_weights() → TP 自动分片
    │
    │  ┌──────── HTTP ────────┐
    │  │ 新端点：/finalize_mooncake_weight_update
    │  │ (flush_cache=True)
    │  └──────────────────────┘
    ▼
推理就绪
```

### 4.2 可行性验证

**结论：有条件可行，但需要 SGLang 侧新增接收端点和缓冲区管理。**

**可行的部分：**

1. **传输层基础设施已就绪：**
   - VeRL 已有 `MooncakeCheckpointEngine`，TransferEngine 初始化、`register_memory`、`transfer_sync_write` 均已实现
   - SGLang 已有 TransferEngine 初始化（`model_runner.py:961-979`）

2. **RDMA 传输本身可行：**
   - VeRL 训练端的 bucket buffer 已注册到 Mooncake（`mooncake_checkpoint_engine.py:85-89`）
   - SGLang 可以新分配一个 receive buffer 并注册
   - `transfer_sync_write` / `transfer_sync_read` 可在两端间传输数据

**不可行的部分（需新开发）：**

3. **SGLang 没有 Mooncake 权重接收端点：**
   - SGLang 现有的 Mooncake 路径（`remote_instance_weight_loader`）仅用于实例间权重克隆（读取已分片权重），不适用于接收全量权重
   - 需要新增 HTTP 端点让 SGLang 分配 receive buffer、接收数据、触发 `load_weights()`

4. **不能直接写入 SGLang 的已注册权重内存：**
   - SGLang `register_memory_region()` 注册的是 `model.named_parameters()` 的地址——这些是**已 TP 分片**的参数
   - VeRL 发送的是**全量未分片**权重，shape 不匹配
   - 例：TP=4 时，SGLang 的 `q_proj.weight` 大小为 `[hidden_size, hidden_size/4]`，而 VeRL 发送的是 `[hidden_size, hidden_size]`
   - 此外 SGLang 有参数融合（`q_proj+k_proj+v_proj → qkv_proj`），名称也不匹配

5. **需要额外的同步/通知机制：**
   - Mooncake 是纯数据传输，没有 barrier 语义
   - 需要信令通知 SGLang "数据已就绪，可以 load_weights()"
   - 需要协调多个 TP rank 的接收进度

### 4.3 VeRL 侧改动

#### 新增文件（1 个）

| 文件 | 说明 |
|------|------|
| `verl/checkpoint_engine/sglang_direct_mooncake_engine.py` | 新的 CE 后端，通过 Mooncake 直接向外部 SGLang 传输权重 |

核心实现：

```python
# 伪代码 - 关键逻辑
class SGLangDirectMooncakeEngine(CheckpointEngine):
    def prepare(self):
        self.engine = TransferEngine()
        self.engine.initialize(hostname, "P2PHANDSHAKE", "rdma", device_name)
        # 分配并注册 bucket buffer
        self.buf = torch.empty(2 * bucket_size, dtype=torch.uint8, device="cuda")
        self.engine.batch_register_memory([self.buf.data_ptr()], [2 * bucket_size])

    def init_process_group(self, rank, world_size, metadata):
        if rank == 0:  # 训练端
            # HTTP 调用 SGLang 新端点，获取各 TP rank 的 receive buffer 信息
            for tp_rank, url in enumerate(sglang_endpoints):
                resp = requests.post(f"{url}/init_mooncake_weight_update",
                    json={"bucket_size": self.bucket_size})
                self.sglang_sessions[tp_rank] = resp.json()
                # { "session_id": "...", "buffer_ptr": ..., "buffer_size": ... }

    def send_weights(self, weights, global_steps):
        # 与现有 MooncakeCheckpointEngine.send_weights 类似：
        # 打包权重到 bucket buffer → transfer_sync_write 到各 SGLang 的 buffer
        # 但改为并行写入（而非链式）
        for tp_rank in range(num_sglang_workers):
            session = self.sglang_sessions[tp_rank]
            self.engine.transfer_sync_write(
                session["session_id"],
                self.buf.data_ptr(), session["buffer_ptr"], data_len)
        # HTTP 通知各 SGLang: load_weights + flush_cache
        for url in sglang_endpoints:
            requests.post(f"{url}/finalize_mooncake_weight_update",
                json={"names": names, "dtypes": dtypes, "shapes": shapes})
```

#### 修改文件（2-3 个）

与方案 A 相同：

| 文件 | 改动 | 说明 |
|------|------|------|
| `verl/checkpoint_engine/base.py` | `CheckpointEngineManager` | 新增 `external_sglang_mooncake` 模式 |
| `verl/workers/config/rollout.py` | `CheckpointEngineConfig` | 新增 `sglang_direct_mooncake` 后端选项 |
| `verl/trainer/ppo/ray_trainer.py` | 同方案 A | 适配新模式 |

### 4.4 SGLang 侧改动

#### 需要新增的内容（约 3-4 个文件改动）

| 文件 | 改动 | 说明 |
|------|------|------|
| `srt/entrypoints/http_server.py` | 新增 2 个 HTTP 端点 | `POST /init_mooncake_weight_update`（分配 receive buffer，返回 session_id + ptr）<br>`POST /finalize_mooncake_weight_update`（从 buffer 解析权重，调用 `load_weights()`） |
| `srt/managers/io_struct.py` | 新增 2 个 Request/Response 数据结构 | `InitMooncakeWeightUpdateReqInput` / `FinalizeMooncakeWeightUpdateReqInput` |
| `srt/model_executor/model_runner.py` | 新增 2 个方法 | `init_mooncake_weight_update()`（分配 + 注册 buffer）<br>`finalize_mooncake_weight_update()`（从 buffer 读取权重 → `load_weights()`） |
| `srt/managers/tokenizer_manager.py` | 路由新端点到 model_runner | 添加新请求的转发逻辑 |

#### 关键实现细节

```python
# SGLang 侧伪代码
class ModelRunner:
    def init_mooncake_weight_update(self, bucket_size):
        if self.remote_instance_transfer_engine is None:
            self.remote_instance_init_transfer_engine()
        # 分配接收缓冲区
        self.weight_update_buf = torch.empty(
            2 * bucket_size, dtype=torch.uint8, device=self.device)
        self.remote_instance_transfer_engine.register_memory(
            self.weight_update_buf.data_ptr(), 2 * bucket_size)
        return {
            "session_id": self.remote_instance_transfer_engine_session_id,
            "buffer_ptr": self.weight_update_buf.data_ptr(),
            "buffer_size": 2 * bucket_size,
        }

    def finalize_mooncake_weight_update(self, names, dtypes, shapes, offsets):
        # 从 buffer 中解析权重
        weights = []
        for name, dtype, shape, offset in zip(names, dtypes, shapes, offsets):
            tensor = self.weight_update_buf[offset:offset+nbytes].view(dtype).view(shape)
            weights.append((name, tensor))
        # 调用 load_weights 处理 TP 分片
        self.model.load_weights(weights)
```

### 4.5 影响面

| 维度 | 影响 |
|------|------|
| 现有 colocated 路径 | 无影响 |
| SGLang 代码库 | **需要提交 PR 到 SGLang 上游**，新增 2 个 HTTP 端点 + buffer 管理 |
| 配置 | 新增 `checkpoint_engine.backend = "sglang_direct_mooncake"` + Mooncake 配置 |
| 依赖 | 需要 Mooncake 库（`pip install mooncake`），训练端和推理端都需要安装 |
| 基础设施 | 需要 RDMA 网络或 Mooncake TCP 模式 + P2PHANDSHAKE 元数据服务 |
| 测试 | 需要新写端到端测试，无法复用 SGLang 现有测试 |

### 4.6 风险

| 风险 | 严重程度 | 说明与缓解 |
|------|---------|-----------|
| SGLang 上游接受度 | **高** | 需要向 SGLang 提 PR 新增 Mooncake 权重更新端点。SGLang 可能不愿接受仅服务于 VeRL 的端点。**缓解**：将其设计为通用的 RDMA 权重更新接口，适用于任何训练框架 |
| Buffer 生命周期管理 | 高 | receive buffer 的分配、注册、释放需要精确管理。RDMA 写入时 buffer 不能被释放。**缓解**：引用计数 + 写入完成信号 |
| 部分写入一致性 | **高** | Mooncake `transfer_sync_write` 没有原子性保证。如果传输中断，SGLang 的 buffer 可能处于不一致状态。**缓解**：double buffer + magic byte 完整性校验（参考 VeRL 现有实现） |
| TP rank 同步 | 中 | 多个 TP rank 各自接收，需确保所有 rank 都收完再 `load_weights()`。**缓解**：VeRL 侧汇总所有 rank 的 finalize 响应 |
| Mooncake 连接建立失败 | 中 | P2PHANDSHAKE 依赖两端可互相发现。跨集群部署需要网络可达性。**缓解**：提供 fallback 到 TCP 模式 |
| 调试困难 | 中 | RDMA 传输问题比 NCCL 更难诊断（无标准工具）。**缓解**：增加详细日志，记录每次传输的 session_id、ptr、size、耗时 |
| 内存开销 | 低 | 每个 SGLang TP rank 需要额外的 receive buffer（大小 = 2 × bucket_size，默认约 200MB）。**缓解**：可按需分配/释放 |

---

## 5. 对比总结

### 5.1 改动量对比

| 维度 | 方案 A：NCCL | 方案 B：Mooncake |
|------|-------------|-----------------|
| VeRL 新增文件 | 1 个 | 1 个 |
| VeRL 修改文件 | 2-3 个 | 2-3 个 |
| SGLang 改动 | **0 个** | **3-4 个文件**（需提 PR） |
| 新增代码量（估算） | ~300-400 行 | ~500-700 行（VeRL）+ ~200-300 行（SGLang） |
| 总改动量 | **小** | **中-大** |

### 5.2 改动位置对比

```
方案 A 改动范围：                        方案 B 改动范围：

verl/                                   verl/
├── checkpoint_engine/                  ├── checkpoint_engine/
│   ├── base.py [修改]                  │   ├── base.py [修改]
│   └── sglang_direct_nccl_engine.py    │   └── sglang_direct_mooncake_engine.py
│       [新增]                          │       [新增]
├── workers/config/rollout.py [修改]    ├── workers/config/rollout.py [修改]
└── trainer/ppo/ray_trainer.py          └── trainer/ppo/ray_trainer.py
    [可能修改]                              [可能修改]

                                        sglang/ (需提 PR 到上游)
                                        ├── srt/entrypoints/http_server.py [修改]
                                        ├── srt/managers/io_struct.py [修改]
                                        ├── srt/managers/tokenizer_manager.py [修改]
                                        └── srt/model_executor/model_runner.py [修改]
```

### 5.3 影响面对比

| 维度 | 方案 A：NCCL | 方案 B：Mooncake |
|------|-------------|-----------------|
| 对 VeRL 现有功能影响 | 无（纯新增后端） | 无（纯新增后端） |
| 对 SGLang 代码库影响 | 无 | 中（新增端点 + buffer 管理） |
| 跨项目协调 | 不需要 | **需要**（SGLang PR review + merge） |
| 基础设施要求 | NCCL 网络可达即可 | 需要 Mooncake + RDMA/TCP 网络 |
| 弹性扩缩容 | 需 destroy/rebuild NCCL group | 无 group 概念，天然弹性 |
| 部署复杂度 | 低 | 中（需配置 Mooncake 元数据服务） |

### 5.4 性能对比（理论分析）

| 维度 | 方案 A：NCCL | 方案 B：Mooncake |
|------|-------------|-----------------|
| 跨节点传输 | NCCL broadcast（tree/ring） | RDMA P2P 直传 |
| 带宽利用 | 全量权重广播到每个 TP rank，TP=8 时浪费 7/8 | 全量权重传到每个 TP rank（同样冗余），但 P2P 可并行 |
| 延迟 | NCCL 集合通信开销 + TCP rendezvous 建立 | RDMA 单次传输延迟更低，但需 buffer 管理开销 |
| 同节点场景 | NVLink NCCL 非常快 | 无优势，反而有 RDMA 注册开销 |
| Group 建立开销 | 每次 rebuild 约 1-3 秒 | 无 group，首次 session 建立约 0.5 秒 |

### 5.5 风险等级对比

| 风险类别 | 方案 A：NCCL | 方案 B：Mooncake |
|---------|-------------|-----------------|
| 技术可行性 | ✅ 已有端到端测试验证 | ⚠️ 需新开发，未经验证 |
| 上游依赖 | ✅ 无需 SGLang 改动 | ❌ 需 SGLang PR 被接受 |
| 数据一致性 | ✅ NCCL broadcast 有原子性保证 | ⚠️ RDMA write 无原子性，需自行保证 |
| 网络要求 | ✅ TCP 即可 | ⚠️ 最佳性能需 RDMA |
| 弹性伸缩 | ⚠️ 需 destroy/rebuild group | ✅ 天然弹性 |
| 调试难度 | ✅ NCCL 有成熟工具链 | ⚠️ RDMA 调试困难 |

---

## 6. 建议

### 短期（MVP）：方案 A（NCCL）

理由：
- SGLang 侧零改动，VeRL 单侧开发即可
- 已有端到端测试验证完整协议
- 技术风险最低，可快速验证外部 SGLang 的可行性
- 弹性扩缩容通过 destroy/rebuild group 解决（VeRL 已有此模式）

### 长期（性能优化）：方案 B（Mooncake）

适用场景：
- 跨数据中心/跨集群的大规模部署
- 需要频繁弹性扩缩容
- 已有 RDMA 基础设施
- SGLang 社区愿意接受通用 RDMA 权重更新端点

建议在方案 A 验证可行后，再根据实际性能瓶颈决定是否投入方案 B。

### 混合方案（可选）

保留 CE Worker 但连接外部 SGLang：CE Worker 仍部署在 SGLang 同节点上，通过 NCCL/Mooncake chain 接收权重，再通过 CUDA IPC 推送到 SGLang。这种方式对 VeRL 代码改动最小（几乎为零），但要求 CE Worker 和 SGLang 在同一节点同一 GPU 上，部署灵活性受限。
