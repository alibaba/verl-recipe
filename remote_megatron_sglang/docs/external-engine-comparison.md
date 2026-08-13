# External Engine 方案分析与对比

## 一、我们的方案：External SGLang

### 1.1 架构概述

External SGLang 是一个正在开发中的方案，目标是将 SGLang 推理引擎独立部署在训练集群之外，训练侧使用 verl 的 trainer 编排（FSDP），推理侧使用独立的 SGLang 服务。

架构分层：
- **训练侧**：verl 的 RayPPOTrainer / OneStepOffRayTrainer，使用 FSDP 训练引擎
- **推理侧**：独立部署的 SGLang 服务（可以是不同集群、不同网络）
- **权重同步**：NCCL-over-HTTP 或 Mooncake RDMA 两种路径（均已验证 standalone，NCCL 路径尚未完整训练验证）
- **请求转发**：ExternalSGLangProxyActor 通过 HTTP 将 generate 请求转发到外部 SGLang

### 1.2 核心组件

**verl 核心代码** (`verl/workers/rollout/external_sglang/`)：

| 文件 | 组件 | 职责 |
|------|------|------|
| `proxy_actor.py` | ExternalSGLangProxyActor | Ray Actor，将 generate 请求通过 HTTP 转发到外部 SGLang |
| `server_manager.py` | ExternalLLMServerManager | 从 external_sglang_endpoints 配置创建 shim actor，注册到 GlobalRequestLoadBalancer |
| `checkpoint_manager.py` | ExternalCheckpointManager + ReceiverCE | 通过 Mooncake RDMA 同步权重 |

**Recipe 层代码** (`recipe/remote_megatron_sglang/`，原 `recipe/external_sglang/`)：

| 文件 | 组件 | 职责 |
|------|------|------|
| `checkpoint_engine.py` | ExternalSGLangNCCLEngine + ExternalSGLangCheckpointManager | NCCL broadcast 权重同步 |
| `train_agentic_ack_external.sh` | 启动脚本 | Mooncake 路径 |
| `train_agentic_ack_external_nccl.sh` | 启动脚本 | NCCL 路径 |
| `manager_test/` | 测试目录 | 验证和 benchmark 脚本 |

### 1.3 两条权重同步路径

| 特性 | Mooncake 路径 (verl核心) | NCCL 路径 (recipe) |
|------|--------------------------|---------------------|
| 外部 SGLang 是否需加入 Ray | 是（需要 sglang_node 资源） | 否 |
| 中间 Receiver 进程 | 需要 ReceiverCE（colocated） | 不需要 |
| 传输协议 | Mooncake RDMA P2P daisy-chain | NCCL broadcast |
| 是否需 SGLang 端修改 | 否 | 否 |
| 性能 (27B, TP=8) | ~14.7s | ~2.2s |
| 带宽 | - | 55.7 GiB/s |
| 验证状态 | 已验证端到端 | 已验证 standalone，未完整训练 |

### 1.4 Benchmark 数据

来自 `recipe/remote_megatron_sglang/test/external_sglang/weight_sync_benchmark_results.md`：

| 方案 | TP=2 | TP=8 | 带宽 |
|------|------|------|------|
| NCCL collective | ~1.5s | ~2.2s | 55.7 GiB/s |
| Mooncake GPU | ~4.0s | ~14.7s | ~12.9 GiB/s |
| Mooncake Host | ~6.6s | ~25.2s | ~7.9 GiB/s |

NCCL collective broadcast 比 P2P daisy chain 快约 6.5x，且随 TP 规模增大带宽反而提升。

### 1.5 数据流

```
GENERATION 平面:
  agent/proxy → LB.acquire_server() → ExternalSGLangProxyActor.generate.remote()
                                    → HTTP /generate on external SGLang

WEIGHT SYNC 平面 (Mooncake):
  trainer FSDP rank0 --Mooncake RDMA--> ReceiverCE (colocated on SGLang GPU)
  --CUDA IPC--> external SGLang

WEIGHT SYNC 平面 (NCCL):
  trainer FSDP rank0 --NCCL broadcast--> external SGLang TP workers
  (HTTP /init_weights_update_group 建组, /update_weights_from_distributed 触发)
```

### 1.6 上游状态

**尚未进入 upstream。** `recipe/remote_megatron_sglang/`（原 `recipe/external_sglang/`）在上游 origin/main 不存在，完全在本地分支开发。上游化进度为零。

### 1.7 当前问题与挑战

#### 1.7.1 权重同步性能瓶颈

- **Mooncake 路径性能较差**：P2P daisy-chain 架构导致权重同步延迟随 TP 规模线性增长（TP=8 时 ~14.7s），远慢于 NCCL collective（~2.2s）
- **NCCL 路径尚未完成端到端验证**：standalone benchmark 已验证，但尚未在完整训练流程中跑通
- **跨节点场景**：训练集群和推理集群跨节点时，NCCL 建组和 Mooncake RDMA 的网络配置复杂度高

#### 1.7.2 SGLang 集成复杂度

- **Mooncake 路径要求 SGLang 加入 Ray 集群**：需要 sglang_node 资源和 ReceiverCE colocated 进程，增加了部署复杂度
- **flush_cache 死锁风险**：SGLang weight update 默认 flush_cache=True 会导致 CPU 死锁，必须显式禁用
- **SGLang 版本依赖**：工具调用（tool call）需要显式配置 --tool-call-parser，Qwen3 系列需要 qwen3_coder parser

#### 1.7.3 训练-推理一致性

- **prompt 超长问题**：rollout.prompt_length 必须小于模型 max_model_len，多轮 agent loop 中 prompt_ids 超限会导致级联崩溃
- **权重同步后 generation 质量问题**：Mooncake 权重同步数据正确但 generation 仍有问题（乱码），根因涉及 magic 信号缓冲区污染

#### 1.7.4 工程与运维

- **两条权重同步路径并存**：Mooncake 和 NCCL 两条路径增加维护成本，需收敛为一条主路径
- **K8s 部署复杂**：外部 SGLang 的 K8s 部署涉及节点亲和、sandbox pod 清理、Python 模块热更新等多重运维挑战
- **缺乏端到端自动化测试**：当前测试以 standalone 验证为主，缺少完整的端到端训练+推理回归测试

#### 1.7.5 上游化障碍

- **无统领性 RFC**：上游没有专门的 issue 追踪 external SGLang 端到端方案
- **与上游架构方向可能冲突**：上游正推进 Delta Weight Sync (#6974) 和 Colocated CE (#6225)，上游化时需评估是否需要基于上游新方案重新适配

---

## 二、社区方案

### 2.1 RemoteBackend RFC (#6537, PR #6422)

**提出者**：Snowflake AI Research (Karthik Ganesan)

**核心思路**：定义一个含 10 个方法的抽象基类（ABC），让任意进程外的 RL 后端以"一个文件"方式接入 verl，无需 fork。verl 只负责编排（PPO/GRPO 算法、数据管线、Metric 计算），训练和推理引擎全部外置。

**10 个抽象方法**：

| 类别 | 方法 | 说明 |
|------|------|------|
| 生命周期 | `from_config(config, handle=None)` | 构造器，handle=None 新建，handle=dict 重连 |
| 生命周期 | `reconnect_handle() → dict` | 可序列化的重连信息 |
| 生命周期 | `destroy()` | 幂等销毁 |
| RL 核心 | `compute_log_prob(data, ref, ...)` | 前向推理（actor 或 ref model） |
| RL 核心 | `update_actor(data, ...)` | 前向+反向+优化器步进 |
| RL 核心 | `generate(prompt_ids, params)` | 采样 rollout |
| RL 核心 | `update_weights()` | 训练→推理权重同步 |
| RL 核心 | `save_checkpoint()` | 持久化 |
| 并行性 | `requires_single_forwarder()` | 是否要求单 forwarder |

**设计原则**：
- verl 不介入通信协议（支持 Ray/HTTP/gRPC 自选）
- verl 不介入 loss 计算逻辑
- 输入 TensorDict，输出 dict
- 零侵入：trainer.remote_backend 默认 null，legacy 路径不变

**配套组件**：
- RemoteBackendRegistry — 名称到类的注册表，延迟导入
- RemoteBackendActorRolloutRefWorker — CPU-only forwarder (~230 LOC)
- RemoteBackendTrainer — RayPPOTrainer 子类

**当前状态**：Draft PR，维护者反馈积极（"Generally looks good"），要求移植到 V1 trainer。CLA 仅 1/5 提交者签署。

### 2.2 Snowflake Arctic RL（唯一参考实现）

**技术栈**：

| 组件 | 选型 |
|------|------|
| 训练引擎 | DeepSpeed ZeRO（全参数训练） |
| 推理引擎 | vLLM + ArcticInference |
| 参考模型 | Forward-only DeepSpeed |
| 权重同步 | NCCL 或 CUDA-IPC |
| 通信协议 | Ray actor handle 或 HTTP |
| 核心优化 | ZoRRo (Split Attention + Forest Cascade Attention) |

**DeepSpeed 对接方式（三层代理架构）**：

```
verl Trainer (CPU)
    │ await backend.update_actor(data)
    ▼
ArcticRLClientWrapper (~650 LOC) ← 适配层，CPU-only
    │ Ray actor handle
    ▼
ArcticRLClient (Ray actor, CPU-only 代理)
    │ 内部管理
    ▼
DeepSpeed ZeRO Engine (GPU, 零修改)
```

**训练引擎暴露的 wire 操作**：

| Wire 调用 | DeepSpeed 操作 |
|-----------|---------------|
| `fwd_no_grad(ref=True)` | ref_engine.forward() 无梯度 |
| `fwd_no_grad(ref=False)` | actor_engine.forward() 无梯度 |
| `fwd_bwd(payload)` | engine.forward() + engine.backward(loss) |
| `step()` | engine.step() (optimizer + lr scheduler) |

**关键设计 — Single Forwarder**：verl 侧只启动 1 个 CPU-only forwarder，把完整 global batch 发给 Arctic，Arctic 服务端自行处理 ZeRO 分片、micro-batch 切分、gradient accumulation。

**性能数据**：
- Arctic-Text2SQL-R2, 32x H200: 训练从 ~5天降至 ~36小时 (3.5x 加速)
- BIRD dev 准确率: 59.92% → 70.35% (+10.43)
- ZoRRo 训练侧: 高达 6x actor-update 加速

### 2.3 其他上游相关进展

| ID | 标题 | 状态 | 说明 |
|----|------|------|------|
| #6117 | SGLang PD Disaggregated Rollout | MERGED | 1P:ND 非对称 PD 分离 |
| #6974 | Delta Weight Sync + Sharded | OPEN | 增量传输，降低 100-1000x |
| #6225 | Colocated Checkpoint Engine | OPEN | Qwen 团队，CUDA IPC |
| #4003 | Use CE to accelerate weight sync | OPEN | 最早 RFC |
| #6373 | MooncakeStoreConnector | MERGED | KV cache hard-reset |
| #6266 | NCCL Suspend/Resume | OPEN | 释放 communicator 内存 |
| #5400 | TransferQueue Integration | OPEN | TQ 数据面 |
| #4784 | SGLang remote_instance | DRAFT | 远端实例支持 |

### 2.4 26Q3 Roadmap 重点

- Delta Weight Sync — 最高优先级
- DeviceMesh P2P remapping
- NCCL suspend/resume
- Mooncake 分布式 KV cache store 集成
- Dynamic trainer↔rollout switch in fully async

---

## 三、方案对比

### 3.1 架构定位对比

| 维度 | 我们的 External SGLang | 社区 RemoteBackend |
|------|----------------------|-------------------|
| **抽象层级** | 外部推理引擎 + 权重同步 | 外部完整 RL 后端（训练+推理） |
| **训练引擎** | verl 内部 FSDP | 外部系统自有（DeepSpeed/Megatron/任意） |
| **推理引擎** | 外部 SGLang | 外部系统自有（vLLM/SGLang/任意） |
| **权重同步** | verl 的 CheckpointEngine 管理 | 后端自行管理（verl 黑盒） |
| **verl 改动** | ServerManager + CheckpointManager 插件 | RemoteBackend ABC + Trainer 子类 |

### 3.2 适用场景对比

| 场景 | External SGLang | RemoteBackend |
|------|----------------|---------------|
| 只有推理在外部，训练在 verl | ✅ 完美匹配 | ❌ 抽象不自然（compute_log_prob/update_actor 需回调 verl） |
| 训练+推理都在外部 | ❌ 不适用 | ✅ 完美匹配 |
| 使用 verl 的 FSDP/Megatron 训练 | ✅ 原生支持 | ❌ 需额外桥接 |
| 使用外部训练引擎 | ❌ 不适用 | ✅ 原生支持 |
| 快速接入新推理引擎 | ✅ ServerManager 插件 | ⚠️ 需实现完整 ABC |
| 快速接入新完整 RL 系统 | ❌ 需大量适配 | ✅ 一个文件 |

### 3.3 权重同步方案对比

| 维度 | External SGLang | Arctic RL (RemoteBackend) |
|------|----------------|--------------------------|
| 传输协议 | NCCL-over-HTTP / Mooncake RDMA | NCCL / CUDA-IPC |
| verl 可见性 | CheckpointEngine 管理，verl 可见 | 后端黑盒，verl 只调 update_weights() |
| SGLang 加入 Ray | Mooncake 路径需要，NCCL 不需要 | 不需要 |
| 性能 (27B TP=8) | NCCL ~2.2s, Mooncake ~14.7s | 未公开具体数据 |
| Benchmark | 有完整 benchmark 脚本和结果文档 | 仅有 smoke test |

### 3.4 成熟度对比

| 维度 | External SGLang | RemoteBackend |
|------|----------------|---------------|
| 开发阶段 | 开发中，核心功能已验证 | Draft PR，一个参考实现 |
| 代码完整度 | 中高（两条权重同步路径、K8s 部署、benchmark） | 中（~650 LOC 适配器 + Arctic Platform） |
| 测试覆盖 | manager_test + standalone 验证 | E2E smoke test (4-step) |
| 文档 | README + 部署文档 + benchmark 结果 | RFC Issue + 工程博客 |
| 上游状态 | 未提交 | Draft PR，维护者反馈积极 |
| 生产验证 | 内部验证中 | Snowflake 生产环境 (32x H200) |

### 3.5 训练与推理引擎接口契约

两种方案对引擎的要求不同：

**训练引擎需要的接口**：

| 操作 | External SGLang 中谁提供 | RemoteBackend 中谁提供 |
|------|------------------------|----------------------|
| forward (no grad) | verl 内部 FSDP | 后端自有引擎 |
| forward + backward | verl 内部 FSDP | 后端自有引擎 |
| optimizer step | verl 内部 FSDP | 后端自有引擎 |
| 提取权重 | CheckpointEngine 提取 | 后端自行提取 |

**推理引擎需要的接口**：

| 操作 | External SGLang | RemoteBackend |
|------|----------------|---------------|
| generate | 外部 SGLang HTTP API | 后端自有推理引擎 |
| 加载权重 | SGLang /update_weights API | 后端自行实现 |
| KV cache 管理 | flush_cache=True (注意死锁风险) | 后端自行管理 |
| 请求中断 | abort pending requests | 后端自行管理 |

### 3.6 结论与建议

1. **短期**：以 External SGLang 为主要开发方向，持续完善权重同步路径、端到端训练验证和测试覆盖。当前的 CheckpointEngine + ServerManager 插件模式与「外部推理引擎+权重同步」的定位完全匹配，无需迁移到 RemoteBackend。

2. **中期关注**：
   - 上游 Delta Weight Sync (#6974) 成熟后，评估是否可复用增量传输优化
   - RemoteBackend RFC 稳定后，评估是否将 external SGLang 的 generate + update_weights 适配为 RemoteBackend 的部分方法（混合模式）

3. **长期**：如果 external SGLang 演进为完整的外部 RL 后端（训练也在外部集群进行），则 RemoteBackend 适配将变得自然且必要。

4. **上游化建议**：向上游提交时，以 CheckpointEngine 插件 + ServerManager 插件的方式提交（而非 RemoteBackend），这与现有架构更匹配，也更容易被接受。
