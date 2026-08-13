# Weight-sync timing benchmark — results

Measured weight-sync **transport** time for the different ways verl syncs weights
to SGLang. Driver: [`benchmark_weight_sync.py`](./benchmark_weight_sync.py)
(same stand-in-trainer → colocated-receiver → CUDA-IPC-push pipeline as
`ExternalCheckpointManager.update_weights()`).

**Setup (all rows):** Qwen3.6-27B, 51.75 GiB bf16 (1199 tensors), TP=2, ONE
external SGLang (2 receivers), Mooncake bucket 3 GiB, `--mem-fraction-static 0.55`,
Wulan ACK cluster (trainer node 602 → sglang pod node 601). 3–4 measured rounds
per run, warm-up round excluded. `transport` = sender-reported pure transfer;
`full sync` = driver wall-clock of send + receive_and_push (transport + host→GPU
stage + CUDA-IPC push). Date: 2026-07-07.

## 1. Three ways compared

| way | landing buffer | transport (s) | bandwidth (GiB/s) | full sync (s) | status |
|---|---|---|---|---|---|
| **mooncake disaggregated** | GPU (GPUDirect RDMA) | **~4.0** | **~12.9** | ~4.0 | ✅ measured |
| **external mooncake** | pinned host + H2D stage | **~6.6** | **~7.9** | ~6.6 | ✅ measured |
| **nccl disaggregated** | GPU (collective broadcast) | — | — | — | ⚠️ fabric now works (§5); benchmark harness path still blocked |

Notes:
- mooncake disaggregated and external mooncake share the **same** RDMA P2P daisy
  chain (rank-0 sender → chain of receivers, each gets the full weight set, then
  CUDA-IPC push). The only difference is **where the RDMA buffer lands** (GPU vs
  host) + the host path's per-bucket host→GPU staging copy.
- CUDA-IPC push is fully hidden (full sync ≈ transport).
- nccl: see §5 — the RoCEv2/IPv6 fabric now works cross-node (a 2-rank probe
  passes), but the multi-rank weight-sync benchmark path is still blocked.

## 2. mooncake landing buffer: GPU vs host (+ `chunk_tensors` sensitivity)

| landing buffer | `chunk_tensors` | transport mean (s) | range (s) | bandwidth (GiB/s) | vs GPU |
|---|---|---|---|---|---|
| GPU (GPUDirect) | 16 | **4.0** | 3.86–4.14 | ~12.9 | 1.0× |
| host + H2D stage | 16 | **6.6** | 5.90–7.35 | ~7.9 | ~1.6× |
| host + H2D stage | 64 | **5.11** | 4.57–5.60 | 10.1 | ~1.28× |

- Host memory frees ~6 GB of the colocated inference GPU (→ high mem-fraction) at
  a ~1.3–1.6× transport cost.
- The per-bucket host→GPU copy is on the critical path, so it dominates the gap
  and adds jitter. Raising `chunk_tensors` 16→64 (fewer/larger H2D copies, better
  overlap with RDMA receive) recovers most of the penalty — near-free when the GPU
  has headroom for the larger transient staging.

## 2b. TP scaling — TP=2 vs TP=8 (chain depth 2 → 8)

Same model/setup, re-run with the external SGLang at **TP=8** (8 receivers, one
8-GPU pod on node 601; sender on node 602). Mooncake uses a **P2P daisy chain**
(rank-0 sender → rank1 → … → rankN), so the receiver count = chain length.

| variant | TP=2 transport (GiB/s) | TP=8 transport (GiB/s) | slowdown (4× receivers) |
|---|---|---|---|
| mooncake-gpu (GPU buffer, chunk 16) | 4.0 s (12.9) | **14.66 s (3.53)** | ~3.7× |
| host buffer, chunk 16 | 6.6 s (7.9) | **25.18 s (2.06)** | ~3.8× |
| host buffer, chunk 64 | 5.11 s (10.1) | **22.00 s (2.35)** | ~4.3× |

Findings:
- **The Mooncake P2P daisy chain scales ~linearly with receiver count** — 2→8
  receivers (4×) costs ~3.7–4.3× more time, for *both* GPU and host buffers. The
  chain fills/drains serially, so weight-sync latency grows with TP. This is the
  dominant cost at TP=8, not the buffer choice.
- **Host stays ~1.5–1.7× above GPU** at both TP=2 and TP=8 — the host-staging
  penalty is roughly a constant multiplier, independent of chain depth.
- **`chunk_tensors` helps less at scale**: 16→64 saved ~23% at TP=2 but only ~13%
  at TP=8, because the chain-depth cost dwarfs per-hop staging there.
- Implication: for large-TP inference engines, the P2P chain is the bottleneck. A
  **collective broadcast (NCCL)** delivers to all receivers concurrently and would
  scale far better — but it doesn't establish cross-node on this fabric (§1), which
  is the trade-off this deployment accepts by using Mooncake.

## 3. Raw per-round data (transport seconds)

| variant | run | rounds (measured) | mean |
|---|---|---|---|
| mooncake-gpu (chunk 16) | 1 | 3.87, 3.86, 3.85 | 3.86 |
| mooncake-gpu (chunk 16) | 2 | 3.98, 4.56, 3.87 | 4.14 |
| host (chunk 16) | 1 | 5.95, 8.97, 6.92 | 7.28 |
| host (chunk 16) | 2 | (summary) | 5.90 |
| host (chunk 16) | 3 | 7.11, 6.02, 6.20, 7.35 | 6.67 |
| host (chunk 64) | 1 | 4.57, 5.27, 5.60, 5.00 | 5.11 |
| **TP=8** mooncake-gpu (chunk 16) | 1 | 14.75, 14.53, 14.71 | 14.66 |
| **TP=8** host (chunk 16) | 1 | 25.71, 25.26, 24.57 | 25.18 |
| **TP=8** host (chunk 64) | 1 | 23.37, 21.41, 21.21 | 22.00 |

## 4. Caveat — stock path needs staging code

In the microbenchmark, "mooncake disaggregated" and "external mooncake" run the
same receiver (`ReceiverCE`), which stages host→GPU. The **stock** path
(`CheckpointEngineWorker.update_weights`, `verl/checkpoint_engine/base.py`) passes
`checkpoint_engine.receive_weights()` **straight to** the CUDA-IPC push, so
`mooncake.device=cpu` alone would hand host tensors to `update_weights_from_tensor`
and fail. Using host memory in stock mooncake disaggregated requires porting
`ReceiverCE.receive_and_push`'s host→GPU staging into the stock worker.

## 5. NCCL backend availability (2026-07-07)

Making the `nccl` checkpoint-engine backend usable with the external SGLang pod:

- **Root cause of the original failure:** the sglang pod was missing the RoCEv2/IPv6
  NCCL env the training worker has (`NCCL_IB_ADDR_FAMILY=AF_INET6`,
  `NCCL_IB_ADDR_RANGE=2001:db8:80f:e000::/60`, `NCCL_NET_PLUGIN=none`,
  `NCCL_SOCKET_IFNAME=eth0`). My earlier `NCCL_IB_GID_INDEX=7` also **forced the
  wrong GID** — NCCL auto-selects a *different* GID per node (trainer=7, sglang
  pod=11); a forced index breaks one side.
- **Fix applied:** added the correct NCCL env to the pod spec
  (`k8s/sglang-external-8gpu.yaml`) — do NOT pin `NCCL_IB_GID_INDEX`.
- **Proven working:** a minimal 2-rank cross-node NCCL all-reduce (rank0 on the
  trainer node 602, rank1 colocated on the sglang pod 601) returns the correct
  result → `NCCL CROSS-NODE OK`. Driver: [`nccl_probe.py`](./nccl_probe.py).
  So NCCL **is now available** at the fabric/pod level.
- **Still open (harness):** the full 9-rank weight-sync benchmark path
  (`NcclSender` + 8 `NcclReceiverCE`) hangs in the `ray.util.collective` imperative
  rendezvous *before* any NCCL comm init (0 `NCCL INFO` lines) — a coordination
  issue the 2-rank probe doesn't hit. Tried: unique group name (not "default",
  which had 9 stale `NCCLUniqueIDStore` actors), gloo device-mesh (to avoid a
  second in-process NCCL context), GID auto-select — none cleared it. This is a
  stand-in-harness quirk, not a fabric limit; stock verl's `backend=nccl` uses the
  WorkerGroup collective path (gloo global group + sglang-owned device mesh), which
  should work now that the pod has the RoCE env.

## 6. Verl-managed disaggregated — nccl vs mooncake (stock path, 2026-07-07)

The §1–§5 numbers used a stand-in harness against an *external* SGLang; the NCCL
transport there couldn't be measured (9-rank rendezvous hang). This section uses
the **real** stock path instead: removed the external pod, scaled up a
`rollout-workers` Ray node, and ran `verl.experimental.one_step_off_policy.main_ppo`
so **verl launches its own SGLang rollout** and syncs weights through the stock
`CheckpointEngineManager`. Driver: [`train_gsm8k_managed.sh`](./train_gsm8k_managed.sh).

**Setup:** Qwen2.5-3B-Instruct (5.75 GiB bf16), gsm8k GRPO, trainer FSDP 8 GPU
(node 602) + verl-managed SGLang rollout 8 GPU (node 601) = **4 replicas × TP2 = 8
rollout CE receivers**, mooncake/nccl world_size = 9 (1 trainer rank0 + 8), bucket
2048 MB. Metric = trainer `timing_s/update_weights` = the **full**
`CheckpointEngineManager.update_weights()` (abort → release_kv → build group →
transfer → resume), steady-state (warmup steps excluded).

| backend | full weight sync (update_weights) | engine transport self-report |
|---|---|---|
| **nccl** (collective broadcast) | **~1.35 s** (1.19–1.57) | 9-rank NCCL group init COMPLETE; collective |
| **mooncake** (P2P chain, GPU) | **~2.4 s** (2.19–2.69, spikes 3.4) | send 0.65 s @ 8.8 GB/s, recv ~1.35 s @ 4.5 GB/s |

**NCCL is ~1.8× faster than mooncake** for the full sync in this verl-managed
disaggregated setup. Notes:
- NCCL `backend=nccl` **now works cross-node** end-to-end via the stock WorkerGroup
  path (9-rank group `Init COMPLETE`), confirming the pod NCCL-env fix (§5). The
  earlier failure was purely the stand-in harness's `ray.util.collective` +
  in-process device-mesh combination, not the fabric.
- Mooncake's raw transport is fast (send 0.65 s), but the full per-step sync (~2.4 s)
  carries the P2P-chain build/handshake + orchestration each step; NCCL's persistent
  collective group amortizes that.
- This is a **different metric and model** than §1–§2 (full stock sync + 3B here vs
  transport-only + 27B there), so compare the nccl-vs-mooncake *ratio*, not absolute
  seconds, across sections.

### Raw per-step `update_weights` (seconds)

NCCL (`gsm8k_nccl`):

| step | 1 | 2 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s | 0.91 | 1.75 | 1.30 | 1.34 | 1.34 | 1.37 | 1.47 | 1.57 | 1.19 | 1.31 | 1.32 | 1.30 | 1.20 | 1.46 | 1.35 |

- steady-state (steps 17–29): mean **1.35 s**, min 1.19, max 1.57 (steps 1–2 are warm-up: group build / first sync).
- NCCL transport (per rank, from engine log at group init): 9-rank group (`nranks 9`, 1 trainer + 8 rollout CE), NCCL 2.28.9, `Init COMPLETE` cross-node (602↔601).

Mooncake (`gsm8k_mooncake`):

| step | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s | 2.52 | 2.34 | 2.34 | 2.39 | 3.47 | 3.26 | 2.55 | 2.37 | 2.69 | 2.19 | 2.24 | 2.29 |

- steady-state: mean **~2.5 s** (excl. the 3.4 s spikes), min 2.19, max 3.47.
- engine transport self-report: `send weights done total bytes 6171877376 (5.75 GiB) time cost 0.65 s @ 8.83 GB/s`; `receive weights done ~1.3–1.4 s @ 4.5–4.8 GB/s` (×8 receivers).

## 7. NCCL weight sync to an EXTERNAL SGLang — NCCL-over-HTTP (2026-07-08)

Answering "can we NCCL-sync to an out-of-Ray SGLang?" — **yes**, via SGLang's
native decoupled API (NOT `backend=nccl` + `ReceiverCE`). Trainer rank 0 forms a
NCCL group directly with the external SGLang's TP workers and broadcasts full HF
weights; SGLang's `update_weights_from_distributed` re-shards per TP. No colocated
receiver, no CUDA IPC, SGLang not in Ray. Driver:
[`validate_weight_sync_nccl_http.py`](./validate_weight_sync_nccl_http.py); engine:
`recipe/remote_megatron_sglang/checkpoint_engine.py` (`backend=external_sglang_nccl`).

**Setup:** Qwen2.5-3B (5.75 GiB, 434 tensors), external SGLang TP=2 (out of Ray,
on node 601), trainer rank-0 actor on node 602, NCCL group world_size=3.

| phase | group build | broadcast (5.75 GiB) | result |
|---|---|---|---|
| sync base | 0.011 s | 1.08 s (cold) | generation unchanged ✓ |
| sync zero-embed | 0.011 s | 0.16 s (warm) | generation → `!!!!` ✓ |
| sync restore | 0.023 s | 0.16 s (warm) | generation recovered ✓ |

- The **key fix** was replacing `torch.distributed.new_group` with SGLang's
  `init_custom_process_group` (a fresh NCCL group over a TCP store that can span
  the out-of-process SGLang workers) — the scaffolding's own TODO.
- Cross-node RoCE NCCL is what makes this work (same fabric fix as §5).
- This is the recommended external-NCCL path; `backend=nccl` + `ReceiverCE`
  (colocated, `ray.util.collective` + CUDA IPC) is the harder one that hung.

### Timing (BENCH=1, 5 rounds after warmup)

Per-sync breakdown (each round rebuilds+destroys the group, matching the engine's
`finalize()`):

| component | mean | note |
|---|---|---|
| **total / sync** | **1.057 s** (1.04–1.09) | group rebuild+destroy every step |
| group build | 0.011 s | `init_custom_process_group` + SGLang join |
| **broadcast (transport+load)** | **0.160 s @ 35.9 GiB/s** | actual weight movement |
| flush_cache | 0.135 s | SGLang cache flush |
| **destroy (group teardown)** | **0.746 s** | NCCL group destroy — dominant (70%) |

- The **NCCL transport itself is very fast** (0.16 s @ ~36 GiB/s over the bonded
  RoCE NICs). The per-step cost is dominated by **NCCL group teardown (0.75 s)**.
- **Persistent group ⇒ ~0.30 s/sync** (broadcast 0.16 + flush 0.135): keep the
  group across steps instead of rebuild+destroy each step (`ExternalSGLangNCCLEngine.finalize`
  currently destroys — make it reuse for a big win).
- vs §6 verl-managed NCCL (~1.35 s): that was 8 receivers (world_size 9) + full
  `CheckpointEngineManager` orchestration; this is 2 receivers (world_size 3) with
  only flush_cache, but pays the teardown.

### 27B TP=8 (Qwen3.6-27B, 51.75 GiB, external TP=8, world_size=9)

| component | mean | note |
|---|---|---|
| **total / sync** | **2.238 s** (2.12–2.70) | group rebuild+destroy each step |
| group build | 0.011 s | |
| **broadcast (transport+load)** | **0.928 s @ 55.7 GiB/s** | collective to all 8 TP at once |
| flush_cache | 0.598 s | |
| destroy (teardown) | 0.696 s | |

**Headline — NCCL collective vs mooncake P2P chain at 27B/TP=8 (external):**

| method | full sync | transport |
|---|---|---|
| mooncake **host** chain (§2b) | ~25.2 s | ~2 GiB/s |
| mooncake **GPU** chain (§2b) | ~14.7 s | 3.5 GiB/s |
| **NCCL-over-HTTP** | **~2.2 s** | **0.93 s @ 55.7 GiB/s** |

- **NCCL is ~6.5× faster than the GPU mooncake chain, ~11× faster than the host
  chain.** The collective broadcast hits all 8 TP workers at once, so its transport
  bandwidth *rose* with scale (55.7 GiB/s at TP=8 vs 35.9 at TP=2 — more bonded NICs
  participating), whereas the P2P daisy chain degrades ~linearly with receiver count
  (§2b). This is the decisive argument for NCCL when the fabric supports it.
- Persistent group would drop the 2.24 s to ~1.5 s (bcast 0.93 + flush 0.60).

## How to reproduce

```bash
# from the Ray head pod, in the verl repo root
VARIANTS=external-mooncake,mooncake-gpu CHUNK_TENSORS=16 ROUNDS=4 \
  python3 benchmark_weight_sync.py           # host vs GPU landing buffer
VARIANTS=external-mooncake CHUNK_TENSORS=64 ROUNDS=5 \
  python3 benchmark_weight_sync.py           # host, larger staging batch
# nccl also selectable via VARIANTS=nccl (cross-node rendezvous currently fails here)

# TP=8 (deploy k8s/sglang-external-8gpu.yaml, ray start on the pod, sglang --tp 8):
SGLANG_TP=8 VARIANTS=external-mooncake,mooncake-gpu CHUNK_TENSORS=16 ROUNDS=4 \
  python3 benchmark_weight_sync.py

# §6 verl-managed (stock path): remove external pod, scale rollout-workers=1, then
BACKEND=nccl     ROLLOUT_TP=2 ./train_gsm8k_managed.sh   # read timing_s/update_weights
BACKEND=mooncake ROLLOUT_TP=2 ./train_gsm8k_managed.sh
```
