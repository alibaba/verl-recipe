# Root Cause Analysis: Mooncake `transfer_sync_write` Corrupts Local GPU Memory

## Summary

Mooncake's `transfer_sync_write` on intra-node RDMA has a side effect: when writing to a remote GPU's memory address, it also modifies unrelated memory on the calling GPU. This causes verl's `MooncakeCheckpointEngine` to corrupt model weights during the daisy-chain weight sync, producing degenerate inference output (`!!!!!!` repeated to max_response_length).

## Bug Impact

- **Affected component:** `verl/checkpoint_engine/mooncake_checkpoint_engine.py`
- **Affected models:** Any model using Mooncake backend with multi-rank rollout on the same node
- **Symptom:** Model outputs a single token repeated (e.g. `!`) after weight sync. `response_length` maxes out at 4096.
- **Severity:** Critical — training produces no useful gradients
- **NCCL backend:** Not affected (does not use Mooncake TransferEngine)
- **Upstream status:** Bug present in verl `main` as of commit `e4a4c189`. Not fixed.

## Root Cause

### The completion signaling mechanism

`MooncakeCheckpointEngine` uses a double-buffered RDMA pipeline with a daisy-chain topology (trainer rank 0 → rollout ranks 1–8). After a receiver finishes reading a bucket, it writes a 4-byte magic marker (`[0xAB, 0xDC, 0xEF, 0x88]`) to the sender's data buffer as a completion signal:

```python
# receive_weights: write magic to previous rank's DATA buffer
ret = self.engine.transfer_sync_write(
    self.buffer_info["session_id"],  # remote session (previous rank)
    self.magic_buf.data_ptr(),        # local source (magic bytes)
    ptr,                               # remote destination (previous rank's data buffer)
    4,
)
```

### The Mooncake side effect

Instrumentation with CUDA synchronize fences and address logging proved that `transfer_sync_write` to a remote GPU address **also modifies unrelated memory on the calling GPU**:

```
[R4 B1] D-post-yield  buf[:4]=[43, 187, 187, 188]    ok
         local_buf=0x7f0e80000000  gpu=cuda:3

[R4 B1] E-post-magic  buf[:4]=[171, 220, 239, 136]    LOCAL-HIT!
         changed_by_own_write=True
         wrote_to=0x7fc760000000@10.8.0.75:16699       (R3's buffer on cuda:2)
         from=0x7f0f93400000                            (R4's magic_buf)
```

- R4 wrote magic to R3's buffer at `0x7fc760000000` (remote GPU, cuda:2)
- R4's own data buffer at `0x7f0e80000000` (local GPU, cuda:3) was also modified
- These are **completely different addresses on different GPUs**
- `changed_by_own_write=True`: CHECK-D was clean (after CUDA sync), CHECK-E had magic, and the ONLY operation between them was `transfer_sync_write`
- This is **deterministic**, not a race condition

### The cascading corruption

The corrupted buffer propagates through the daisy chain:

```
R4's transfer_sync_write → corrupts R4's own local buffer
    ↓
R5 RDMA-reads from R4's corrupted buffer → R5 also gets magic bytes
    ↓
R5's yields expose corrupted data to the consumer
```

Instrumentation confirmed:
```
[R5 B1] D-post-yield  buf[:4]=[171, 220, 239, 136]  MAGIC!
         local_buf=0x7fcaa0000000  remote_ptr=0x7f0e80000000   ← R4's buffer (corrupted)
```

### Which tensor is corrupted

The first tensor at offset 0 of the affected bucket. In Qwen3.6-27B, this is `model.language_model.embed_tokens.weight` (tied with `lm_head.weight`). Only the first 4 bytes (2 bf16 elements) are overwritten:

| Element | Original | Corrupted |
|---------|----------|-----------|
| `embed_tokens[0]` | -0.0026 | **-3.85e+17** |
| `embed_tokens[1]` | -0.0228 | **-1.44e-33** |
| `embed_tokens[2:]` | correct | correct |

The corrupted values are `[0xAB, 0xDC, 0xEF, 0x88]` interpreted as two bf16 floats. The astronomical value in the embedding layer saturates all downstream attention computations, causing the model to output a single token repeated.

## Verification Steps

### Step 1: SHA256 tensor comparison

Added a diagnostic to `ServerAdapter.update_weights()` comparing each tensor against the original safetensors on disk:

- **214 parameters matched** (byte-identical SHA256)
- **2 parameters mismatched**: `embed_tokens.weight` and `lm_head.weight`
- Only the first 4 bytes differed — exactly the magic marker

### Step 2: Instrumented `receive_weights`

Added 5 CHECK points (A through E) with CUDA synchronize fences around each:

| Check | When | R4 B1 result |
|-------|------|-------------|
| A | Before RDMA read | `[0,0,0,0]` ok |
| B | After RDMA read | `[43,187,187,188]` ok |
| C | Before yield | `[43,187,187,188]` ok |
| D | After all yields (CUDA synced) | `[43,187,187,188]` **ok** |
| E | After `transfer_sync_write` (CUDA synced) | `[171,220,239,136]` **LOCAL-HIT!** |

`changed_by_own_write=True` — the ONLY operation between D and E was `transfer_sync_write`. The write targeted `0x7fc760000000` (R3's GPU) but modified `0x7f0e80000000` (R4's GPU).

### Step 3: RDMA-only experiment

Ran weight sync with `sgl_update_weights` disabled (RDMA transfer completes but SGLang doesn't load weights). Inference was correct — proving TransferEngine init and RDMA transfer alone don't corrupt SGLang.

### Step 4: enforce_eager experiment

Ran with `--disable-cuda-graph`. Inference still produced `!!!!!!` — proving CUDA graphs are not the cause of the garbage output (they cause a separate scheduler deadlock issue).

## Fix

### verl fix (applied and verified)

Introduced a separate RDMA-registered buffer (`magic_recv`, 8 bytes) for completion signals. Magic is written to `magic_ptr` (dedicated slot) instead of `ptr` (data buffer):

```python
# __init__
self.magic_recv = torch.zeros(8, dtype=torch.uint8, device=self.device)
self.engine.batch_register_memory(
    [self.buf.data_ptr(), self.magic_buf.data_ptr(), self.magic_recv.data_ptr()],
    [2 * self.bucket_size, 4 * 1024, 8],
)

# send_weights: include magic_ptr in info
info["magic_ptr"] = magic_slots[idx].data_ptr()

# receive_weights: write magic to magic_ptr, not ptr
self.engine.transfer_sync_write(session, self.magic_buf.data_ptr(), magic_ptr, 4)
```

Even if `transfer_sync_write` has the same local side effect, it modifies `magic_recv` (which only holds magic bytes) instead of the data buffer — harmless.

**Verification:** With the fix, full training completed successfully:
- `response_length/mean`: 506.25 (vs 4096 with bug)
- `num_turns/mean`: 4.0
- Agent trajectories: coherent English text analyzing Django issues
- Checkpoint saved, no crashes

### Mooncake issue to report

`transfer_sync_write` modifies memory on the calling GPU when writing to a remote GPU on the same node. This violates standard RDMA WRITE semantics where only the specified remote address should be modified. The affected addresses are unrelated (different virtual addresses on different GPUs), suggesting a bug in Mooncake's intra-node RDMA path.

## Reproduction

Run `docs/reproduce_magic_overwrite.py` (requires only `torch`, no Mooncake):

```
python reproduce_magic_overwrite.py
```

This replays the memory operations and shows:
- Case 1 (Sender): SAFE — magic is transient
- Case 2 (Daisy chain): BUG CONFIRMED — intermediate rank's data corrupted
- Case 3 (Fix): Data intact

## Timeline

| Date | Event |
|------|-------|
| Jun 13 | First Mooncake weight sync attempt, garbage output observed |
| Jun 14 | NCCL backend verified correct — bug isolated to Mooncake |
| Jun 16-17 | Weight norms verified correct, scheduler deadlock investigated |
| Jun 18 | Hypothesis A (flush_cache timing) disproved — identical flow for NCCL/Mooncake |
| Jun 18 | Hypothesis B (CUDA graphs) partially confirmed for deadlock, disproved for garbage |
| Jun 18 | RDMA-only experiment: inference correct without `sgl_update_weights` |
| Jun 20 | SHA256 comparison: `embed_tokens`/`lm_head` first 4 bytes = magic marker |
| Jun 20 | `magic_recv` fix applied and verified — training completes successfully |
| Jun 22 | Instrumented `receive_weights` proves `transfer_sync_write` local side effect |
| Jun 22 | Address logging proves corruption is deterministic, not a race condition |
