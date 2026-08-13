# verl Framework Bugs — Mooncake Agentic RL Training

This document records six bugs in the verl framework discovered during agentic RL training with Qwen3.6-27B on a disaggregated Mooncake + SGLang setup (Wulan ACK cluster, 2 nodes, TP=8, GRPO).

---

## Bug 1: Mooncake RDMA Completion Magic Overwrites Data Buffer

**Severity:** Critical
**Affected file:** `verl/checkpoint_engine/mooncake_checkpoint_engine.py`
**Upstream status:** Present in `main` as of commit `e4a4c189`. Not fixed.

### Symptom

After Mooncake weight sync, the SGLang inference model outputs degenerate text — a single token (e.g. `!`) repeated to `max_response_length`. The model's attention is completely broken. NCCL weight sync with the same model and pipeline produces correct output.

### When it happens

Every Mooncake weight sync corrupts the first tensor in the first bucket. Because `get_per_tensor_param()` yields parameters in module order, the first parameter is typically `embed_tokens.weight` (and its tied alias `lm_head.weight`). The corruption is deterministic but only observable on specific daisy-chain ranks depending on timing.

### Root cause

`MooncakeCheckpointEngine` uses a double-buffered RDMA pipeline with a daisy-chain topology (trainer rank 0 → rollout ranks 1–8). After a receiver finishes reading data from the sender's buffer, it writes a 4-byte magic marker (`[0xAB, 0xDC, 0xEF, 0x88]`) back to the sender's buffer as a completion signal:

```python
# receive_weights() — writes magic to sender's DATA buffer
ret = self.engine.transfer_sync_write(
    self.buffer_info["session_id"],
    self.magic_buf.data_ptr(),
    ptr,   # ← ptr = info["ptr"] = sender's data buffer start address
    4,
)
```

The sender polls for this magic at `buf[:4]`:

```python
# wait_for_complete() — checks DATA buffer for magic
async def wait_for_complete(self, buf):
    magic = torch.tensor([0xAB, 0xDC, 0xEF, 0x88], dtype=torch.uint8, device=self.device)
    while True:
        if torch.equal(buf[:4], magic):
            break
        await asyncio.sleep(0)
```

In the daisy chain, each rank forwards data to the next rank and updates `info["ptr"]` to point to its own local buffer. The next rank will eventually write magic back to this buffer. This creates a race:

1. Rank N yields tensor views from its buffer (including `embed_tokens` at offset 0).
2. Rank N forwards `info["ptr"] = buffer.data_ptr()` to rank N+1.
3. The consumer calls `.clone()` on each yielded tensor (async CUDA kernel).
4. Rank N+1 finishes its RDMA read and writes magic to rank N's `buffer[:4]`.
5. If the `.clone()` CUDA kernel for `embed_tokens` hasn't read the first 4 bytes before step 4, those bytes are overwritten with `[0xAB, 0xDC, 0xEF, 0x88]`.

The RDMA write in step 4 goes directly to GPU memory via PCIe DMA, bypassing CUDA stream ordering. There is no synchronization between the RDMA completion write and the CUDA `.clone()` kernel.

The corrupted bytes `[0xAB, 0xDC, 0xEF, 0x88]` interpreted as two bf16 values are `(-3.85e+17, -1.44e-33)`. These astronomical values in the embedding layer cause all downstream attention computations to saturate, producing degenerate output.

### Verification

A tensor dump with SHA256 comparison against the original safetensors confirmed:

- 214 parameters: byte-identical match (**[MATCH]**)
- 2 parameters: **[MISMATCH]** — `embed_tokens.weight` and `lm_head.weight`
- Only the first 2 bf16 elements (4 bytes) differ; all subsequent elements are correct
- The corrupted bytes are exactly `[0xAB, 0xDC, 0xEF, 0x88]` — the magic marker

### Fix

Introduce a separate RDMA-registered buffer (`magic_recv`, 8 bytes — one 4-byte slot per double-buffer) for completion signals. Send a `magic_ptr` address in the info dict alongside `ptr`. The receiver writes magic to `magic_ptr` (dedicated signal buffer) instead of `ptr` (data buffer). The sender polls `magic_recv` slots instead of `buf[:4]`.

```python
# __init__: allocate dedicated magic receive buffer
self.magic_recv = torch.zeros(8, dtype=torch.uint8, device=self.device)
# register for RDMA alongside data buffer and magic source buffer
self.engine.batch_register_memory(
    [self.buf.data_ptr(), self.magic_buf.data_ptr(), self.magic_recv.data_ptr()],
    [2 * self.bucket_size, 4 * 1024, 8],
)

# send_weights: include magic_ptr in info
info = {
    "ptr": current.data_ptr(),
    "magic_ptr": magic_slots[idx].data_ptr(),  # dedicated slot
    ...
}
# poll dedicated slot, not data buffer
await self.wait_for_complete(magic_slots[idx])

# receive_weights: write magic to magic_ptr, not ptr
ret = self.engine.transfer_sync_write(
    self.buffer_info["session_id"],
    self.magic_buf.data_ptr(),
    magic_ptr,    # ← dedicated signal address, not data buffer
    4,
)
# forward magic_ptr for next rank in daisy chain
info["magic_ptr"] = magic_slots[idx % 2].data_ptr()
```

---

## Bug 2: mRoPE Nested Tensor `maybe_fix_3d_position_ids` Incomplete

**Severity:** High
**Affected file:** `verl/utils/tensordict_utils.py`
**Upstream status:** Original one-liner (`pos._ragged_idx = 2`) is in upstream. Full fix is not.

### Symptom

Training step crashes with:
```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to N
(input tensor's size at dimension D), but got split_sizes=[M]
```

Two crash patterns observed:
- **Case A:** `sum exactly to 11 (dimension 1), got split_sizes=[4]` — offsets track mRoPE dim
- **Case B:** `sum exactly to 4 (dimension 0), got split_sizes=[4978]` — offsets track seq_len dim

### When it happens

After Ray pickle/unpickle of TensorDict (e.g., when data is sent between trainer and rollout workers via Ray object store), nested tensor `_ragged_idx` reverts to its default value of 1. For Qwen3's mRoPE position_ids (shape per sample: `[mrope_channels=4, seq_len]`), `ragged_idx` should be 2 (ragged over seq_len). The mismatch between stored offsets and reverted `_ragged_idx` causes `unbind()` / `split()` to use the wrong dimension.

This crash occurs at `engine_workers.py:243` (`train_mini_batch` entry) when `maybe_fix_3d_position_ids(data)` is called.

### Root cause

**Original code** — just sets the attribute, doesn't rebuild the tensor:

```python
def maybe_fix_3d_position_ids(data):
    if ... data["position_ids"].is_nested:
        data["position_ids"]._ragged_idx = 2
```

Setting `_ragged_idx = 2` without rebuilding causes an inconsistency: the internal `offsets` tensor (computed at construction time) doesn't match the new `_ragged_idx`. When `unbind()` tries to split along the dimension indicated by `_ragged_idx=2`, it uses offsets that were computed for `_ragged_idx=1`.

Two cases arise depending on how the nested tensor was originally constructed:

| Case | Offsets track | `_ragged_idx` after unpickle | Split crashes because |
|------|--------------|------------------------------|----------------------|
| A | dim 0 (mRoPE channels, e.g. `[0, 4, 8]`) | 1 → set to 2 | `sum([4,4]) = 8 ≠ values.shape[1]` |
| B | dim 1 (seq_len, e.g. `[0, 1821]`) | 1 → set to 2 | `sum([1821]) = 1821 ≠ values.shape[0]=4` |

### Fix

Distinguish the two cases by checking which dimension the offsets are consistent with:

```python
def maybe_fix_3d_position_ids(data):
    if "position_ids" in data.keys() and data["position_ids"].dim() == 3 and data["position_ids"].is_nested:
        pos = data["position_ids"]
        if getattr(pos, "_ragged_idx", 1) != 2:
            offsets = pos.offsets()
            values = pos.values()
            splits = (offsets[1:] - offsets[:-1]).tolist()
            total = sum(splits)

            if total == values.shape[0]:
                # Case A: offsets track dim 0 (mRoPE channels).
                # Rebuild with ragged_idx=2 so offsets will track seq_len.
                individual_tensors = list(torch.split(values, splits, dim=0))
                data["position_ids"] = nested_tensor_from_tensor_list(
                    individual_tensors, ragged_idx=2
                )
            elif total == values.shape[1]:
                # Case B: offsets already track dim 1 (seq_len).
                # Tensor is internally consistent for ragged_idx=2;
                # only the attribute was lost during serialization.
                pos._ragged_idx = 2
```

---

## Bug 3: `ppo_mini_batch_size` Silently Multiplied by `rollout.n`

**Severity:** Medium
**Affected file:** `verl/experimental/one_step_off_policy/ray_trainer.py` (line 1311)
**Upstream status:** Not documented.

### Symptom

```
AssertionError: 1 % 2 != 0
```

Training fails at the mini-batch splitting step.

### When it happens

When `ppo_mini_batch_size` is set without accounting for the internal multiplication by `rollout.n`. For example: `train_batch_size=4, ppo_mini_batch_size=8, rollout.n=2, dp_size=8`. The effective global mini-batch becomes `8 * 2 = 16`, but per-GPU batch is `(4 * 2) / 8 = 1`. Then `1 % 2 != 0` triggers the assertion.

### Fix

Set `ppo_mini_batch_size` equal to `train_batch_size` (i.e. 4). The `× rollout.n` multiplication should be documented in the config schema or removed.

---

## Bug 4: `flush_cache` Deadlock After Weight Sync

**Severity:** High
**Affected files:**
- `verl/checkpoint_engine/base.py` (`CheckpointEngineManager.update_weights`)
- `verl/workers/rollout/sglang_rollout/sglang_rollout.py` (`ServerAdapter.update_weights`)
- SGLang `sglang/srt/weight_sync/utils.py` (`update_weights`)

**Upstream status:** Step 7-8-9 reordering is in `base.py`. `flush_cache=False` patch to SGLang's `update_weights` is not upstream.

### Symptom

After Mooncake weight sync, the SGLang scheduler enters a CPU spinloop (58–96% CPU per scheduler process, 0% GPU utilization) and stops responding to both HTTP and Ray actor requests. All inference requests time out at 60s.

### When it happens

During `CheckpointEngineManager.update_weights()` step 5. The flow is:

1. Step 1: `abort_replicas()` — pauses the SGLang scheduler
2. Step 5: `ServerAdapter.update_weights()` calls `sgl_update_weights()` then `self._engine.flush_cache()`
3. Inside `sgl_update_weights`: `UpdateWeightsFromTensorReqInput` is created with `flush_cache=True` (SGLang default)
4. SGLang's `flush_cache_after_weight_update()` calls `scheduler.flush_cache()` which requires `is_fully_idle()` — but the scheduler is paused with aborted requests in `running_batch` → assert fails or deadlocks
5. `ServerAdapter.update_weights()` line 358 then calls `self._engine.flush_cache()` as a separate HTTP request — scheduler still paused → hangs or fails

### Fix

Two changes required:

1. **Set `flush_cache=False` in SGLang's `sgl_update_weights`** — prevents the assert/deadlock inside the weight update handler:
   ```python
   # In sglang/srt/weight_sync/utils.py
   UpdateWeightsFromTensorReqInput(
       serialized_named_tensors=[...],
       load_format=load_format,
       flush_cache=False,  # ← added
   )
   ```

2. **Reorder steps 7-8 in `CheckpointEngineManager.update_weights`** — resume the scheduler before flushing cache:
   ```python
   # 7. resume generation FIRST
   await self.resume_generation_replicas()
   # 8. THEN flush cache (scheduler is now idle)
   for replica in self.replicas:
       await replica.clear_kv_cache()
   ```

---

## Bug 5: FSDP `param_offload=True` Causes Zeroed Weights

**Severity:** High
**Affected files:** Interaction between FSDP engine and checkpoint engine weight extraction.
**Upstream status:** Not fixed (no guard in code).

### Symptom

Mooncake transfers ~51 GB of zeros. The rollout model has all-zero weights. Inference produces random/uniform token distribution.

### When it happens

When `actor_rollout_ref.actor.fsdp_config.param_offload=True` (or `ref.fsdp_config.param_offload=True`) is set with any cross-node checkpoint engine (Mooncake, NCCL, NIXL).

### Root cause

`get_per_tensor_param()` returns a **lazy generator**. The generator holds references to GPU tensors but doesn't materialize them until iteration. After `get_per_tensor_param()` returns, `offload_fsdp_model_to_cpu()` is called, which **zeros the GPU buffers** (the CPU copy is the canonical version). When the checkpoint engine later iterates the generator to send weights, it reads the zeroed GPU buffers.

### Fix

Set `param_offload=False` for both actor and ref:

```yaml
actor_rollout_ref.actor.fsdp_config.param_offload: False
actor_rollout_ref.ref.fsdp_config.param_offload: False
```

A proper fix would either: (a) materialize the generator before offloading, or (b) read from the CPU copy instead of the zeroed GPU buffers.

---

## Bug 6: `vllm_provider.py` Incompatible with `transformers >= 5.6.0`

**Severity:** Medium
**Affected file:** `recipe/agentic/proxyserver/vllm_provider.py`
**Upstream status:** Not fixed.

### Symptom

```
ValueError: input_ids should be a list of lists for batch processing.
```

Agent proxy server crashes when tokenizing prompts for SGLang inference.

### When it happens

When using `transformers >= 5.6.0` where `tokenizer.apply_chat_template(tokenize=True)` returns a `BatchEncoding` object (dict-like with keys `input_ids` and `attention_mask`) instead of a plain `list[int]`.

### Root cause

```python
# vllm_provider.py:_generate
prompt_ids: list[int] = self.tokenizer.apply_chat_template(...)
```

The type annotation says `list[int]`, but `BatchEncoding` is returned. When passed to SGLang's `GenerateReqInput(input_ids=BatchEncoding)`, the `_determine_batch_size` function sees `len(BatchEncoding) = 2` (number of dict keys) and `isinstance(BatchEncoding[0], int) = False` (it's an `Encoding` object), so it treats it as a batch → `_expand_inputs` → error.

### Fix

```python
encoded = self.tokenizer.apply_chat_template(...)
prompt_ids: list[int] = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
```
