# External SGLang weight-sync test — RESULT: PASS

Validated on the Wulan ACK cluster (2× H20 nodes) that verl can push training
weights into an **externally-deployed** SGLang instance (one verl did NOT launch),
using the **Mooncake** TransferEngine (RDMA) + **CUDA IPC**.

## Topology used

- `rollout-workers` scaled to 0 (freed 8 GPUs on `e01-cn-iaj4pmqd601`).
- Standalone pod `sglang-external` (1× H20, node 601): SGLang 0.5.12 serving
  `Qwen2.5-3B-Instruct` (TP=1) at `:30000`, launched by us.
- **Receiver** (`receiver.py`) runs as a 2nd process in that pod → shares the GPU
  + IPC namespace with SGLang. It is Mooncake rank 1; it pushes each received
  batch into local SGLang via `sgl_update_weights` (`/update_weights_from_tensor`,
  CUDA IPC).
- **Sender** (`sender.py`) runs in the training pod (node 602, 10.8.0.42). Mooncake
  rank 0; loads a HF checkpoint from `/mnt/models` and streams it.
- Cross-node transfer: Mooncake RDMA, ~0.3–0.5 GB/s, 434 tensors / 4.26 GB, ~12 s.

## Verification (behavioral, reversible)

Prompt `"The capital of France is"`, greedy, 16 tokens:

| Round | Sender weights | SGLang generation | Meaning |
|------|-----------------|-------------------|---------|
| baseline | (none) | ` Paris. The capital of Germany is Berlin...` | reference |
| 1 | base, unchanged | ` Paris. The capital of Germany is Berlin...` | pipeline does not corrupt |
| 2 | base w/ `model.embed_tokens.weight` **zeroed** | `!!!!!!!!!!!!!!!!` | pushed weights **take effect** |
| 3 | base, restored | ` Paris. The capital of Germany is Berlin...` | **reversible** round-trip |

Zeroed input embeddings → constant logits → argmax = token 0 (`!`) repeated.
Round 3 returning to the exact baseline rules out coincidence: verl fully controls
the external SGLang's weights over the wire.

## Gotchas found (important for productionizing)

1. **Mooncake ignores `MOONCAKE_PROTOCOL`.** `MooncakeCheckpointEngine.__init__`
   hardcodes the `rdma` transport. Both pods must have RDMA. The SGLang pod
   originally found "0 HCAs" and fell back to TCP → transfer `assert ret == 0`
   failed. Fix: request `rdma/hca: "1"` on the SGLang pod (see
   `../k8s/sglang-external.yaml`).
2. **GPU admission race.** After scaling rollout to 0, the SGLang pod must be
   (re)created *after* the device plugin frees the GPUs, else
   `UnexpectedAdmissionError: devices unavailable`.
3. **`get_weights_by_name` is unsupported for Qwen2** in SGLang 0.5.12
   ("TODO: Add support for Qwen models") → no byte-level readback; verify
   behaviorally instead.
4. **CUDA IPC needs shared GPU + IPC namespace.** Running the receiver as a 2nd
   process in the SGLang container guarantees this (matches verl's own colocated
   CE-worker model). A separate sidecar *container* would need
   `shareProcessNamespace: true` + matched `NVIDIA_VISIBLE_DEVICES`.

## Reproduce

```bash
export KUBECONFIG=~/.kube/config-wulan
kubectl apply -f recipe/remote_megatron_sglang/k8s/sglang-external.yaml
kubectl cp recipe/remote_megatron_sglang/test/external_sglang/receiver.py default/sglang-external:/tmp/receiver.py -c sglang
# launch sglang in the pod: python3 -m sglang.launch_server --model-path /mnt/models/Qwen2.5-3B-Instruct --tp 1 --host 0.0.0.0 --port 30000 --mem-fraction-static 0.6 --trust-remote-code
# sender (training pod): python3 sender.py --bind-addr <trainer_ip> --port 29500 --model-path /mnt/models/Qwen2.5-3B-Instruct [--zero-name model.embed_tokens.weight]
# receiver (sglang pod):  python3 receiver.py --sender-addr <trainer_ip> --sender-port 29500 --model-path /mnt/models/Qwen2.5-3B-Instruct
```
Start the sender first (binds the rank-0 TCPStore), then the receiver.
