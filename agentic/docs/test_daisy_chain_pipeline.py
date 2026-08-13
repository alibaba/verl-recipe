#!/usr/bin/env python3
"""
Test: Mooncake daisy-chain weight sync pipeline.

Reproduces the full daisy-chain protocol (R0 → R1 → R2 → ...) from
MooncakeCheckpointEngine to detect:

  Bug 1 (protocol): magic written to prev rank's DATA buffer overwrites
                     tensor data at offset 0.
  Bug 2 (Mooncake): transfer_sync_write to a remote GPU also modifies
                     memory on the calling GPU (local side effect).

Both bugs corrupt the first 4 bytes (2 bf16 elements) of the first tensor
in the affected bucket — typically embed_tokens.weight.

Protocol modelled (receive_weights, ORIGINAL buggy version):
  1. RDMA read from prev rank into local double-buffer
  2. Forward buffer ptr to next rank (daisy chain)
  3. Yield tensor VIEWS from the buffer (no copy)
  4. transfer_sync_write magic to prev rank's DATA buffer  ← BUG
  5. Advance to next buffer slot

Requirements:
  - 3+ GPUs on the same node (for meaningful daisy chain)
  - mooncake installed
  - CUDA available

Usage:
  torchrun --nproc_per_node=3 test_daisy_chain_pipeline.py
  torchrun --nproc_per_node=3 test_daisy_chain_pipeline.py --buckets 4
  torchrun --nproc_per_node=3 test_daisy_chain_pipeline.py --fixed
"""

import argparse
import asyncio
import os
import pickle
import sys
import time

import torch
import torch.distributed as dist

# ─── Constants ───────────────────────────────────────────────────────────────

MAGIC_BYTES = [0xAB, 0xDC, 0xEF, 0x88]

# ─── Guards ──────────────────────────────────────────────────────────────────

try:
    from mooncake.engine import TransferEngine
except ImportError:
    print("ERROR: mooncake not installed. pip install mooncake")
    sys.exit(1)

if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
    print(f"ERROR: Need ≥ 3 GPUs, found {torch.cuda.device_count()}")
    sys.exit(1)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _magic_tensor(device):
    return torch.tensor(MAGIC_BYTES, dtype=torch.uint8, device=device)


def _is_magic(t):
    return t.cpu().tolist()[:4] == MAGIC_BYTES


def _hex4(lst):
    """Format first 4 elements of a list as hex."""
    return [hex(b) if isinstance(b, int) else b for b in lst[:4]]


# ─── Simple point-to-point messaging over TCPStore ──────────────────────────

class StoreMessenger:
    """send_obj / recv_obj over dist.TCPStore, mimicking StatelessProcessGroup."""

    def __init__(self, store, rank):
        self._store = store
        self._rank = rank
        self._send_seq = {}   # dst -> counter
        self._recv_seq = {}   # src -> counter

    def send_obj(self, obj, dst: int):
        seq = self._send_seq.get(dst, 0)
        key = f"msg_{self._rank}_{dst}_{seq}"
        self._store.set(key, pickle.dumps(obj))
        self._send_seq[dst] = seq + 1

    def recv_obj(self, src: int):
        seq = self._recv_seq.get(src, 0)
        key = f"msg_{src}_{self._rank}_{seq}"
        data = self._store.get(key)          # blocks until key exists
        self._recv_seq[src] = seq + 1
        return pickle.loads(bytes(data))


# ─── Mooncake setup ──────────────────────────────────────────────────────────

def init_mooncake(rank, bucket_size, device):
    """Initialize TransferEngine, allocate and register buffers."""
    engine = TransferEngine()
    hostname = os.environ.get("HOSTNAME", "127.0.0.1")
    try:
        import ray
        hostname = ray.util.get_node_ip_address().strip("[]")
    except ImportError:
        pass

    ret = engine.initialize(hostname, "P2PHANDSHAKE", "rdma", "")
    assert ret == 0, f"TransferEngine initialize failed ret={ret}"

    rpc_port = engine.get_rpc_port()
    session_id = f"{hostname}:{rpc_port}"

    buf = torch.zeros(2 * bucket_size, dtype=torch.uint8, device=device)
    magic_buf = _magic_tensor(device)
    magic_recv = torch.zeros(8, dtype=torch.uint8, device=device)

    ret = engine.batch_register_memory(
        [buf.data_ptr(), magic_buf.data_ptr(), magic_recv.data_ptr()],
        [2 * bucket_size, 4, 8],
    )
    assert ret == 0, f"batch_register_memory failed ret={ret}"

    return {
        "engine": engine,
        "session_id": session_id,
        "buf": buf,
        "bufs": [buf[:bucket_size], buf[bucket_size:]],
        "magic_buf": magic_buf,
        "magic_recv": magic_recv,
        "magic_slots": [magic_recv[:4], magic_recv[4:]],
    }


async def wait_for_complete(buf, device):
    """Poll until magic appears in buf[:4], then reset."""
    magic = _magic_tensor(device)
    while not torch.equal(buf[:4], magic):
        await asyncio.sleep(0)
    buf[:4] = 0


# ─── Rank 0: send_weights ────────────────────────────────────────────────────

async def send_weights(rank, world, mc, msg, tensors, bucket_size, use_fixed):
    """R0: pack tensors into buckets, send to R1, wait for completion."""
    engine = mc["engine"]
    bufs = mc["bufs"]
    device = mc["buf"].device
    idx = 0
    current = bufs[idx]
    offset = 0
    bucket_meta = {}
    should_wait = False

    for name, weight in tensors:
        weight = weight.to(device=device, dtype=torch.bfloat16)
        raw = weight.view(-1).view(torch.uint8)

        if offset + raw.numel() > bucket_size:
            torch.cuda.synchronize()
            info = {
                "bucket_meta": bucket_meta,
                "ptr": current.data_ptr(),
                "len": offset,
                "is_last": False,
            }
            if use_fixed:
                info["magic_ptr"] = mc["magic_slots"][idx].data_ptr()
            msg.send_obj(info, 1)

            idx ^= 1
            current = bufs[idx]
            bucket_meta = {}
            offset = 0

            if should_wait:
                target = mc["magic_slots"][idx] if use_fixed else current
                await wait_for_complete(target, device)
            should_wait = True

        assert offset + raw.numel() <= bucket_size, (
            f"Tensor {name}({weight.shape}) too large for bucket"
        )
        bucket_meta[name] = {
            "shape": weight.shape, "dtype": weight.dtype, "offset": offset,
        }
        current[offset : offset + raw.numel()].copy_(raw, non_blocking=True)
        offset += raw.numel()

    # Send last bucket
    torch.cuda.synchronize()
    info = {
        "bucket_meta": bucket_meta,
        "ptr": current.data_ptr(),
        "len": offset,
        "is_last": True,
    }
    if use_fixed:
        info["magic_ptr"] = mc["magic_slots"][idx].data_ptr()
    msg.send_obj(info, 1)
    target = mc["magic_slots"][idx] if use_fixed else current
    await wait_for_complete(target, device)

    return {"status": "ok", "buckets_sent": idx + 1}


# ─── Rank 1+: receive_weights (daisy chain) ──────────────────────────────────

async def receive_weights(rank, world, mc, msg, bucket_size, use_fixed,
                          prev_session, prev_ptr):
    """
    Daisy-chain receiver: read from prev rank, forward to next, yield views,
    write completion magic.

    Mirrors MooncakeCheckpointEngine.receive_weights (original buggy version).
    Returns consumed tensors and diagnostic checkpoints.
    """
    engine = mc["engine"]
    bufs = mc["bufs"]
    device = mc["buf"].device
    magic_buf = mc["magic_buf"]

    idx = 0
    current = bufs[idx]
    all_tensors = {}
    checkpoints = []

    while True:
        # ── Receive bucket metadata from prev rank ──
        info = msg.recv_obj(rank - 1)

        # ── Double-buffer backpressure (mirrors real code: idx >= 2) ──
        if idx >= 2 and rank < world - 1:
            await wait_for_complete(current, device)

        remote_ptr = info["ptr"]

        # ── CHECK A: before RDMA read ──
        a = current[:4].clone().cpu()

        # ── RDMA READ: copy from prev rank's buffer ──
        ret = engine.transfer_sync_read(
            prev_session, current.data_ptr(), remote_ptr, info["len"],
        )
        assert ret == 0, f"transfer_sync_read failed ret={ret}"

        # ── CHECK B: after RDMA read ──
        torch.cuda.synchronize()
        b = current[:4].clone().cpu()

        # ── Forward to next rank (daisy chain) ──
        is_intermediate = rank < world - 1
        if is_intermediate:
            fwd = dict(info)
            fwd["ptr"] = current.data_ptr()
            if use_fixed:
                fwd["magic_ptr"] = mc["magic_slots"][idx].data_ptr()
            msg.send_obj(fwd, rank + 1)

        # ── CHECK C: after forward, before yield ──
        c = current[:4].clone().cpu()

        # ── Consumer: clone tensor views ──
        bucket_tensors = {}
        for name, meta in info["bucket_meta"].items():
            dtype, shape = meta["dtype"], meta["shape"]
            size = dtype.itemsize * shape.numel()
            tensor = current[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
            bucket_tensors[name] = tensor.clone()
            all_tensors[name] = bucket_tensors[name]

        # ── CHECK D: after consumer clone, CUDA-synced ──
        torch.cuda.synchronize()
        d = current[:4].clone().cpu()

        # ── Write completion magic to prev rank ──
        if use_fixed:
            magic_dest = info["magic_ptr"]
        else:
            magic_dest = remote_ptr  # ORIGINAL BUG

        ret = engine.transfer_sync_write(
            prev_session, magic_buf.data_ptr(), magic_dest, 4,
        )
        assert ret == 0, f"transfer_sync_write failed ret={ret}"

        # ── CHECK E: after magic write — detects local side effect ──
        torch.cuda.synchronize()
        e = current[:4].clone().cpu()

        ckpt = {
            "bucket": idx,
            "A_pre_read":   a.tolist(),
            "B_post_read":  b.tolist(),
            "C_pre_yield":  c.tolist(),
            "D_post_yield": d.tolist(),
            "E_post_magic": e.tolist(),
            "D_eq_E":       d.tolist() == e.tolist(),
            "E_is_magic":   _is_magic(e),
            "B_is_magic":   _is_magic(b),
            "data_ptr":     hex(current.data_ptr()),
            "remote_ptr":   hex(remote_ptr),
            "is_intermediate": is_intermediate,
        }
        checkpoints.append(ckpt)

        # ── Advance double-buffer ──
        idx += 1
        current = bufs[idx % 2]
        torch.cuda.synchronize()

        if info["is_last"]:
            break

    return {"tensors": all_tensors, "checkpoints": checkpoints, "buckets": idx}


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-size", type=int, default=64 * 1024)
    parser.add_argument("--fixed", action="store_true",
                        help="Use magic_ptr (fixed) instead of data ptr (buggy)")
    parser.add_argument("--tensor-shape", type=str, default="32,32",
                        help="Comma-separated shape for test tensor")
    args = parser.parse_args()

    use_fixed = args.fixed
    bucket_size = args.bucket_size
    tensor_shape = [int(x) for x in args.tensor_shape.split(",")]

    # ── Init distributed ──
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world >= 3, f"Need ≥ 3 ranks for daisy chain, got {world}"

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # ── Init Mooncake ──
    mc = init_mooncake(rank, bucket_size, device)

    # ── Exchange session info via all_gather ──
    my_info = {"session_id": mc["session_id"], "ptr": mc["buf"].data_ptr()}
    all_info = [None] * world
    dist.all_gather_object(all_info, my_info)

    # ── Point-to-point messaging ──
    tcp_store = dist.TCPStore(
        os.environ.get("MASTER_ADDR", "127.0.0.1"),
        int(os.environ.get("MASTER_PORT", "29500")) + 1,
        world,
        rank == 0,
    )
    msg = StoreMessenger(tcp_store, rank)

    # ── Create test tensors (known pattern) ──
    numel = 1
    for s in tensor_shape:
        numel *= s
    indices = torch.arange(numel, dtype=torch.float32)
    values = torch.sin(indices.float()) * 0.01
    embed = values.view(tensor_shape).to(torch.bfloat16)

    tensors_to_send = [("embed_tokens", embed)]

    # ── Run protocol ──
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()

    if rank == 0:
        result = await send_weights(
            rank, world, mc, msg, tensors_to_send, bucket_size, use_fixed,
        )
    else:
        prev_session = all_info[rank - 1]["session_id"]
        prev_ptr = all_info[rank - 1]["ptr"]
        result = await receive_weights(
            rank, world, mc, msg, bucket_size, use_fixed,
            prev_session, prev_ptr,
        )

    dist.barrier()
    elapsed = time.time() - t0

    # ── Gather results ──
    results = [None] * world
    my_result = {
        "rank": rank,
        "data": {
            k: (v if k != "tensors" else {
                name: t.cpu().tolist() for name, t in v.items()
            })
            for k, v in result.items()
        },
    }
    dist.all_gather_object(results, my_result)

    # ── Print diagnostics ──
    mode = "FIXED (magic_ptr)" if use_fixed else "BUGGY (data ptr)"
    print(f"\n{'='*70}")
    print(f"  Daisy-Chain Pipeline Test  |  Mode: {mode}")
    print(f"  Ranks: {world}  |  Time: {elapsed:.2f}s")
    print(f"{'='*70}\n")

    for r in range(1, world):
        res = results[r]["data"]
        role = "intermediate" if r < world - 1 else "final"
        print(f"── Rank {r} ({role}) ──")
        for ckpt in res["checkpoints"]:
            b = ckpt["bucket"]
            flag_b = " ← MAGIC in RDMA read!" if ckpt["B_is_magic"] else ""
            flag_e = " ← LOCAL SIDE EFFECT!" if not ckpt["D_eq_E"] else ""
            print(f"  B{b}:")
            print(f"    B-post-read:  {_hex4(ckpt['B_post_read'])}{flag_b}")
            print(f"    D-post-yield: {_hex4(ckpt['D_post_yield'])}")
            print(f"    E-post-magic: {_hex4(ckpt['E_post_magic'])}{flag_e}")
            print(f"    D==E: {ckpt['D_eq_E']}  data={ckpt['data_ptr']}  remote={ckpt['remote_ptr']}")
        print()

    # ── Data integrity ──
    print("── Data Integrity ──")
    any_corruption = False
    for r in range(1, world):
        res = results[r]["data"]
        if "tensors" not in res:
            continue
        for name, received_list in res["tensors"].items():
            received_t = torch.tensor(received_list, dtype=torch.bfloat16)
            match = torch.equal(received_t.cpu(), embed.cpu())

            if match:
                print(f"  R{r} [{name}]: OK")
            else:
                any_corruption = True
                flat_r = received_t.view(-1)
                flat_e = embed.view(-1)
                diff_mask = flat_r != flat_e
                n_diff = diff_mask.sum().item()
                first_idx = diff_mask.nonzero(as_tuple=True)[0]
                if len(first_idx) > 0:
                    i = first_idx[0].item()
                    print(f"  R{r} [{name}]: CORRUPTED")
                    print(f"    {n_diff} elements differ, first at index {i}")
                    print(f"    expected: {flat_e[i].float().item():.6e}")
                    print(f"    received: {flat_r[i].float().item():.6e}")
                    magic_bf16 = _magic_tensor("cpu").view(torch.bfloat16)
                    if i < 2 and flat_r[i] == magic_bf16[i]:
                        print(f"    → Value IS magic bytes as bf16 ({magic_bf16[i].float().item():.6e})")

    # ── Side effect check ──
    print(f"\n── Mooncake Local Side Effect ──")
    any_side_effect = False
    for r in range(1, world):
        res = results[r]["data"]
        for ckpt in res["checkpoints"]:
            if not ckpt["D_eq_E"]:
                any_side_effect = True
                print(f"  R{r} B{ckpt['bucket']}: DETECTED")
                print(f"    Wrote magic to {ckpt['remote_ptr']} (prev rank)")
                print(f"    Local buffer {ckpt['data_ptr']} changed:")
                print(f"      D (before): {_hex4(ckpt['D_post_yield'])}")
                print(f"      E (after):  {_hex4(ckpt['E_post_magic'])}")
    if not any_side_effect:
        print("  No local side effect detected.")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    print(f"  Protocol:             {mode}")
    print(f"  Data corruption:      {'YES' if any_corruption else 'No'}")
    print(f"  Local side effect:    {'YES' if any_side_effect else 'No'}")

    if any_corruption and any_side_effect:
        print(f"\n  Both bugs present: transfer_sync_write local side effect")
        print(f"  corrupted the buffer, and the next rank read the magic bytes")
        print(f"  as model weights (embed_tokens).")
    elif any_side_effect:
        print(f"\n  Local side effect detected but consumer data was OK.")
        print(f"  In production with more buckets/ranks, this WILL corrupt data.")
    elif any_corruption:
        print(f"\n  Data corrupted without local side effect — protocol bug.")
    else:
        print(f"\n  All checks passed.")

    print()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    asyncio.run(main())
