#!/usr/bin/env python3
"""
Verify: Mooncake transfer_sync_write local side effect.

This script tests whether transfer_sync_write to a REMOTE GPU also modifies
memory on the LOCAL GPU. It does NOT test the daisy-chain protocol — it only
tests the raw RDMA primitive.

Setup: 2 GPUs on the same node (e.g., cuda:0 and cuda:1).

    GPU 0 (rank 0): holds data_buf with known pattern + magic_buf
    GPU 1 (rank 1): holds data_buf with known pattern + magic_buf

    Test: GPU 1 calls transfer_sync_write to write magic into GPU 0's data_buf.
    Check: Does GPU 1's own data_buf get modified?

Expected (correct RDMA): GPU 1's data_buf unchanged after the write.
Actual (bug):           GPU 1's data_buf[:4] == magic after the write.

Requirements:
    - 2+ GPUs on the same node
    - mooncake installed (pip install mooncake)
    - CUDA available

Usage:
    torchrun --nproc_per_node=2 verify_transfer_sync_write_side_effect.py
"""

import os
import sys
import torch
import torch.distributed as dist

MAGIC = [0xAB, 0xDC, 0xEF, 0x88]
PATTERN = [0x41, 0x42, 0x43, 0x44]  # "ABCD" — clearly different from MAGIC

# ─── Guard ──────────────────────────────────────────────────────────────────

try:
    from mooncake.engine import TransferEngine
except ImportError:
    print("ERROR: mooncake not installed. Run: pip install mooncake")
    sys.exit(1)

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    print("ERROR: Need at least 2 CUDA GPUs. Available:", torch.cuda.device_count())
    sys.exit(1)


def main():
    # ─── Init torch distributed ─────────────────────────────────────────────
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world >= 2, f"Need world_size >= 2, got {world}"

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # ─── Init Mooncake TransferEngine ───────────────────────────────────────
    import ray
    engine = TransferEngine()
    hostname = ray.util.get_node_ip_address().strip("[]")
    ret = engine.initialize(hostname, "P2PHANDSHAKE", "rdma", "")
    assert ret == 0, f"TransferEngine initialize failed ret={ret}"

    rpc_port = engine.get_rpc_port()
    session_id = f"{hostname}:{rpc_port}"

    # ─── Exchange session info ──────────────────────────────────────────────
    # Only need rank 0 and rank 1
    if rank < 2:
        info = {"session_id": session_id}
        all_info = [None] * world
        dist.all_gather_object(all_info, info)

        peer_session = all_info[1 - rank]["session_id"]

        # ─── Allocate and register buffers ──────────────────────────────────
        SIZE = 4096
        data_buf = torch.full((SIZE,), 0x00, dtype=torch.uint8, device=device)
        data_buf[:4] = torch.tensor(PATTERN, dtype=torch.uint8, device=device)

        magic_buf = torch.tensor(MAGIC, dtype=torch.uint8, device=device)

        ret = engine.batch_register_memory(
            [data_buf.data_ptr(), magic_buf.data_ptr()],
            [SIZE, 4],
        )
        assert ret == 0, f"batch_register_memory failed ret={ret}"

        # Exchange data_buf pointers
        ptrs = [None] * world
        dist.all_gather_object(ptrs, data_buf.data_ptr())
        peer_data_ptr = ptrs[1 - rank]

        # ─── Synchronize: both ranks ready ──────────────────────────────────
        dist.barrier()
        torch.cuda.synchronize()

        # ─── Snapshot before ────────────────────────────────────────────────
        before = data_buf[:8].clone().cpu().tolist()

        # ─── Rank 1 writes magic to Rank 0's data_buf ──────────────────────
        if rank == 1:
            ret = engine.transfer_sync_write(
                peer_session,
                magic_buf.data_ptr(),   # local source: magic bytes
                peer_data_ptr,           # remote destination: rank 0's data_buf
                4,
            )
            assert ret == 0, f"transfer_sync_write failed ret={ret}"

        # ─── Synchronize: wait for write to complete ────────────────────────
        torch.cuda.synchronize()
        dist.barrier()

        # ─── Check results ──────────────────────────────────────────────────
        after = data_buf[:8].clone().cpu().tolist()
        is_magic = after[:4] == MAGIC

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Rank 0 (remote target) — GPU: {device}")
            print(f"{'='*60}")
            print(f"  data_buf[:8] before: {[hex(b) for b in before]}")
            print(f"  data_buf[:8] after:  {[hex(b) for b in after]}")
            print(f"  Magic received:      {is_magic}")
            print(f"  → Rank 0 received magic from Rank 1. {'OK' if is_magic else 'FAILED — magic not received'}")

        if rank == 1:
            print(f"\n{'='*60}")
            print(f"Rank 1 (caller/writer) — GPU: {device}")
            print(f"{'='*60}")
            print(f"  data_buf[:8] before: {[hex(b) for b in before]}")
            print(f"  data_buf[:8] after:  {[hex(b) for b in after]}")
            print(f"  Local side effect:   {is_magic}")

            if is_magic:
                print(f"\n  *** BUG CONFIRMED ***")
                print(f"  transfer_sync_write targeted Rank 0's GPU ({hex(peer_data_ptr)})")
                print(f"  but also modified Rank 1's own GPU ({hex(data_buf.data_ptr())})")
                print(f"  These are DIFFERENT addresses on DIFFERENT GPUs.")
                print(f"  This violates standard RDMA WRITE semantics.")
            else:
                changed = after[:4] != before[:4]
                if changed:
                    print(f"\n  PARTIAL SIDE EFFECT: data_buf[:4] changed but not to magic.")
                    print(f"  before: {[hex(b) for b in before[:4]]}")
                    print(f"  after:  {[hex(b) for b in after[:4]]}")
                else:
                    print(f"\n  No local side effect detected. data_buf unchanged.")
                    print(f"  transfer_sync_write behaved correctly.")

        print()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
