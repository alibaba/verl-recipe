#!/usr/bin/env python3
"""Mooncake weight-sync RECEIVER (rank 1) for the external-SGLang test.

Runs as a second process INSIDE the SGLang pod (shares the GPU + IPC namespace
with the SGLang server). Receives weights over the Mooncake TransferEngine from
the sender (rank 0), then pushes each batch into the local, externally-deployed
SGLang server via CUDA IPC (`/update_weights_from_tensor`).

This mirrors verl's colocated CheckpointEngineWorker -> ServerAdapter path, but
against an SGLang instance that verl did NOT launch.
"""
import argparse
import asyncio
import os

parser = argparse.ArgumentParser()
parser.add_argument("--sender-addr", required=True, help="rank-0 TCPStore host")
parser.add_argument("--sender-port", type=int, default=29500)
parser.add_argument("--sglang-host", default="127.0.0.1")
parser.add_argument("--sglang-port", type=int, default=30000)
parser.add_argument("--model-path", default="/mnt/models/Qwen2.5-3B-Instruct")
parser.add_argument("--bucket-mb", type=int, default=2048)
parser.add_argument("--chunk-tensors", type=int, default=64)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

os.environ.setdefault("RANK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

import torch  # noqa: E402
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402

from verl.checkpoint_engine.mooncake_checkpoint_engine import MooncakeCheckpointEngine  # noqa: E402
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter  # noqa: E402
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights  # noqa: E402


async def main():
    bucket_size = args.bucket_mb << 20

    # A trivial (world_size=1) torch.distributed group so init_device_mesh works;
    # sgl_update_weights uses the mesh to gather IPC handles across TP ranks (TP=1 here).
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", init_method="tcp://127.0.0.1:29666", rank=0, world_size=1
        )
    device_mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("infer_tp",))

    engine = MooncakeCheckpointEngine(bucket_size=bucket_size, device="cuda", is_master=False)
    metadata = {"addr": args.sender_addr, "port": args.sender_port}
    print(f"[receiver] init_process_group rank=1 world_size=2 rendezvous={metadata}", flush=True)
    engine.init_process_group(rank=1, world_size=2, metadata=metadata)

    server = AsyncHttpServerAdapter(
        model_path=args.model_path,
        host=args.sglang_host,
        port=args.sglang_port,
        launch_server=False,
        trust_remote_code=True,
    )
    print(f"[receiver] pushing to SGLang at {args.sglang_host}:{args.sglang_port}", flush=True)

    # Clone tensors out of the reused mooncake bucket, then push in chunks.
    batch = []
    total = 0

    async def flush():
        nonlocal batch, total
        if not batch:
            return
        await sgl_update_weights(
            engine=server,
            params_batch=batch,
            device_mesh_key="infer_tp",
            device_mesh=device_mesh,
        )
        total += len(batch)
        print(f"[receiver] pushed {total} tensors", flush=True)
        batch = []

    async for name, tensor in engine.receive_weights():
        batch.append((name, tensor.clone()))
        if len(batch) >= args.chunk_tensors:
            await flush()
    await flush()

    await server.flush_cache()
    print(f"[receiver] DONE, updated {total} tensors", flush=True)
    engine.finalize()


if __name__ == "__main__":
    asyncio.run(main())
