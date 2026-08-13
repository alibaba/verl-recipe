#!/usr/bin/env python3
"""Mooncake weight-sync SENDER (rank 0) for the external-SGLang test.

Runs on a training node. Loads a HF checkpoint from disk and streams its weights
over the Mooncake TransferEngine to the receiver (rank 1), which pushes them into
an externally-deployed SGLang via CUDA IPC.

Coordination: rank 0 binds a TCPStore at --bind-addr:--port; the receiver
connects to it. Start the sender first, then the receiver.
"""
import argparse
import asyncio
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--bind-addr", required=True, help="this pod's IP, reachable by the receiver")
parser.add_argument("--port", type=int, default=29500)
parser.add_argument("--model-path", default="/mnt/models/Qwen2.5-3B-Instruct")
parser.add_argument("--bucket-mb", type=int, default=2048)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--zero-name", default="", help="if set, this tensor is sent as all-zeros (mutation test)")
args = parser.parse_args()

os.environ.setdefault("RANK", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from verl.checkpoint_engine.mooncake_checkpoint_engine import MooncakeCheckpointEngine  # noqa: E402


def weight_generator(model_path):
    """Yield (name, cuda bf16 tensor) for every tensor in the HF checkpoint."""
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    assert files, f"no safetensors under {model_path}"
    for f in files:
        sd = load_file(f, device="cpu")
        for name, tensor in sd.items():
            t = tensor.to(device="cuda", dtype=torch.bfloat16)
            if args.zero_name and name == args.zero_name:
                print(f"[sender] MUTATION: zeroing {name} {tuple(t.shape)}", flush=True)
                t = torch.zeros_like(t)
            yield name, t


async def main():
    bucket_size = args.bucket_mb << 20
    engine = MooncakeCheckpointEngine(bucket_size=bucket_size, device="cuda", is_master=True)
    metadata = {"addr": args.bind_addr, "port": args.port}
    print(f"[sender] init_process_group rank=0 world_size=2 rendezvous={metadata}", flush=True)
    engine.init_process_group(rank=0, world_size=2, metadata=metadata)
    print("[sender] group ready; streaming weights...", flush=True)

    n = 0
    gen = weight_generator(args.model_path)

    def counting_gen():
        nonlocal n
        for name, t in gen:
            n += 1
            yield name, t

    await engine.send_weights(counting_gen())
    print(f"[sender] DONE, sent {n} tensors", flush=True)
    engine.finalize()


if __name__ == "__main__":
    asyncio.run(main())
