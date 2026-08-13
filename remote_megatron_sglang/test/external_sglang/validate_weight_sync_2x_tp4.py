#!/usr/bin/env python3
"""Validate HOST-MEMORY Mooncake weight-sync to TWO external SGLang instances, TP=4 each.

Exercises the real module ``ReceiverCE`` (recv_device="cpu" -> RDMA lands in
pinned HOST memory, staged to GPU only for the CUDA-IPC push) across a
multi-instance layout. Two topologies are supported via the INSTANCES env var:

  Two pods, 4 GPUs each (the clean fan-out; each pod sees GPUs 0-3):
    INSTANCES="sglang_a:<podA_ip>:30000:0,sglang_b:<podB_ip>:30000:0"

  One 8-GPU pod, two servers via --base-gpu-id 0/4 (the default):
    INSTANCES="sglang_node:127.0.0.1:30000:0,sglang_node:127.0.0.1:30001:4"

Each INSTANCES entry is ``resource:host:port:gpu_offset``:
  resource   Ray custom resource pinning the receivers to that SGLang's pod.
  host       IP used for the external /generate check (from the driver/head).
  port       SGLang HTTP port; receivers always talk to 127.0.0.1:<port> (local).
  gpu_offset physical base GPU for that instance's receivers (0 when the pod owns
             only its own 4 GPUs; 4 for the 2nd server in a shared 8-GPU pod).

Orchestration mirrors ExternalCheckpointManager.update_weights(): ONE Mooncake
P2P daisy chain (trainer rank0 -> all receivers, flat), each instance's TP
receivers form their own device_mesh for the per-TP IPC gather.

Correctness: base -> embed-zeroed (generation collapses) -> restored, on ALL
instances.

Run from the Ray head pod:  python3 validate_weight_sync_2x_tp4.py
"""
import asyncio
import glob
import os

import ray
import requests

from verl.checkpoint_engine.mooncake_checkpoint_engine import MooncakeCheckpointEngine
from verl.workers.rollout.external_sglang.checkpoint_manager import ReceiverCE

MODEL = os.environ.get("MODEL_PATH", "/mnt/models/Qwen3.6-27B")
TP = int(os.environ.get("SGLANG_TP", "4"))
RECV_DEVICE = os.environ.get("RECV_DEVICE", "cpu")
BUCKET = int(os.environ.get("BUCKET_MB", "3072")) << 20
CHUNK = int(os.environ.get("CHUNK_TENSORS", "8"))
ZERO_NAME = os.environ.get("ZERO_NAME", "model.language_model.embed_tokens.weight")

# resource:host:port:gpu_offset per external SGLang instance.
_DEFAULT_INSTANCES = "sglang_node:127.0.0.1:30000:0,sglang_node:127.0.0.1:30001:4"


def _parse_instances():
    out = []
    for spec in os.environ.get("INSTANCES", _DEFAULT_INSTANCES).split(","):
        resource, host, port, offset = spec.split(":")
        out.append({"resource": resource, "host": host, "port": int(port), "offset": int(offset)})
    return out


INSTANCES = _parse_instances()


@ray.remote(num_gpus=1, num_cpus=4)
class _TrainerSender:
    """Stand-in for the trainer's CE worker (Mooncake rank 0, sends full weights)."""

    def __init__(self, model_path, bucket_size):
        os.environ["RANK"] = "0"
        self.model_path = model_path
        self.engine = MooncakeCheckpointEngine(bucket_size=bucket_size, device="cuda", is_master=True)

    def prepare(self):
        return self.engine.prepare()

    def init_process_group(self, rank, world_size, metadata):
        self.engine.init_process_group(rank=rank, world_size=world_size, metadata=metadata)

    def send(self, zero_name=""):
        import torch
        from safetensors.torch import load_file

        n = {"c": 0}

        def gen():
            for f in sorted(glob.glob(os.path.join(self.model_path, "*.safetensors"))):
                sd = load_file(f, device="cpu")
                for name, t in sd.items():
                    t = t.to(device="cuda", dtype=torch.bfloat16)
                    if zero_name and name == zero_name:
                        t = torch.zeros_like(t)
                    n["c"] += 1
                    yield name, t

        asyncio.run(self.engine.send_weights(gen()))
        return n["c"]

    def finalize(self):
        self.engine.finalize()


def _build_receivers():
    """Flat mooncake-chain list of receivers across all instances (one device_mesh each)."""
    receivers = []
    for i, inst in enumerate(INSTANCES):
        recv = [
            ReceiverCE.options(
                num_cpus=0,  # SGLang pod raylet is --num-cpus=0; receiver must request 0 CPU
                resources={inst["resource"]: 1},
                runtime_env={"env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}},
            ).remote(
                sglang_url=f"http://127.0.0.1:{inst['port']}", model_path=MODEL, bucket_size=BUCKET,
                tp_rank=r, tp_size=TP, recv_device=RECV_DEVICE, gpu_offset=inst["offset"],
            )
            for r in range(TP)
        ]
        addr, mport = ray.get(recv[0].get_rendezvous.remote())
        ray.get([r.setup_tp_group.remote(addr, mport) for r in recv])
        receivers.extend(recv)
        print(f"  instance {i}: resource={inst['resource']} {inst['host']}:{inst['port']} "
              f"tp={TP} gpu_offset={inst['offset']} -> {TP} receivers", flush=True)
    return receivers


def sync_once(trainer, receivers, zero_name=""):
    """One weight-sync cycle across all instances (same as ExternalCheckpointManager)."""
    n_recv = len(receivers)
    metas = ray.get([trainer.prepare.remote()] + [r.prepare.remote() for r in receivers])
    tk, rk = MooncakeCheckpointEngine.build_topology(1, n_recv, metas)
    refs = [trainer.init_process_group.remote(tk["rank"][0], tk["world_size"][0], tk["metadata"][0])]
    for i, r in enumerate(receivers):
        refs.append(r.init_process_group.remote(rk["rank"][i], rk["world_size"][i], rk["metadata"][i]))
    ray.get(refs)
    res = ray.get(
        [trainer.send.remote(zero_name)] + [r.receive_and_push.remote(chunk_tensors=CHUNK) for r in receivers]
    )
    ray.get([trainer.finalize.remote()] + [r.finalize.remote() for r in receivers])
    return res  # [sent, pushed_r0, pushed_r1, ...]


def generate(inst):
    r = requests.post(
        f"http://{inst['host']}:{inst['port']}/generate",
        json={"text": "The capital of France is", "sampling_params": {"temperature": 0, "max_new_tokens": 16}},
        timeout=120,
    )
    return r.json()["text"]


def show(tag):
    for inst in INSTANCES:
        print(f"  [{tag}] {inst['host']}:{inst['port']} -> {generate(inst)!r}", flush=True)


def main():
    ray.init(address="auto")
    print(f"MODEL={MODEL} TP={TP} recv_device={RECV_DEVICE} bucket={BUCKET >> 20}MB chunk={CHUNK}", flush=True)
    print(f"INSTANCES={INSTANCES}", flush=True)
    trainer = _TrainerSender.remote(MODEL, BUCKET)
    print("building receivers...", flush=True)
    receivers = _build_receivers()

    print("baseline:", flush=True); show("base")
    print(">>> round 1: sync base (unchanged)", flush=True)
    print("   sent/pushed:", sync_once(trainer, receivers, ""), flush=True); show("r1")
    print(f">>> round 2: zero {ZERO_NAME} (expect collapse on ALL)", flush=True)
    print("   sent/pushed:", sync_once(trainer, receivers, ZERO_NAME), flush=True); show("r2")
    print(">>> round 3: restore base (expect recovery on ALL)", flush=True)
    print("   sent/pushed:", sync_once(trainer, receivers, ""), flush=True); show("r3")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
