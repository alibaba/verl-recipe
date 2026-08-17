#!/usr/bin/env python3
"""Validate HOST-MEMORY Mooncake weight-sync to PD-DISAGGREGATED external SGLang.

In PD disaggregation each prefill and each decode worker is a separate SGLang
engine holding a FULL copy of the weights, so weight sync must reach EVERY
worker. This mirrors ExternalCheckpointManager: one flat Mooncake P2P chain
(trainer rank0 -> all worker receivers), each worker gets a colocated ReceiverCE
(recv_device="cpu": RDMA in host RAM, staged host->GPU for the CUDA-IPC push).

Push targets = the workers (direct HTTP, CUDA IPC). Generation is checked through
the PD ROUTERS (prefill -> KV transfer -> decode happens inside SGLang).

Env:
  WORKERS  "port:gpu_offset,..."   weight-push targets (all 127.0.0.1, one pod).
           e.g. "31000:0,31001:1,31002:2,32000:3,32001:4,32002:5"
  ROUTERS  "host:port,..."         PD router endpoints for the generation check.
           e.g. "10.8.0.5:40000,10.8.0.5:40001"
  RESOURCE Ray custom resource pinning receivers to the pod (default sglang_node).

Run from the Ray head pod:  python3 validate_weight_sync_pd.py
"""

import asyncio
import glob
import os

import ray
import requests

from verl.checkpoint_engine.mooncake_checkpoint_engine import MooncakeCheckpointEngine
from verl.workers.rollout.external_sglang.checkpoint_manager import ReceiverCE

MODEL = os.environ.get("MODEL_PATH", "/mnt/models/Qwen2.5-3B-Instruct")
RESOURCE = os.environ.get("RESOURCE", "sglang_node")
RECV_DEVICE = os.environ.get("RECV_DEVICE", "cpu")
BUCKET = int(os.environ.get("BUCKET_MB", "2048")) << 20
CHUNK = int(os.environ.get("CHUNK_TENSORS", "16"))
ZERO_NAME = os.environ.get("ZERO_NAME", "model.embed_tokens.weight")

_DEFAULT_WORKERS = "31000:0,31001:1,31002:2,32000:3,32001:4,32002:5"


def _parse_workers():
    out = []
    for spec in os.environ.get("WORKERS", _DEFAULT_WORKERS).split(","):
        port, offset = spec.split(":")
        out.append({"port": int(port), "offset": int(offset)})
    return out


def _parse_routers():
    return [r for r in os.environ.get("ROUTERS", "").split(",") if r]


WORKERS = _parse_workers()
ROUTERS = _parse_routers()


@ray.remote(num_gpus=1, num_cpus=4)
class _TrainerSender:
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
    """One ReceiverCE per PD worker (tp=1), flat mooncake-chain order."""
    receivers = []
    for i, w in enumerate(WORKERS):
        r = ReceiverCE.options(
            num_cpus=0,  # SGLang pod raylet is --num-cpus=0; receiver must request 0 CPU
            resources={RESOURCE: 1},
            runtime_env={"env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}},
        ).remote(
            sglang_url=f"http://127.0.0.1:{w['port']}",
            model_path=MODEL,
            bucket_size=BUCKET,
            tp_rank=0,
            tp_size=1,
            recv_device=RECV_DEVICE,
            gpu_offset=w["offset"],
        )
        addr, mport = ray.get(r.get_rendezvous.remote())
        ray.get(r.setup_tp_group.remote(addr, mport))
        receivers.append(r)
        print(f"  worker {i}: :{w['port']} gpu_offset={w['offset']} -> receiver", flush=True)
    return receivers


def sync_once(trainer, receivers, zero_name=""):
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
    return res


def generate(router):
    r = requests.post(
        f"http://{router}/generate",
        json={"text": "The capital of France is", "sampling_params": {"temperature": 0, "max_new_tokens": 16}},
        timeout=120,
    )
    return r.json()["text"]


def show(tag):
    for router in ROUTERS:
        print(f"  [{tag}] router {router} -> {generate(router)!r}", flush=True)


def main():
    ray.init(address="auto")
    print(f"MODEL={MODEL} recv_device={RECV_DEVICE} bucket={BUCKET >> 20}MB chunk={CHUNK}", flush=True)
    print(f"WORKERS={WORKERS}", flush=True)
    print(f"ROUTERS={ROUTERS}", flush=True)
    trainer = _TrainerSender.remote(MODEL, BUCKET)
    print("building receivers (one per PD worker)...", flush=True)
    receivers = _build_receivers()

    print("baseline (via PD routers):", flush=True)
    show("base")
    print(">>> round 1: sync base (unchanged)", flush=True)
    print("   sent/pushed:", sync_once(trainer, receivers, ""), flush=True)
    show("r1")
    print(f">>> round 2: zero {ZERO_NAME} (expect collapse on ALL engines)", flush=True)
    print("   sent/pushed:", sync_once(trainer, receivers, ZERO_NAME), flush=True)
    show("r2")
    print(">>> round 3: restore base (expect recovery on ALL engines)", flush=True)
    print("   sent/pushed:", sync_once(trainer, receivers, ""), flush=True)
    show("r3")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
