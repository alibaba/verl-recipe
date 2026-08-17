#!/usr/bin/env python3
"""Validate the real ExternalCheckpointManager.update_weights() orchestration.

Exercises the committed manager class end-to-end: it creates the ReceiverCE from
config, builds the mooncake process group, and drives send+recv+finalize. Only
the trainer RayWorkerGroup is a minimal faithful stand-in (single rank-0 mooncake
sender) implementing the two methods the manager calls:
  - execute_checkpoint_engine(...)  (both positional-methods and kwargs forms)
  - update_weights(global_steps, mode)

Verifies base -> embed-zeroed -> restored via generation collapse/restore.

Run from the Ray head pod:  python3 validate_checkpoint_manager.py
"""

import asyncio
import glob
import os

import ray
import requests

from verl.checkpoint_engine.mooncake_checkpoint_engine import MooncakeCheckpointEngine
from verl.workers.config.rollout import CheckpointEngineConfig
from verl.workers.rollout.external_sglang.checkpoint_manager import ExternalCheckpointManager

MODEL = os.environ.get("MODEL_PATH", "/mnt/models/Qwen2.5-3B-Instruct")
SGLANG_IP = os.environ.get("SGLANG_NODE_IP", "10.8.0.4")
SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "30000"))
BUCKET_MB = int(os.environ.get("BUCKET_MB", "2048"))
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))


@ray.remote(num_gpus=1, num_cpus=2)
class _TrainerCEWorker:
    """Mimics a single CheckpointEngineWorker (mooncake rank 0) on the trainer side."""

    def __init__(self, model_path, bucket_size):
        os.environ["RANK"] = "0"
        self.model_path = model_path
        self.zero_name = ""
        self.engine = MooncakeCheckpointEngine(bucket_size=bucket_size, device="cuda", is_master=True)

    # --- checkpoint_engine method proxies (what execute_checkpoint_engine dispatches) ---
    def prepare(self):
        return self.engine.prepare()

    def init_process_group(self, rank, world_size, metadata):
        self.engine.init_process_group(rank=rank, world_size=world_size, metadata=metadata)

    def finalize(self):
        self.engine.finalize()

    def set_zero_name(self, zero_name):
        self.zero_name = zero_name

    # --- what trainer.update_weights dispatches to (send_weights) ---
    def do_update_weights(self, global_steps=None, mode="mooncake"):
        import torch
        from safetensors.torch import load_file

        def gen():
            for f in sorted(glob.glob(os.path.join(self.model_path, "*.safetensors"))):
                sd = load_file(f, device="cpu")
                for name, t in sd.items():
                    t = t.to(device="cuda", dtype=torch.bfloat16)
                    if self.zero_name and name == self.zero_name:
                        t = torch.zeros_like(t)
                    yield name, t

        asyncio.run(self.engine.send_weights(gen(), global_steps=global_steps))


class FakeTrainerWG:
    """Minimal RayWorkerGroup stand-in exposing the two methods the manager calls."""

    def __init__(self, ce_worker):
        self.ce = ce_worker
        self.world_size = 1

    def execute_checkpoint_engine(self, methods=None, **kwargs):
        # positional form: a list of method names, no args (e.g. ["prepare"], ["finalize"])
        if methods is not None:
            return [getattr(self.ce, m).remote() for m in methods]
        # kwargs form: per-rank arg lists + a "method" list
        method = kwargs.pop("method")
        refs = []
        for i, m in enumerate(method):
            call_kwargs = {k: v[i] for k, v in kwargs.items()}
            refs.append(getattr(self.ce, m).remote(**call_kwargs))
        return refs

    def update_weights(self, global_steps=None, mode="mooncake"):
        return [self.ce.do_update_weights.remote(global_steps=global_steps, mode=mode)]


def generate():
    r = requests.post(
        f"http://{SGLANG_IP}:{SGLANG_PORT}/generate",
        json={"text": "The capital of France is", "sampling_params": {"temperature": 0, "max_new_tokens": 16}},
        timeout=60,
    )
    return r.json()["text"]


def main():
    ray.init(address="auto")
    ce = _TrainerCEWorker.remote(MODEL, BUCKET_MB << 20)
    trainer = FakeTrainerWG(ce)

    cfg = CheckpointEngineConfig(
        backend="mooncake",
        update_weights_bucket_megabytes=BUCKET_MB,
        engine_kwargs={
            "external_sglang": {
                "model_path": MODEL,
                "receivers": [
                    {"url": f"http://127.0.0.1:{SGLANG_PORT}", "resource": "sglang_node", "tp_size": TP_SIZE}
                ],
            }
        },
    )
    mgr = ExternalCheckpointManager(config=cfg, trainer=trainer, replicas=[])

    print("baseline    :", repr(generate()), flush=True)
    print(">>> update_weights #1 (base)", flush=True)
    ray.get(ce.set_zero_name.remote(""))
    mgr.update_weights(global_steps=1)
    print("after #1    :", repr(generate()), flush=True)
    print(">>> update_weights #2 (embed zeroed)", flush=True)
    ray.get(ce.set_zero_name.remote("model.embed_tokens.weight"))
    mgr.update_weights(global_steps=2)
    print("after #2    :", repr(generate()), flush=True)
    print(">>> update_weights #3 (restore)", flush=True)
    ray.get(ce.set_zero_name.remote(""))
    mgr.update_weights(global_steps=3)
    print("after #3    :", repr(generate()), flush=True)


if __name__ == "__main__":
    main()
