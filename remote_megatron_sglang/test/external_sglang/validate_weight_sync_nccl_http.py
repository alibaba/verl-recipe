#!/usr/bin/env python3
"""Validate NCCL-over-HTTP weight sync to an EXTERNAL (out-of-Ray) SGLang.

This is the decoupled NCCL path (NOT backend=nccl + ReceiverCE): trainer rank 0
forms a torch.distributed NCCL group DIRECTLY with the external SGLang's TP
workers via SGLang's HTTP API, then broadcasts full HF weights. No colocated
receiver, no CUDA IPC, no Ray membership for SGLang.

  rank0 (this actor, on a trainer GPU)  --NCCL broadcast-->  SGLang TP workers
     POST /init_weights_update_group        (SGLang joins the group as ranks 1..tp)
     torch.distributed.broadcast(src=0)  ×  POST /update_weights_from_distributed
     POST /flush_cache

Logic mirrors recipe/remote_megatron_sglang/checkpoint_engine.py (ExternalSGLangNCCLEngine),
inlined so the Ray actor needs no recipe import on the worker.

Correctness: base -> zero embed (generation collapses) -> restore.

Env: SGLANG_ENDPOINT=http://<pod-ip>:30000  [MODEL_PATH, ZERO_NAME]
Run from the Ray head:  python3 validate_weight_sync_nccl_http.py
"""

import glob
import os
import threading

import ray
import requests

MODEL = os.environ.get("MODEL_PATH", "/mnt/models/Qwen2.5-3B-Instruct")
ENDPOINT = os.environ["SGLANG_ENDPOINT"]
ZERO_NAME = os.environ.get("ZERO_NAME", "model.embed_tokens.weight")
GROUP = "verl_ext_nccl"
CHUNK = int(os.environ.get("CHUNK_TENSORS", "64"))

_NCCL_RT = {
    "env_vars": {
        "NCCL_IB_ADDR_FAMILY": "AF_INET6",
        "NCCL_IB_ADDR_RANGE": os.environ.get("NCCL_IB_ADDR_RANGE", "2001:db8:80f:e000::/60"),
        "NCCL_NET_PLUGIN": "none",
        "NCCL_SOCKET_IFNAME": "eth0",
        "GLOO_SOCKET_IFNAME": "eth0",
    }
}


@ray.remote(num_gpus=1, num_cpus=4)
class Rank0:
    def __init__(self, endpoint, model_path):
        self.endpoint = endpoint.rstrip("/")
        self.model_path = model_path

    def preload(self):
        import torch
        from safetensors.torch import load_file

        self.weights = []
        total = 0
        for f in sorted(glob.glob(os.path.join(self.model_path, "*.safetensors"))):
            sd = load_file(f, device="cpu")
            for n, t in sd.items():
                t = t.to("cuda", dtype=torch.bfloat16)
                self.weights.append((n, t))
                total += t.nbytes
        self._gib = total / (1024**3)
        return {"n": len(self.weights), "GiB": self._gib}

    def _tp_size(self):
        info = requests.get(f"{self.endpoint}/get_server_info", timeout=30).json()
        for k in ("tp_size", "tensor_parallel_size"):
            if k in info:
                return int(info[k])
        return int(info.get("server_args", {}).get("tp_size", 2))

    def sync(self, zero_name=""):
        import time

        import torch
        from sglang.srt.utils import init_custom_process_group

        from verl.utils.net_utils import get_free_port

        t_all = time.time()
        master_addr = ray.util.get_node_ip_address().strip("[]")
        master_port = get_free_port(master_addr)[0]
        tp = self._tp_size()
        world_size = 1 + tp

        def _join():
            requests.post(
                f"{self.endpoint}/init_weights_update_group",
                json={
                    "master_address": master_addr,
                    "master_port": master_port,
                    "rank_offset": 1,
                    "world_size": world_size,
                    "group_name": GROUP,
                    "backend": "nccl",
                },
                timeout=600,
            )

        th = threading.Thread(target=_join, daemon=True)
        th.start()
        torch.cuda.set_device(0)
        t0 = time.time()
        group = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=0,
            group_name=GROUP,
        )
        th.join()
        group_s = time.time() - t0

        def flush(batch):
            if not batch:
                return
            names = [n for n, _ in batch]
            dtypes = ["bfloat16"] * len(batch)
            shapes = [list(t.shape) for _, t in batch]

            def _post():
                requests.post(
                    f"{self.endpoint}/update_weights_from_distributed",
                    json={
                        "names": names,
                        "dtypes": dtypes,
                        "shapes": shapes,
                        "group_name": GROUP,
                        "flush_cache": False,
                    },
                    timeout=600,
                )

            pth = threading.Thread(target=_post, daemon=True)
            pth.start()
            for _, t in batch:
                torch.distributed.broadcast(t, src=0, group=group)
            pth.join()

        t1 = time.time()
        batch = []
        for n, t in self.weights:
            tt = torch.zeros_like(t) if (zero_name and n == zero_name) else t
            batch.append((n, tt))
            if len(batch) >= CHUNK:
                flush(batch)
                batch = []
        flush(batch)
        bcast_s = time.time() - t1

        t2 = time.time()
        requests.post(f"{self.endpoint}/flush_cache", json={}, timeout=60)
        flush_s = time.time() - t2
        t3 = time.time()
        torch.distributed.destroy_process_group(group)
        try:
            requests.post(f"{self.endpoint}/destroy_weights_update_group", json={"group_name": GROUP}, timeout=60)
        except Exception:
            pass
        destroy_s = time.time() - t3
        return {
            "world_size": world_size,
            "group_s": group_s,
            "bcast_s": bcast_s,
            "flush_s": flush_s,
            "destroy_s": destroy_s,
            "total_s": time.time() - t_all,
            "GiB": self._gib,
        }


def generate(ep):
    r = requests.post(
        f"{ep}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 16},
        },
        timeout=120,
    )
    return r.json()["text"]


def main():
    ray.init(address="auto")
    print(f"MODEL={MODEL} ENDPOINT={ENDPOINT} zero_name={ZERO_NAME}", flush=True)
    a = Rank0.options(runtime_env=_NCCL_RT).remote(ENDPOINT, MODEL)
    print("preload:", ray.get(a.preload.remote()), flush=True)

    # Benchmark mode: N unchanged syncs, warmup (round 0) excluded.
    if os.environ.get("BENCH"):
        import statistics

        rounds = int(os.environ.get("ROUNDS", "6"))
        rows = []
        for i in range(rounds):
            r = ray.get(a.sync.remote(""))
            tag = "warmup" if i == 0 else "measure"
            print(
                f"  round {i} [{tag}]: total={r['total_s']:.3f}s "
                f"(group={r['group_s']:.3f}s bcast={r['bcast_s']:.3f}s "
                f"flush={r['flush_s']:.3f}s destroy={r['destroy_s']:.3f}s) "
                f"bw={r['GiB'] / r['bcast_s']:.1f} GiB/s",
                flush=True,
            )
            rows.append(r)
        m = rows[1:]
        gib = m[0]["GiB"]

        def mean(k):
            return statistics.mean(x[k] for x in m)

        print(
            f"\nSUMMARY (n={len(m)}, {gib:.2f} GiB, world_size={m[0]['world_size']}): "
            f"total={mean('total_s'):.3f}s = group {mean('group_s'):.3f} + "
            f"bcast {mean('bcast_s'):.3f} + flush {mean('flush_s'):.3f} + destroy {mean('destroy_s'):.3f}  |  "
            f"bcast_bw={gib / mean('bcast_s'):.1f} GiB/s  "
            f"total_range=[{min(x['total_s'] for x in m):.3f},{max(x['total_s'] for x in m):.3f}]",
            flush=True,
        )
        print("DONE", flush=True)
        return

    print("baseline:", repr(generate(ENDPOINT)), flush=True)
    print(">>> sync base (unchanged):", ray.get(a.sync.remote("")), flush=True)
    print("after r1:", repr(generate(ENDPOINT)), flush=True)
    print(f">>> sync zero {ZERO_NAME} (expect collapse):", ray.get(a.sync.remote(ZERO_NAME)), flush=True)
    print("after zero:", repr(generate(ENDPOINT)), flush=True)
    print(">>> sync restore (expect recovery):", ray.get(a.sync.remote("")), flush=True)
    print("after restore:", repr(generate(ENDPOINT)), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
