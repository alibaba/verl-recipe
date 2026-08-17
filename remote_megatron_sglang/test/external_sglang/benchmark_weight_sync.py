#!/usr/bin/env python3
"""Benchmark weight-sync time: NCCL vs Mooncake(GPU) vs external-Mooncake(host).

Runs the SAME stand-in-trainer -> colocated-receiver -> CUDA-IPC-push pipeline
that ExternalCheckpointManager.update_weights() uses, for three transports,
against ONE external SGLang instance (default Qwen3.6-27B, TP=2):

  * nccl               : NCCLCheckpointEngine  (collective broadcast, GPU buffers)
  * mooncake-gpu       : MooncakeCheckpointEngine device="cuda" (P2P chain, GPU landing buffer)
  * external-mooncake  : MooncakeCheckpointEngine device="cpu"  (P2P chain, HOST landing buffer + H2D stage)

All three finish identically with a CUDA-IPC push (sgl_update_weights ->
/update_weights_from_tensor), so the delta is purely the trainer->receiver
transport (+ the host->GPU staging that the external path adds).

Weights are preloaded ONCE onto the sender GPU (bf16) so per-round timing
measures transport, not disk I/O. Each variant runs ROUNDS rounds; round 0 is a
warm-up (process-group / RDMA handshake / NCCL group build) and is excluded from
the reported statistics.

Metrics per round:
  transport_s : sender-reported pure transfer time (RDMA / NCCL broadcast)
  sync_s      : driver wall-clock of send + receive_and_push (transport + stage + IPC push)
  setup_s     : prepare + build_topology + init_process_group
  finalize_s  : finalize

Run from the Ray head pod:  python3 benchmark_weight_sync.py
"""

import asyncio
import glob
import os
import statistics
import time

import ray

MODEL = os.environ.get("MODEL_PATH", "/mnt/models/Qwen3.6-27B")
TP = int(os.environ.get("SGLANG_TP", "2"))
BUCKET = int(os.environ.get("BUCKET_MB", "3072")) << 20
CHUNK = int(os.environ.get("CHUNK_TENSORS", "16"))
PORT = int(os.environ.get("SGLANG_PORT", "30000"))
RESOURCE = os.environ.get("SGLANG_RESOURCE", "sglang_node")
ROUNDS = int(os.environ.get("ROUNDS", "4"))  # round 0 = warmup, excluded
# Which variants to run (comma-sep): nccl,mooncake-gpu,external-mooncake
VARIANTS = os.environ.get("VARIANTS", "external-mooncake,mooncake-gpu,nccl").split(",")
# Unique ray.util.collective group name per run — NEVER "default", which
# accumulates stale detached NCCLUniqueIDStore actors from prior/failed runs and
# makes the rendezvous hang before NCCL even initializes.
NCCL_GROUP = os.environ.get("NCCL_GROUP", f"wsync{os.getpid()}")

_RT = {"env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}}
# NCCL RoCEv2/IPv6 env for cross-node comm (trainer node <-> sglang pod). Must
# match the training worker's pod env; the sglang pod lacks it by default. NCCL
# auto-selects the correct GID PER NODE from NCCL_IB_ADDR_RANGE — do NOT force
# NCCL_IB_GID_INDEX (the trainer uses GID 7 but the sglang pod uses GID 11, so a
# forced index breaks one side and the rendezvous hangs).
_NCCL_RT = {
    "env_vars": {
        "NCCL_IB_ADDR_FAMILY": "AF_INET6",
        "NCCL_IB_ADDR_RANGE": os.environ.get("NCCL_IB_ADDR_RANGE", "2001:db8:80f:e000::/60"),
        "NCCL_NET_PLUGIN": "none",
        "NCCL_SOCKET_IFNAME": "eth0",
        "GLOO_SOCKET_IFNAME": "eth0",
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
    }
}


# ---------------------------------------------------------------------------
# Senders (rank 0). Preload full weights to the sender GPU once, stream from GPU.
# ---------------------------------------------------------------------------
class _SenderBase:
    def _preload(self):
        import torch
        from safetensors.torch import load_file

        self._weights = []  # list[(name, cuda bf16 tensor)]
        total = 0
        for f in sorted(glob.glob(os.path.join(self.model_path, "*.safetensors"))):
            sd = load_file(f, device="cpu")
            for name, t in sd.items():
                t = t.to(device="cuda", dtype=torch.bfloat16)
                self._weights.append((name, t))
                total += t.nbytes
        self._total_bytes = total
        return {"n_tensors": len(self._weights), "total_bytes": total}

    def _gen(self):
        yield from self._weights


@ray.remote(num_gpus=1, num_cpus=4)
class MooncakeSender(_SenderBase):
    def __init__(self, model_path, bucket_size):
        os.environ["RANK"] = "0"
        self.model_path = model_path
        from verl.checkpoint_engine.mooncake_checkpoint_engine import MooncakeCheckpointEngine

        self.engine = MooncakeCheckpointEngine(bucket_size=bucket_size, device="cuda", is_master=True)

    def preload(self):
        return self._preload()

    def prepare(self):
        return self.engine.prepare()

    def init_process_group(self, rank, world_size, metadata):
        self.engine.init_process_group(rank=rank, world_size=world_size, metadata=metadata)

    def send(self):
        t0 = time.time()
        asyncio.run(self.engine.send_weights(self._gen()))
        return {"transport_s": time.time() - t0, "total_bytes": self._total_bytes}

    def finalize(self):
        self.engine.finalize()


@ray.remote(num_gpus=1, num_cpus=4)
class NcclSender(_SenderBase):
    def __init__(self, model_path, bucket_size, group_name="default"):
        os.environ["RANK"] = "0"
        self.model_path = model_path
        from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine

        self.engine = NCCLCheckpointEngine(
            bucket_size=bucket_size, is_master=True, rebuild_group=False, group_name=group_name
        )

    def preload(self):
        return self._preload()

    def prepare(self):
        return self.engine.prepare()

    def init_process_group(self, rank, world_size, master_metadata):
        self.engine.init_process_group(rank=rank, world_size=world_size, master_metadata=master_metadata)

    def send(self):
        t0 = time.time()
        asyncio.run(self.engine.send_weights(self._gen()))
        return {"transport_s": time.time() - t0, "total_bytes": self._total_bytes}

    def finalize(self):
        self.engine.finalize()


# ---------------------------------------------------------------------------
# NCCL receiver (mooncake receiver is the real ReceiverCE, imported below).
# Mirrors ReceiverCE.receive_and_push but with NCCLCheckpointEngine (GPU bufs).
# ---------------------------------------------------------------------------
@ray.remote(num_gpus=0, num_cpus=0)
class NcclReceiverCE:
    def __init__(self, sglang_url, model_path, bucket_size, tp_rank=0, tp_size=1, gpu_offset=0, group_name="default"):
        device_id = gpu_offset + tp_rank
        os.environ["RANK"] = str(device_id)
        import torch

        from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine
        from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter

        self._tp_rank = tp_rank
        self._tp_size = tp_size
        torch.cuda.set_device(device_id)
        self._engine = NCCLCheckpointEngine(
            bucket_size=bucket_size, is_master=False, rebuild_group=False, group_name=group_name
        )
        self._device_mesh = None
        host, _, port_s = sglang_url.replace("http://", "").rstrip("/").partition(":")
        self._server = AsyncHttpServerAdapter(
            model_path=model_path,
            host=host,
            port=int(port_s or 30000),
            launch_server=False,
            trust_remote_code=True,
        )

    def get_rendezvous(self):
        from verl.utils.net_utils import get_free_port

        ip = ray.util.get_node_ip_address().strip("[]")
        return ip, get_free_port(ip)[0]

    def setup_tp_group(self, master_addr, master_port):
        # GLOO (not NCCL) device-mesh: the transport already uses a ray.util.collective
        # NCCL group in THIS process; a second NCCL context (an nccl device-mesh)
        # deadlocks the rendezvous. Stock verl's CE worker likewise keeps its global
        # group on gloo. The mesh is only used by sgl_update_weights to all_gather
        # the per-TP-rank CUDA-IPC handles (small CPU objects) — gloo is sufficient.
        import torch
        from torch.distributed.device_mesh import init_device_mesh

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=self._tp_size,
                rank=self._tp_rank,
            )
        self._device_mesh = init_device_mesh("cpu", (self._tp_size,), mesh_dim_names=("infer_tp",))

    def prepare(self):
        return self._engine.prepare()

    def init_process_group(self, rank, world_size, master_metadata):
        self._engine.init_process_group(rank=rank, world_size=world_size, master_metadata=master_metadata)

    def receive_and_push(self, chunk_tensors=16):
        from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights

        from verl.utils.device import get_torch_device

        async def _run():
            batch, total = [], 0

            async def flush():
                nonlocal batch, total
                if not batch:
                    return
                get_torch_device().synchronize()
                await sgl_update_weights(
                    engine=self._server,
                    params_batch=batch,
                    device_mesh_key="infer_tp",
                    device_mesh=self._device_mesh,
                )
                total += len(batch)
                batch = []

            async for name, tensor in self._engine.receive_weights():
                batch.append((name, tensor.to("cuda", non_blocking=True, copy=True)))
                if len(batch) >= chunk_tensors:
                    await flush()
            await flush()
            if self._tp_rank == 0:
                await self._server.flush_cache()
            return total

        return asyncio.run(_run())

    def finalize(self):
        self._engine.finalize()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _build_mooncake_receivers(recv_device):
    from verl.workers.rollout.external_sglang.checkpoint_manager import ReceiverCE

    # expandable_segments only for the HOST-memory receiver (recv_device="cpu"),
    # where it eases fragmentation of the transient GPU staging clones and the
    # RDMA buffer is host memory (not ibv-registered on GPU). For recv_device=
    # "cuda" the RDMA landing buffer IS on GPU and must stay in the default pool
    # (expandable/cuMemMap memory fails ibv_reg_mr -> "Bad address").
    # num_cpus=0: the SGLang pod's raylet is started with --num-cpus=0 (deny-by-
    # default isolation), so receivers must request 0 CPU to be placeable there.
    opts = {"num_cpus": 0, "resources": {RESOURCE: 1}}
    if recv_device == "cpu":
        opts["runtime_env"] = _RT
    recv = [
        ReceiverCE.options(**opts).remote(
            sglang_url=f"http://127.0.0.1:{PORT}",
            model_path=MODEL,
            bucket_size=BUCKET,
            tp_rank=r,
            tp_size=TP,
            recv_device=recv_device,
            gpu_offset=0,
        )
        for r in range(TP)
    ]
    addr, mport = ray.get(recv[0].get_rendezvous.remote())
    ray.get([r.setup_tp_group.remote(addr, mport) for r in recv])
    return recv


def _build_nccl_receivers():
    # No expandable_segments: NCCL's GPU send/recv buffers stay in the default
    # pool, and the staged IPC clones use legacy cudaIpcGetMemHandle (which does
    # not work on cuMemMap/expandable memory).
    recv = [
        NcclReceiverCE.options(resources={RESOURCE: 1}, runtime_env=_NCCL_RT).remote(
            sglang_url=f"http://127.0.0.1:{PORT}",
            model_path=MODEL,
            bucket_size=BUCKET,
            tp_rank=r,
            tp_size=TP,
            gpu_offset=0,
            group_name=NCCL_GROUP,
        )
        for r in range(TP)
    ]
    addr, mport = ray.get(recv[0].get_rendezvous.remote())
    ray.get([r.setup_tp_group.remote(addr, mport) for r in recv])
    return recv


def _sync_round(sender, receivers, backend):
    """One weight-sync cycle, mirroring ExternalCheckpointManager.update_weights."""
    from verl.checkpoint_engine.mooncake_checkpoint_engine import MooncakeCheckpointEngine
    from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine

    engine_cls = NCCLCheckpointEngine if backend == "nccl" else MooncakeCheckpointEngine
    meta_key = "master_metadata" if backend == "nccl" else "metadata"
    n_recv = len(receivers)

    t = time.time()
    metas = ray.get([sender.prepare.remote()] + [r.prepare.remote() for r in receivers])
    tk, rk = engine_cls.build_topology(1, n_recv, metas)
    refs = [sender.init_process_group.remote(tk["rank"][0], tk["world_size"][0], tk[meta_key][0])]
    for i, r in enumerate(receivers):
        refs.append(r.init_process_group.remote(rk["rank"][i], rk["world_size"][i], rk[meta_key][i]))
    ray.get(refs)
    setup_s = time.time() - t

    t = time.time()
    res = ray.get([sender.send.remote()] + [r.receive_and_push.remote(chunk_tensors=CHUNK) for r in receivers])
    sync_s = time.time() - t
    send_info = res[0]

    t = time.time()
    ray.get([sender.finalize.remote()] + [r.finalize.remote() for r in receivers])
    finalize_s = time.time() - t

    return {
        "setup_s": setup_s,
        "sync_s": sync_s,
        "finalize_s": finalize_s,
        "transport_s": send_info["transport_s"],
        "total_bytes": send_info["total_bytes"],
    }


def _run_variant(name, sender, receivers, backend):
    print(f"\n===== variant: {name} (backend={backend}) =====", flush=True)
    rows = []
    for i in range(ROUNDS):
        r = _sync_round(sender, receivers, backend)
        tag = "warmup" if i == 0 else "measure"
        gbps = r["total_bytes"] / r["transport_s"] / (1024**3)
        print(
            f"  round {i} [{tag}]: transport={r['transport_s']:.2f}s ({gbps:.2f} GB/s)  "
            f"sync={r['sync_s']:.2f}s  setup={r['setup_s']:.2f}s  finalize={r['finalize_s']:.2f}s",
            flush=True,
        )
        rows.append(r)
    return name, rows[1:]  # drop warmup


def _summ(name, rows):
    def stat(k):
        vals = [r[k] for r in rows]
        return statistics.mean(vals), (min(vals), max(vals))

    tb = rows[0]["total_bytes"]
    tr_mean, tr_rng = stat("transport_s")
    sy_mean, sy_rng = stat("sync_s")
    gbps = tb / tr_mean / (1024**3)
    return {
        "variant": name,
        "n": len(rows),
        "model_GB": tb / (1024**3),
        "transport_s_mean": tr_mean,
        "transport_s_range": tr_rng,
        "GB_s": gbps,
        "sync_s_mean": sy_mean,
        "sync_s_range": sy_rng,
        "setup_s_mean": stat("setup_s")[0],
        "finalize_s_mean": stat("finalize_s")[0],
    }


def main():
    ray.init(address="auto")
    print(
        f"MODEL={MODEL} TP={TP} bucket={BUCKET >> 20}MB chunk={CHUNK} rounds={ROUNDS} variants={VARIANTS}", flush=True
    )

    summaries = []

    mc_variants = [v for v in VARIANTS if v in ("external-mooncake", "mooncake-gpu")]
    if mc_variants:
        print("building mooncake sender + preloading weights to GPU...", flush=True)
        # No expandable_segments on the sender: its 6GB GPU buffer is RDMA-
        # registered (ibv_reg_mr), and cuMemMap/expandable-segment memory is not
        # registerable -> "Bad address". (Host-memory receiver keeps it; see below.)
        mc_sender = MooncakeSender.remote(MODEL, BUCKET)
        info = ray.get(mc_sender.preload.remote())
        print(f"  preloaded {info['n_tensors']} tensors, {info['total_bytes'] / (1024**3):.2f} GB", flush=True)
        for v in mc_variants:
            recv_device = "cpu" if v == "external-mooncake" else "cuda"
            receivers = _build_mooncake_receivers(recv_device)
            name, rows = _run_variant(v, mc_sender, receivers, "mooncake")
            summaries.append(_summ(name, rows))
            for r in receivers:
                ray.kill(r)
            time.sleep(3)
        ray.kill(mc_sender)
        time.sleep(3)

    if "nccl" in VARIANTS:
        print("building nccl sender + preloading weights to GPU...", flush=True)
        nccl_sender = NcclSender.options(runtime_env=_NCCL_RT).remote(MODEL, BUCKET, NCCL_GROUP)
        info = ray.get(nccl_sender.preload.remote())
        print(f"  preloaded {info['n_tensors']} tensors, {info['total_bytes'] / (1024**3):.2f} GB", flush=True)
        receivers = _build_nccl_receivers()
        name, rows = _run_variant("nccl", nccl_sender, receivers, "nccl")
        summaries.append(_summ(name, rows))
        for r in receivers:
            ray.kill(r)
        ray.kill(nccl_sender)

    print("\n\n================ SUMMARY ================", flush=True)
    hdr = f"{'variant':<20} {'model_GB':>8} {'transport_s':>12} {'GB/s':>7} {'sync_s':>9} {'setup_s':>8} {'final_s':>8}"
    print(hdr, flush=True)
    for s in summaries:
        print(
            f"{s['variant']:<20} {s['model_GB']:>8.1f} {s['transport_s_mean']:>12.2f} "
            f"{s['GB_s']:>7.2f} {s['sync_s_mean']:>9.2f} {s['setup_s_mean']:>8.2f} "
            f"{s['finalize_s_mean']:>8.2f}",
            flush=True,
        )
    print("(transport_s/sync_s = mean over measured rounds, warmup excluded)", flush=True)


if __name__ == "__main__":
    main()
