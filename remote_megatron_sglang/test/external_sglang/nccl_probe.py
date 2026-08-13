#!/usr/bin/env python3
"""Minimal cross-node NCCL probe: rank0 on a trainer GPU, rank1 colocated on the
external SGLang pod (sglang_node resource, num_gpus=0 + NOSET). Does one
ray.util.collective allreduce and prints the result. NCCL_DEBUG=INFO is set so
the actor logs show exactly why cross-node comm init succeeds/fails.

Run from the Ray head:  python3 nccl_probe.py
"""
import os
import ray
import ray.util.collective as col

RESOURCE = os.environ.get("SGLANG_RESOURCE", "sglang_node")

# NCCL RoCEv2/IPv6 env matching the training worker (+ INFO logging).
_ENV = {"env_vars": {
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,NET,ENV",
    "NCCL_IB_ADDR_FAMILY": "AF_INET6",
    "NCCL_IB_ADDR_RANGE": os.environ.get("NCCL_IB_ADDR_RANGE", "2001:db8:80f:e000::/60"),
    "NCCL_NET_PLUGIN": "none",
    "GLOO_SOCKET_IFNAME": "eth0",
    "NCCL_SOCKET_IFNAME": "eth0",
}}
# Optional overrides: set NCCL_IB_HCA / NCCL_IB_GID_INDEX via the driver env.
for k in ("NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_IB_DISABLE"):
    if k in os.environ:
        _ENV["env_vars"][k] = os.environ[k]


@ray.remote(num_gpus=1, num_cpus=1)
class Rank0:
    def setup(self):
        import torch
        col.init_collective_group(2, 0, "nccl", "probe")
        self.t = torch.ones(1024, device="cuda")
        return "rank0 group init"

    def run(self):
        import torch
        col.allreduce(self.t, group_name="probe")
        torch.cuda.synchronize()
        return float(self.t[0].item())


@ray.remote(num_gpus=0, num_cpus=0, resources={RESOURCE: 1})
class Rank1:
    def setup(self):
        import torch
        torch.cuda.set_device(0)
        col.init_collective_group(2, 1, "nccl", "probe")
        self.t = torch.ones(1024, device="cuda")
        return "rank1 group init"

    def run(self):
        import torch
        col.allreduce(self.t, group_name="probe")
        torch.cuda.synchronize()
        return float(self.t[0].item())


def main():
    ray.init(address="auto")
    r0 = Rank0.options(runtime_env=_ENV).remote()
    r1 = Rank1.options(runtime_env=_ENV).remote()
    print("setup:", ray.get([r0.setup.remote(), r1.setup.remote()]), flush=True)
    print("running allreduce (expect 2.0 on both)...", flush=True)
    try:
        res = ray.get([r0.run.remote(), r1.run.remote()])
        print("RESULT:", res, "-> NCCL CROSS-NODE OK" if res == [2.0, 2.0] else "unexpected", flush=True)
    except Exception as e:
        print("NCCL PROBE FAILED:", repr(e), flush=True)


if __name__ == "__main__":
    main()
