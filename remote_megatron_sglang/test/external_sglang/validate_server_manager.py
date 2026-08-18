#!/usr/bin/env python3
"""Validate ExternalLLMServerManager via its real create() entry point.

Exercises the exact flow the trainer uses:
  ExternalLLMServerManager.create(config)
    -> _initialize_llm_servers  (build shims from external_sglang_endpoints)
    -> _launch_routers          (no-op)
    -> _init_global_load_balancer
  then acquire a server from the resulting LB and generate.

Run from a Ray node with /mnt/models (e.g. the sglang pod): python3 validate_server_manager.py
"""

import os

import ray
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.workers.rollout.external_sglang.server_manager import ExternalLLMServerManager

URL = os.environ.get("SGLANG_URL", "http://10.8.0.4:30000")
MODEL = os.environ.get("MODEL_PATH", "/mnt/models/Qwen2.5-3B-Instruct")


def main():
    ray.init(address="auto")
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"name": "sglang", "external_sglang_endpoints": [URL], "prometheus": {"enable": False}},
                "model": {"path": MODEL},
            }
        }
    )

    mgr = ExternalLLMServerManager.create(config=config)
    print("get_addresses():", mgr.get_addresses())
    print("get_replicas() :", mgr.get_replicas())

    lb = mgr.global_load_balancer
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok("The capital of France is")["input_ids"]

    sid, server = ray.get(lb.acquire_server.remote(request_id="r1"))
    out = ray.get(
        server.generate.remote(
            request_id="r1",
            prompt_ids=prompt_ids,
            sampling_params={"temperature": 0, "max_new_tokens": 16},
        )
    )
    lb.release_server.remote(sid)
    print("acquired server:", sid)
    print("decoded        :", repr(tok.decode(out.token_ids)))
    print("LB status      :", ray.get(lb.get_status.remote()))


if __name__ == "__main__":
    main()
