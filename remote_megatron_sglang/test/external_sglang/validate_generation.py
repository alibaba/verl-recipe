#!/usr/bin/env python3
"""Validate the GENERATION plane of the external-sglang pipeline (real module code).

Creates the module's ExternalSGLangProxyActor, registers it in a real
GlobalRequestLoadBalancer, and drives it exactly like the proxy/agent-loop do
(acquire_server -> server.generate.remote()). Confirms tokens come back from the
external SGLang and decode to sensible text.

Run from a Ray node with /mnt/models available (e.g. the sglang pod):
    python3 validate_generation.py
"""

import asyncio
import os

import ray
from transformers import AutoTokenizer

from verl.workers.rollout.external_sglang.proxy_actor import ExternalSGLangProxyActor
from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer

BASE_URL = os.environ.get("SGLANG_URL", "http://10.8.0.4:30000")
MODEL = os.environ.get("MODEL_PATH", "/mnt/models/Qwen2.5-3B-Instruct")


async def main():
    ray.init(address="auto")
    tok = AutoTokenizer.from_pretrained(MODEL)

    shim = ExternalSGLangProxyActor.remote(base_url=BASE_URL)
    assert ray.get(shim.health.remote()), "external SGLang not healthy"
    lb = GlobalRequestLoadBalancer.remote(servers={BASE_URL: shim})

    prompt = "The capital of France is"
    prompt_ids = tok(prompt)["input_ids"]

    # exactly what LLMServerClient / VLLMRayProvider do:
    server_id, server = await lb.acquire_server.remote(request_id="req-1")
    try:
        out = await server.generate.remote(
            request_id="req-1",
            prompt_ids=prompt_ids,
            sampling_params={"temperature": 0, "max_new_tokens": 16},
        )
    finally:
        lb.release_server.remote(server_id)

    print("server_id      :", server_id)
    print("prompt         :", repr(prompt))
    print("response tokens:", out.token_ids[:20])
    print("decoded        :", repr(tok.decode(out.token_ids)))
    print("stop_reason    :", out.stop_reason)
    print("LB status      :", ray.get(lb.get_status.remote()))


if __name__ == "__main__":
    asyncio.run(main())
