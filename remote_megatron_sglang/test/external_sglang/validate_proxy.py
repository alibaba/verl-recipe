#!/usr/bin/env python3
"""Validate the PROXY generation path against external SGLang.

The agentic pipeline generates through the proxy (VLLMRayProvider -> LB.acquire_server
-> server.generate.remote()), not the direct LLMServerClient. This drives that exact
path: build ExternalLLMServerManager (LB holds shims -> external SGLang), start the
proxy bound to that LB, and hit its OpenAI endpoint.

Run from the Ray head pod:  cd /workspace && python3 validate_proxy.py
"""
import os

import ray
import requests
from omegaconf import OmegaConf

from recipe.agentic.proxyserver.ray_actor import start_proxy_server
from verl.workers.rollout.external_sglang.server_manager import ExternalLLMServerManager

URL = os.environ.get("SGLANG_URL", "http://10.8.0.5:30000")
MODEL = os.environ.get("MODEL_PATH", "/mnt/models/Qwen2.5-3B-Instruct")


def main():
    ray.init(address="auto")
    config = OmegaConf.create({
        "actor_rollout_ref": {
            "rollout": {"name": "sglang", "external_sglang_endpoints": [URL],
                        "prometheus": {"enable": False}},
            "model": {"path": MODEL},
        }
    })
    mgr = ExternalLLMServerManager.create(config=config)
    proxy_url = start_proxy_server(
        load_balancer=mgr.global_load_balancer, model_path=MODEL,
        host="0.0.0.0", port=0, tool_format="hermes",
    )
    print("proxy_url:", proxy_url, flush=True)

    resp = requests.post(
        f"{proxy_url}/session-1/v1/chat/completions",
        json={"model": MODEL,
              "messages": [{"role": "user", "content": "The capital of France is?"}],
              "temperature": 0, "max_tokens": 24},
        timeout=180,
    )
    print("status:", resp.status_code, flush=True)
    data = resp.json()
    print("content:", repr(data["choices"][0]["message"]["content"]), flush=True)


if __name__ == "__main__":
    main()
