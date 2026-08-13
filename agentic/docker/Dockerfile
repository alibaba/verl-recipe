# Dockerfile for verl + Mooncake fully disaggregated RL training
#
# Architecture:
#   - Training-Rollout disaggregation (separate GPU pools)
#   - SGLang PD disaggregation (prefill + decode on separate GPUs)
#   - Mooncake checkpoint engine (P2P weight sync trainer -> rollout)
#   - Mooncake HiCache KVCache offload (SGLang L3 distributed cache)
#   - Mooncake PD transfer backend (KV cache: prefill -> decode)
#
# Build context: verl-main repo root (needs verl/, setup.py, recipe/)
#   docker build -f recipe/agentic/docker/Dockerfile .
#
# Base: verlai/verl:sgl0512.dev1 (SGLang, PyTorch, CUDA, Ray, verl deps)

FROM verlai/verl:sgl0512.dev1

ARG PIP_NO_CACHE_DIR=1
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# ============================================================
# 1. Install RDMA / InfiniBand runtime dependencies
# ============================================================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        rdma-core \
        ibverbs-providers \
        libibverbs1 \
        librdmacm1 \
        libnuma1 \
        liburing2 \
        libyaml-0-2 \
        libcurl4 \
        perftest \
        iproute2 && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# 2. Install Mooncake Python SDK (transfer engine + store)
#    Used by: checkpoint engine, HiCache L3 backend, PD transfer
# ============================================================
RUN pip install mooncake-transfer-engine nvidia-cuda-runtime-cu12

# Mooncake is built against CUDA 12; link the real CUDA 12 runtime into the cu13 lib dir
# so the existing LD_LIBRARY_PATH finds it without modification
RUN REAL_CU12=$(find /usr/local/lib -path '*/nvidia/cuda_runtime/lib/libcudart.so.12*' -not -type l -print -quit 2>/dev/null) && \
    CU13_DIR=$(find /usr/local/lib -path '*/nvidia/cu13/lib' -type d -print -quit 2>/dev/null) && \
    if [ -n "$REAL_CU12" ] && [ -n "$CU13_DIR" ]; then \
        ln -sf "$REAL_CU12" "$CU13_DIR/libcudart.so.12"; \
        echo "Linked $CU13_DIR/libcudart.so.12 -> $REAL_CU12"; \
    else \
        echo "WARNING: CUDA 12 runtime or cu13 dir not found, skipping symlink"; \
    fi
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
# Note: mooncake import check requires libcuda.so.1 (NVIDIA driver), only available at runtime

# ============================================================
# 2b. Install SGLang Router (PD-aware HTTP proxy for AI gateway)
#     Launched by verl as a Ray actor, not as a K8s sidecar
# ============================================================
RUN pip install sglang-router

# ============================================================
# 2c. Install verl from source (editable mode)
#     Keeps source at /workspace/verl/ so Hydra file:// paths resolve
#     correctly and code changes are picked up without rebuild.
# ============================================================
WORKDIR /workspace
COPY verl /workspace/verl
COPY setup.py pyproject.toml README.md /workspace/
RUN pip install -e /workspace

# ============================================================
# 3. Copy helper scripts
# ============================================================
COPY recipe/agentic/train_grpo_disagg_sglang_pd_hicache.sh /workspace/train_grpo_disagg_sglang_pd_hicache.sh
RUN chmod +x /workspace/train_grpo_disagg_sglang_pd_hicache.sh

# ============================================================
# 4. Harbor — agent execution framework (ACKEnvironment, Trial)
# ============================================================
RUN pip install transformers==5.6.0

RUN git clone https://github.com/alibaba/harbor.git /tmp/harbor && \
    pip install --ignore-installed /tmp/harbor && \
    rm -rf /tmp/harbor && \
    pip install kubernetes

# Override harbor files with our patched versions:
#   ack.py:       thread-safety, PVC volume mount, UV_INDEX_URL env vars
#   trial.py:     agent setup timeout 360s -> 1200s
#   swe_agent.py: miniconda python, cp from PVC, aliyun pip mirror
COPY recipe/agentic/docker/overrides/harbor/environments/ack.py \
     /usr/local/lib/python3.12/dist-packages/harbor/environments/ack.py
COPY recipe/agentic/docker/overrides/harbor/trial/trial.py \
     /usr/local/lib/python3.12/dist-packages/harbor/trial/trial.py
COPY recipe/agentic/docker/overrides/harbor/agents/installed/swe_agent.py \
     /usr/local/lib/python3.12/dist-packages/harbor/agents/installed/swe_agent.py

# ============================================================
# 5. Agentic recipe (entry point, proxy server, agent loop)
#    Copied from the verl-recipe submodule (no git clone needed)
#    recipe/__init__.py is needed so `python -m recipe.agentic.xxx` works.
# ============================================================
COPY recipe/agentic /workspace/recipe/agentic
# Create package marker so `recipe` is importable as a namespace package
RUN touch /workspace/recipe/__init__.py
RUN pip install shortuuid aiohttp

ENV PYTHONPATH="/workspace:${PYTHONPATH}"
ENV LOCAL_TEST=true

COPY recipe/agentic/train_agentic_disagg_sglang_sweagent.sh /workspace/train_agentic_disagg_sglang_sweagent.sh
RUN chmod +x /workspace/train_agentic_disagg_sglang_sweagent.sh

# ============================================================
# 6. Pin shared deps to resolve cross-package conflicts
#    - tokenizers: Harbor pulls 0.23.1, but transformers 5.x needs <=0.23.0
#    - cupy-cuda13x: NCCL checkpoint engine backend (CUDA 13.x)
# ============================================================
RUN pip install "tokenizers>=0.22.0,<=0.23.0" cupy-cuda13x

CMD ["/bin/bash"]
