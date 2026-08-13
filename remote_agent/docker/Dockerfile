# Dockerfile for verl + Mooncake RL training with the remote_agent recipe
#
# Architecture:
#   - Training-Rollout disaggregation (separate GPU pools)
#   - Mooncake checkpoint engine (P2P weight sync trainer -> rollout)
#   - Mooncake HiCache KVCache offload / PD transfer backend (optional)
#   - remote_agent recipe: RL-train against an external OpenAI-compatible agent;
#     an OpenAI-compatible proxy records the agent's LLM traffic and reconstructs
#     the verl trajectory. Harbor is one pluggable ExternalAgentRunner.
#
# Build context: verl-main repo root (needs verl/, setup.py, recipe/)
#   docker build -f recipe/remote_agent/docker/Dockerfile .
#
# NOTE: the `recipe` submodule must be checked out on a branch that contains
#       `recipe/remote_agent/` (e.g. feat/remote-agent-recipe) for the COPY below.
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
# 3. Harbor — agent execution framework (Trial, DockerEnvironment)
#    Used by the harbor ExternalAgentRunner (local Trial mode).
# ============================================================
RUN pip install transformers==5.6.0

RUN git clone -b release-dev-e5e46809-rollout-server https://github.com/alibaba/harbor.git /tmp/harbor && \
    pip install --ignore-installed /tmp/harbor && \
    rm -rf /tmp/harbor && \
    pip install kubernetes
# NOTE: v1 remote_agent defaults to DockerEnvironment, so the ACK-specific harbor
#       overrides used by the old agentic image are NOT applied here. If you run
#       on ACK or need the trial setup-timeout bump, re-add the corresponding
#       harbor override COPY lines.

# ============================================================
# 4. remote_agent recipe (framework-agnostic core + runners + proxy)
#    Copied from the verl-recipe submodule as a TOP-LEVEL package so
#    `python -m remote_agent.main` works. No `recipe.` namespace needed.
# ============================================================
COPY recipe/remote_agent /workspace/remote_agent
RUN pip install shortuuid aiohttp

ENV PYTHONPATH="/workspace:${PYTHONPATH}"
ENV LOCAL_TEST=true

# Launch helper (Harbor + SWE-agent example)
RUN chmod +x /workspace/remote_agent/scripts/train_harbor_sweagent.sh

# ============================================================
# 5. Pin shared deps to resolve cross-package conflicts
#    - tokenizers: Harbor pulls 0.23.1, but transformers 5.x needs <=0.23.0
#    - cupy-cuda13x: NCCL checkpoint engine backend (CUDA 13.x)
# ============================================================
RUN pip install "tokenizers>=0.22.0,<=0.23.0" cupy-cuda13x

CMD ["/bin/bash"]
