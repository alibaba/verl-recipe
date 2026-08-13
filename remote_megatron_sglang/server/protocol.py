# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The stable HTTP contract between the verl-side ``MegatronTrainClient`` and
the ``train_server`` running inside the PyTorchJob.

Keeping the endpoint paths and the tensor (de)serialization in one shared module
means the client and server can never drift, and a future standalone-Megatron
server (design Option B) can implement the exact same contract without touching
verl internals.

Wire format for tensor payloads: a verl batch is a ``TensorDict``; we serialize
it with ``safetensors`` (a flat name→tensor map plus a JSON sidecar for the
non-tensor / batch-size metadata). safetensors is already a verl dependency and
is zero-copy on load.
"""

from __future__ import annotations

import io
import json
from typing import Any

# ---------------------------------------------------------------------------- #
# Endpoint paths
# ---------------------------------------------------------------------------- #

HEALTH = "/health"
COMPUTE_LOG_PROB = "/compute_log_prob"  # actor forward, no grad → per-token logprobs
COMPUTE_REF_LOG_PROB = "/compute_ref_log_prob"  # ref-model forward, no grad
UPDATE_ACTOR = "/update_actor"  # forward + backward + optimizer step → metrics
PUSH_WEIGHTS = "/push_weights"  # trigger weight export/broadcast (weight sync)
SAVE_WEIGHTS = "/save_weights"  # export HF weights to a shared store (store transport)
SAVE_CHECKPOINT = "/save_checkpoint"  # persist full train state (model + optimizer)
LOAD_CHECKPOINT = "/load_checkpoint"
INIT_BROADCAST_GROUP = "/init_broadcast_group"  # join the cross-cluster NCCL group
DESTROY_BROADCAST_GROUP = "/destroy_broadcast_group"
INIT_MOONCAKE = "/init_mooncake"
DESTROY_MOONCAKE = "/destroy_mooncake"

# Header carrying the JSON metadata sidecar alongside a safetensors body.
META_HEADER = "x-verl-meta"


# ---------------------------------------------------------------------------- #
# TensorDict <-> bytes (safetensors body + JSON metadata sidecar)
# ---------------------------------------------------------------------------- #


def encode_tensordict(td) -> tuple[bytes, dict[str, Any]]:
    """Serialize a TensorDict to (safetensors_bytes, metadata).

    ``metadata`` carries the batch size and the non-tensor entries so the far
    side can reconstruct an equivalent TensorDict. Imports are local so this
    module stays importable without torch on the pure-client side that only
    needs the endpoint constants.
    """
    import torch
    from safetensors.torch import save as st_save

    tensors: dict[str, torch.Tensor] = {}
    non_tensor: dict[str, Any] = {}
    for key in td.keys():
        val = td[key]
        if isinstance(val, torch.Tensor):
            # safetensors only handles dense strided tensors. verl batches may
            # carry sparse (input_ids/position_ids) or nested (remove-padding)
            # tensors — densify them before serialization.
            if val.is_nested:
                val = val.to_padded_tensor(0)
            elif getattr(val, "layout", torch.strided) != torch.strided or val.is_sparse:
                val = val.to_dense()
            tensors[key] = val.contiguous()
        else:
            non_tensor[key] = val
    batch_size = list(td.batch_size) if hasattr(td, "batch_size") else []
    body = st_save(tensors) if tensors else b""
    meta = {"batch_size": batch_size, "non_tensor": non_tensor, "tensor_keys": list(tensors.keys())}
    return body, meta


def decode_tensordict(body: bytes, meta: dict[str, Any]):
    """Inverse of :func:`encode_tensordict`."""
    import torch  # noqa: F401
    from safetensors.torch import load as st_load
    from tensordict import TensorDict

    tensors = st_load(body) if body else {}
    batch_size = meta.get("batch_size") or []
    td = TensorDict(tensors, batch_size=batch_size)
    for key, val in (meta.get("non_tensor") or {}).items():
        td[key] = val
    return td


def _json_default(o: Any):
    """Coerce non-JSON values (numpy scalars/arrays) found in a verl batch's
    non-tensor fields. These fields (data_source, uids, ...) are metadata not
    consumed by the forward/loss, so a lossy fallback to str is acceptable."""
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dumps_meta(meta: dict[str, Any]) -> str:
    return json.dumps(meta, default=_json_default)


def loads_meta(raw: str) -> dict[str, Any]:
    return json.loads(raw) if raw else {}


def to_buffer(body: bytes) -> io.BytesIO:
    return io.BytesIO(body)


# ---------------------------------------------------------------------------- #
# Single-body framing: [4-byte big-endian meta length][meta json][safetensors]
# Keeps the (potentially large) metadata out of HTTP headers, which have a
# strict max line length.
# ---------------------------------------------------------------------------- #

import struct  # noqa: E402


def frame(body: bytes, meta: dict[str, Any]) -> bytes:
    meta_bytes = dumps_meta(meta).encode("utf-8")
    return struct.pack(">I", len(meta_bytes)) + meta_bytes + body


def unframe(blob: bytes) -> tuple[bytes, dict[str, Any]]:
    (meta_len,) = struct.unpack(">I", blob[:4])
    meta = loads_meta(blob[4 : 4 + meta_len].decode("utf-8"))
    body = blob[4 + meta_len :]
    return body, meta


def dumps_td(td) -> bytes:
    """Serialize a full TensorDict with torch.save.

    Unlike the safetensors path, this faithfully round-trips verl's batch
    structure — nested/packed sequence tensors, sparse tensors, and the
    TensorDict ``batch_size`` — which the compute engine relies on. Uses pickle
    under the hood; acceptable for internal train↔forwarder cluster traffic.
    """
    import torch

    buf = io.BytesIO()
    torch.save(td, buf)
    return buf.getvalue()


def loads_td(blob: bytes):
    import torch

    return torch.load(io.BytesIO(blob), weights_only=False)


def frame_tensordict(td) -> bytes:
    return dumps_td(td)


def unframe_tensordict(blob: bytes):
    return loads_td(blob)
