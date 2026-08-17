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
"""Megatron RL training server — runs INSIDE the Kubeflow PyTorchJob.

Reuses verl's own :class:`~verl.workers.engine_workers.TrainingWorker` (the same
unit ``ActorRolloutRefWorker`` builds for its in-process actor) so the mcore
model build, ``mbridge``/``AutoBridge`` HF<->Megatron conversion,
``get_per_tensor_param()`` weight extraction, distributed optimizer, pipeline
forward/backward, and the PPO/GRPO loss (``verl.trainer.ppo.core_algos.ppo_loss``)
are all reused verbatim — the RL math is identical to in-process verl.

`TrainingWorker.__init__` calls `initialize_global_process_group_ray()`, which —
despite the name — just runs `torch.distributed.init_process_group` from
`RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT`. Kubeflow PyTorchJob injects
exactly those, so the worker runs unmodified in a plain PyTorchJob (no Ray).

Process model ("rank-0 serves, others follow" collective pattern):

* All ranks build the same TrainingWorker and join the same process group.
* Rank 0 runs the aiohttp HTTP server (the contract in ``server/protocol.py``).
  On each request it broadcasts a command (+ batch) to all ranks; every rank
  runs the engine op collectively; rank 0 returns the result over HTTP.
* Ranks > 0 sit in :meth:`_follower_loop` waiting for the next broadcast command.

Run (inside the pod):
    python -m recipe.remote_megatron_sglang.server.train_server \
        --config /config/ppo_trainer_megatron.yaml --port 8000
where --config is a full verl PPO config (only ``actor_rollout_ref`` is read).
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
from functools import partial

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# NCCL-over-HTTP weight sync helpers (ported from the former recipe/external_sglang,
# now consolidated in this recipe).
# The train_server (Megatron rank 0) forms one NCCL group spanning itself + the
# external SGLang TP workers, then broadcasts HF-layout weights. SGLang joins
# and receives via its HTTP weight-update API. Intra-node NCCL runs over the pod
# network (TCP) — no RDMA required.
# --------------------------------------------------------------------------- #


def _post(endpoint: str, path: str, payload: dict, timeout: float = 600.0) -> dict:
    import requests

    resp = requests.post(f"{endpoint.rstrip('/')}/{path.lstrip('/')}", json=payload, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


def _get(endpoint: str, path: str, timeout: float = 60.0) -> dict:
    import requests

    resp = requests.get(f"{endpoint.rstrip('/')}/{path.lstrip('/')}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _sglang_tp_size(endpoint: str) -> int:
    info = _get(endpoint, "get_server_info")
    for key in ("tp_size", "tensor_parallel_size"):
        if key in info:
            return int(info[key])
    args = info.get("server_args", {})
    if "tp_size" in args:
        return int(args["tp_size"])
    raise RuntimeError(f"could not determine tp_size from {endpoint}: {info}")


def _routable_ip() -> str:
    """This pod's IP as reachable by the SGLang pod (not 127.0.0.1, which the
    Megatron group uses)."""
    ip = os.environ.get("POD_IP")
    if ip:
        return ip
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets sent; just picks the egress iface
        return s.getsockname()[0]
    finally:
        s.close()


def _dtype_to_str(dtype) -> str:
    import torch

    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
        getattr(torch, "float8_e4m3fn", None): "float8_e4m3fn",
    }.get(dtype, str(dtype).replace("torch.", ""))


# Collective command opcodes broadcast from rank 0 to followers.
CMD_COMPUTE_LOG_PROB = "compute_log_prob"
CMD_UPDATE_ACTOR = "update_actor"
CMD_PUSH_WEIGHTS = "push_weights"
CMD_SAVE_WEIGHTS = "save_weights"
CMD_SAVE_CHECKPOINT = "save_checkpoint"
CMD_LOAD_CHECKPOINT = "load_checkpoint"
CMD_SHUTDOWN = "shutdown"


class MegatronRLServer:
    def __init__(self, config, port: int):
        self.config = config
        self.port = port
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.actor = None  # verl TrainingWorker
        self._weight_group = None  # cross-cluster NCCL group for nccl_http

    # ------------------------------------------------------------------ #
    # Build the verl actor TrainingWorker (mirrors ActorRolloutRefWorker)
    # ------------------------------------------------------------------ #

    def build_actor(self):
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import ActorConfig, HFModelConfig  # noqa: F401
        from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
        from verl.workers.utils.losses import ppo_loss

        arr = self.config.actor_rollout_ref

        # The V1 driver injects trainer.total_training_steps into
        # actor.optim.total_training_steps before building workers
        # (trainer_base._init_dataloader). A standalone server must do the
        # same, else mcore's OptimizerParamScheduler asserts
        # ``lr_decay_steps > 0`` on the -1 default.
        optim = arr.actor.optim
        try:
            tts = int(optim.get("total_training_steps", -1))
        except (TypeError, ValueError):
            tts = -1
        if tts <= 0:
            trainer_tts = (
                self.config.get("trainer", {}).get("total_training_steps", None)
                if hasattr(self.config, "get")
                else None
            )
            optim.total_training_steps = max(1, int(trainer_tts) if trainer_tts else 4)
            logger.info(
                "injected actor.optim.total_training_steps=%d (driver normally does this)",
                optim.total_training_steps,
            )

        # 1. model + actor config (megatron strategy lives under actor.engine)
        model_config = omega_conf_to_dataclass(arr.model)
        actor_config = omega_conf_to_dataclass(arr.actor)
        actor_config.model_config = model_config

        # 2. TrainingWorkerConfig (same fields ActorRolloutRefWorker assembles)
        training_config = TrainingWorkerConfig(
            model_type=actor_config.model_config.get("model_type", "language_model"),
            model_config=actor_config.model_config,
            engine_config=actor_config.engine,
            optimizer_config=actor_config.optim,
            checkpoint_config=actor_config.checkpoint,
        )
        ec = training_config.engine_config
        ec.use_dynamic_bsz = arr.actor.use_dynamic_bsz
        ec.infer_max_token_len_per_gpu = arr.rollout.log_prob_max_token_len_per_gpu
        ec.infer_micro_batch_size_per_gpu = arr.rollout.log_prob_micro_batch_size_per_gpu
        ec.max_token_len_per_gpu = arr.actor.ppo_max_token_len_per_gpu
        ec.micro_batch_size_per_gpu = arr.actor.ppo_micro_batch_size_per_gpu
        ec.use_remove_padding = model_config.get("use_remove_padding", False)

        # 3. build + initialize + attach the GRPO/PPO loss
        self.actor = TrainingWorker(config=training_config)
        self.actor.reset()  # engine.initialize(): builds mcore model, loads HF weights
        self.actor.set_loss_fn(partial(ppo_loss, config=actor_config))
        logger.info("actor TrainingWorker ready (rank=%d/%d)", self.rank, self.world_size)

    # ------------------------------------------------------------------ #
    # Collective ops (run on ALL ranks)
    # ------------------------------------------------------------------ #

    def _inject_sampling_meta(self, td):
        """Inject the sampling fields verl's forward reads from meta (e.g.
        ``temperature``) which the driver normally sets from rollout config.
        Applying the configured sampling temperature server-side keeps the
        importance-sampling correction correct without threading it over the
        wire."""
        from verl.utils import tensordict_utils as tu

        # Instantiate any lazy representation so adding non-tensor fields can't
        # trip "modifying batch size of a lazy TD".
        if hasattr(td, "to_tensordict"):
            td = td.to_tensordict()
        if "temperature" not in td.keys():
            temp = float(self.config.actor_rollout_ref.rollout.get("temperature", 1.0))
            tu.assign_non_tensor(td, temperature=temp)
        # The megatron engine's forward reads ``loss_mask`` (driver-side batches
        # carry it); when a hand-rolled batch only has ``response_mask``, derive
        # it so direct posts keep working.
        if "loss_mask" not in td.keys() and "response_mask" in td.keys():
            td["loss_mask"] = td["response_mask"].to(td["input_ids"].dtype)
        # global_token_num drives the MFU calc; verl's driver sets it as a
        # per-sequence list. Derive it from the input ids when absent so a
        # directly-posted batch works (an int here crashes
        # FlopsCounter.estimate_flops: `sum(int)`).
        if "global_token_num" not in td.keys():
            ids = td["input_ids"]
            if getattr(ids, "is_nested", False):
                gtn = ids.offsets().diff().tolist()
            elif "attention_mask" in td.keys():
                gtn = td["attention_mask"].sum(dim=-1).tolist()
            else:
                gtn = [int(ids.shape[-1])] * int(ids.shape[0])
            tu.assign_non_tensor(td, global_token_num=gtn)
        return td

    def _compute_log_prob(self, td):
        # forward, no grad → old_log_probs (TrainingWorker.infer_batch, compute_loss=False)
        td = self._inject_sampling_meta(td)
        from verl.utils import tensordict_utils as tu

        tu.assign_non_tensor(td, compute_loss=False)
        out = self.actor.infer_batch(td)
        return out.cpu() if out is not None else None

    def _update_actor(self, td):
        # one mini-batch: forward+backward+optimizer step with the PPO/GRPO loss
        td = self._inject_sampling_meta(td)
        out = self.actor.train_mini_batch(td)
        return out.cpu() if out is not None else None

    def _ensure_weight_group(self, sglang_endpoints: list[str], group_name: str):
        """Form one NCCL group: this Megatron rank 0 + the SGLang TP workers.
        Persistent — formed once, reused every step. Mirrors external_sglang."""
        if self._weight_group is not None:
            return
        import torch

        master_addr = _routable_ip()
        from verl.utils.net_utils import get_free_port

        master_port = get_free_port(master_addr)[0]

        # rank layout: trainer == rank 0, then each SGLang instance's TP workers.
        offset = 1
        rank_offset = {}
        for ep in sglang_endpoints:
            rank_offset[ep] = offset
            offset += _sglang_tp_size(ep)
        world_size = offset

        # Clear any stale group from a killed run (else SGLang 400s).
        for ep in sglang_endpoints:
            try:
                _post(ep, "destroy_weights_update_group", {"group_name": group_name})
            except Exception:
                pass

        def _join(ep):
            _post(
                ep,
                "init_weights_update_group",
                {
                    "master_address": master_addr,
                    "master_port": master_port,
                    "rank_offset": rank_offset[ep],
                    "world_size": world_size,
                    "group_name": group_name,
                    "backend": "nccl",
                },
                timeout=600.0,
            )

        threads = [threading.Thread(target=_join, args=(ep,), daemon=True) for ep in sglang_endpoints]
        for t in threads:
            t.start()

        # Form the group on our side as rank 0 using SGLang's own primitive (a
        # fresh NCCL group over a dedicated TCP store that can span out-of-process
        # SGLang TP workers — torch.distributed.new_group cannot).
        from sglang.srt.utils import init_custom_process_group

        torch.cuda.set_device(torch.cuda.current_device())
        self._weight_group = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=0,
            group_name=group_name,
        )
        for t in threads:
            t.join()
        logger.info("weight-sync NCCL group formed: world_size=%d master=%s:%d", world_size, master_addr, master_port)

    def _push_weights(
        self,
        transport: str,
        step: int,
        sglang_endpoints=None,
        group_name="verl_ms_weight_sync",
        chunk_tensors: int = 32,
        **kwargs,
    ) -> dict:
        per_tensor_param, _extra = self.actor.engine.get_per_tensor_param()
        if transport != "nccl":
            raise ValueError(f"train_server only serves transport='nccl' here; got {transport!r}")

        # Followers (rank>0) must still drain the generator so any TP/DP
        # all-gathers inside get_per_tensor_param complete on every rank.
        if self.rank != 0:
            for _ in per_tensor_param:
                pass
            return {"pushed": True, "rank": self.rank}

        import torch.distributed as dist

        sglang_endpoints = [e.rstrip("/") for e in (sglang_endpoints or [])]
        self._ensure_weight_group(sglang_endpoints, group_name)

        n = 0
        chunk: list = []

        def _flush(batch):
            nonlocal n
            if not batch:
                return
            names = [nm for nm, _ in batch]
            dtypes = [_dtype_to_str(t.dtype) for _, t in batch]
            shapes = [list(t.shape) for _, t in batch]
            # SGLang blocks on the NCCL recv inside these HTTP calls → run in
            # threads while rank 0 broadcasts the same tensors in order.
            http = [
                threading.Thread(
                    target=_post,
                    args=(
                        ep,
                        "update_weights_from_distributed",
                        {
                            "names": names,
                            "dtypes": dtypes,
                            "shapes": shapes,
                            "group_name": group_name,
                            "flush_cache": False,
                        },
                    ),
                )
                for ep in sglang_endpoints
            ]
            for t in http:
                t.start()
            for _nm, tensor in batch:
                dist.broadcast(tensor.detach().cuda(), src=0, group=self._weight_group)
                n += 1
            for t in http:
                t.join()

        for name, tensor in per_tensor_param:
            chunk.append((name, tensor))
            if len(chunk) >= chunk_tensors:
                _flush(chunk)
                chunk = []
        _flush(chunk)

        for ep in sglang_endpoints:
            _post(ep, "flush_cache", {})
        return {"pushed": True, "transport": "nccl", "num_tensors": n, "step": step}

    def _save_weights(self, path: str) -> dict:
        """Export current weights as an SGLang-loadable HF dir at ``path`` (for
        the ``store`` transport). Uses the engine's per-tensor HF iterator
        (``bridge.export_hf_weights``) — the same source verl uses for weight
        sync — and copies the base config/tokenizer once so SGLang's
        ``/update_weights_from_disk`` can read it. Only rank 0 writes."""
        import os
        import shutil

        n = 0
        if self.rank == 0:
            from safetensors.torch import save_file

            per_tensor_param, _extra = self.actor.engine.get_per_tensor_param()
            state = {}
            for name, tensor in per_tensor_param:
                state[name] = tensor.detach().to("cpu", dtype=self._export_dtype()).contiguous()
                n += 1
            os.makedirs(path, exist_ok=True)
            save_file(state, os.path.join(path, "model.safetensors"))
            # Copy config/tokenizer from the base model dir (idempotent).
            base = self.config.actor_rollout_ref.model.path
            for f in (
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
            ):
                src = os.path.join(base, f)
                dst = os.path.join(path, f)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)
        # Barrier so followers don't race ahead of rank 0's write.
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
        return {"saved": True, "path": path, "num_tensors": n}

    @staticmethod
    def _export_dtype():
        import torch

        return torch.bfloat16

    def _save_checkpoint(self, path: str, step: int) -> dict:
        self.actor.save_checkpoint(local_path=path, hdfs_path=None, global_step=step)
        return {"saved": True, "path": path, "step": step}

    def _load_checkpoint(self, path: str, del_local_after_load: bool = False) -> dict:
        # Symmetric with _save_checkpoint: the Megatron engine reloads model +
        # optimizer state from ``path`` (the same verl-side global_step_N/actor
        # dir it was saved to — driver and server share the pod filesystem).
        self.actor.load_checkpoint(local_path=path, hdfs_path=None, del_local_after_load=del_local_after_load)
        return {"loaded": True, "path": path}

    # ------------------------------------------------------------------ #
    # Rank-0 HTTP server  /  followers loop
    # ------------------------------------------------------------------ #

    def _broadcast_command(self, cmd: str, payload: dict | None = None):
        import torch.distributed as dist

        obj = [cmd, payload]
        dist.broadcast_object_list(obj, src=0)
        return obj[0], obj[1]

    def _follower_loop(self):
        while True:
            cmd, payload = self._broadcast_command("", None)
            if cmd == CMD_SHUTDOWN:
                break
            try:
                self._dispatch(cmd, payload)
            except Exception:  # keep followers alive; rank 0 surfaces the error
                logger.exception("follower op %s failed", cmd)

    def _dispatch(self, cmd: str, payload: dict):
        from recipe.remote_megatron_sglang.server import protocol as P

        if cmd == CMD_COMPUTE_LOG_PROB:
            return self._compute_log_prob(P.loads_td(payload["blob"]))
        if cmd == CMD_UPDATE_ACTOR:
            return self._update_actor(P.loads_td(payload["blob"]))
        if cmd == CMD_PUSH_WEIGHTS:
            return self._push_weights(**payload)
        if cmd == CMD_SAVE_WEIGHTS:
            return self._save_weights(**payload)
        if cmd == CMD_SAVE_CHECKPOINT:
            return self._save_checkpoint(**payload)
        if cmd == CMD_LOAD_CHECKPOINT:
            return self._load_checkpoint(**payload)
        raise ValueError(f"unknown command {cmd!r}")

    def serve(self):
        self.build_actor()
        if self.rank != 0:
            self._follower_loop()
            return

        from aiohttp import web
        from recipe.remote_megatron_sglang.server import protocol as P

        async def _compute(request, cmd):
            blob = await request.read()
            self._broadcast_command(cmd, {"blob": blob})
            td = P.loads_td(blob)
            out = self._compute_log_prob(td) if cmd == CMD_COMPUTE_LOG_PROB else self._update_actor(td)
            return web.Response(body=P.frame_tensordict(out))

        async def health(_):
            return web.json_response({"status": "ok", "world_size": self.world_size})

        async def compute_log_prob(r):
            return await _compute(r, CMD_COMPUTE_LOG_PROB)

        async def update_actor(r):
            return await _compute(r, CMD_UPDATE_ACTOR)

        async def _json_cmd(request, cmd, fn):
            payload = await request.json() if request.can_read_body else {}
            self._broadcast_command(cmd, payload)
            return web.json_response(fn(**payload))

        async def push_weights(r):
            return await _json_cmd(r, CMD_PUSH_WEIGHTS, self._push_weights)

        async def save_weights(r):
            return await _json_cmd(r, CMD_SAVE_WEIGHTS, self._save_weights)

        async def save_checkpoint(r):
            return await _json_cmd(r, CMD_SAVE_CHECKPOINT, self._save_checkpoint)

        async def load_checkpoint(r):
            return await _json_cmd(r, CMD_LOAD_CHECKPOINT, self._load_checkpoint)

        async def init_broadcast_group(request):
            # Join the cross-cluster NCCL group used by the nccl_http transport.
            # payload: master_address / master_port / world_size / backend.
            # # VALIDATE: build a group spanning this rank 0 + the SGLang TP
            # workers (e.g. a side TCPStore) and stash it in self._weight_group.
            _ = await request.json()
            return web.json_response({"ok": True})

        async def noop(_):
            return web.json_response({"ok": True})

        app = web.Application(client_max_size=1024**4)
        app.add_routes(
            [
                web.get(P.HEALTH, health),
                web.post(P.COMPUTE_LOG_PROB, compute_log_prob),
                web.post(P.COMPUTE_REF_LOG_PROB, compute_log_prob),  # ref uses same infer path
                web.post(P.UPDATE_ACTOR, update_actor),
                web.post(P.PUSH_WEIGHTS, push_weights),
                web.post(P.SAVE_WEIGHTS, save_weights),
                web.post(P.SAVE_CHECKPOINT, save_checkpoint),
                web.post(P.INIT_BROADCAST_GROUP, init_broadcast_group),
                web.post(P.DESTROY_BROADCAST_GROUP, noop),
                web.post(P.LOAD_CHECKPOINT, load_checkpoint),
                web.post(P.INIT_MOONCAKE, noop),
                web.post(P.DESTROY_MOONCAKE, noop),
            ]
        )
        logger.info("Megatron RL server listening on :%d (rank 0, world_size=%d)", self.port, self.world_size)
        web.run_app(app, port=self.port, access_log=None)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Full verl PPO config YAML (only actor_rollout_ref is read).")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from omegaconf import OmegaConf

    config = OmegaConf.load(args.config)
    MegatronRLServer(config, port=args.port).serve()


if __name__ == "__main__":
    main()
