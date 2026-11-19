# Native verl V1 trainer migration

This migration targets verl `a4e0dd2188e3dd458a2cbb097c03d975ca77f159`, which removes the legacy trainer entry point. It enables TransferQueue and inherits the native V1 TaskRunner run/cleanup lifecycle. The existing proxy or standalone load-balancer registry starts before the native agent loop manager. No new AgentLoop implementation is introduced.

Install `TransferQueue>=0.1.11` in every Ray worker environment. Preserve any hardware-specific Torch/rollout installation. Recipe-generated parquet lives on the driver filesystem, so the controller is placed on the driver Ray node; model workers still use all configured nodes. This does not make checkpoint directories shared.

## Validation and remaining integration gap

On two PPU workers, the migrated entry point loaded datasets, initialized 16 model ranks (TP8/PP2), initialized the proxy and native AgentLoopWorkerTQ, and submitted 32 remote Harbor rollouts. A separate 2049-token native engine probe completed forward/backward/optimizer with finite loss and gradient. These checks do not establish end-to-end RL correctness.

The online run reached checkpoint/HF shard creation, then failed in V1 `_compute_metrics` because RemoteAgentLoop outputs lack `min_global_steps`/`max_global_steps` tags. The existing provider reads inference-server metadata but does not preserve the policy version through the session cache into AgentLoopOutput. This remains unresolved in this draft; policy-version tags must be propagated from actual generation responses, not filled with a guessed current training step. The run has not passed model reload or effective weight-update verification. Server final rewards and timeout rewards consumed by training remain separate.

Tests in `agentic/tests/test_agentic_v1.py` cover native lifecycle inheritance and proxy/registry initialization ordering. They do not replace distributed training validation.
