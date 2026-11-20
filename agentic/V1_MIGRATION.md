# Native verl V1 trainer migration

This migration targets verl `a4e0dd2188e3dd458a2cbb097c03d975ca77f159`, which removes the legacy trainer entry point. It enables TransferQueue and inherits the native V1 TaskRunner run/cleanup lifecycle. The existing proxy or standalone load-balancer registry starts before the native agent loop manager. No new AgentLoop implementation is introduced.

Install `TransferQueue>=0.1.11` in every Ray worker environment. Preserve any hardware-specific Torch/rollout installation. Recipe-generated parquet lives on the driver filesystem, so the controller is placed on the driver Ray node; model workers still use all configured nodes. This does not make checkpoint directories shared.

## Behavior-policy metadata and timeout metrics

Generation policy versions now propagate through the provider, both standalone worker caches and Ray-local completion records, into RemoteAgentLoop outputs. Version zero is valid. A trajectory with no generation keeps unknown (`None`) version bounds; it is not discarded or assigned the current trainer step.

`version_metrics.py` installs a recipe-local metric adapter on the native trainer before sampling. Native V1 currently casts every version tag, including padding, to integer. The adapter supplies a disposable metadata view solely for native non-version metrics, discards the provisional off-policy values, and computes published version statistics from known original versions only. Compatibility zeros are never persisted or used for training. Entirely unknown batches report known/unknown counts without span or staleness reductions. Trajectories, group membership and rewards are unchanged. No verl core patch is needed.

Unspecified per-call output length no longer inherits the proxy's old 2048-token default. Explicit client limits remain honored; otherwise the native engine uses its configured remaining context.

## Validation

Run `4b-rl-megatron-021` completed 32 real ACS Sandbox trajectories, one native GRPO update, checkpoint/HF export and reload on PPU. Training used 4 positive / 28 zero rewards, with 3 mixed-reward groups out of 8 groups of 4. Actor gradient norm was 0.2014486492; 307 tensors changed and all 4,539,265,536 model elements were finite. Timeout rewards reflect the 1800-second cutoff, separately from late verifier results. This establishes one stage-2 update, not learning quality or all four pipeline stages.

That distributed run used a native-core defensive metric patch. The replacement recipe-only metric adapter is a subsequent change, covered by local regression tests; it has not yet been rerun in distributed training. Policy-version propagation and output-default changes were exercised in the distributed run. No custom AgentLoop class is added.
