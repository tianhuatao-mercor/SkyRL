# B300 SkyRL lifecycle qualification

This recipe qualifies the currently untested integrated seam without repeating
the already-passed container, Megatron checkpoint, or standalone vLLM gates.
It runs one dense Qwen3-0.6B Megatron policy GPU and one eager vLLM GPU on a
single B300 worker, with an initial and post-update two-rank NCCL weight sync.

The training input is the exact read-only artifact
`/shared/datasets/skyrl-multiply-lifecycle-ebcf5477cdd43cf4`. The harness uses
real pinned vLLM rollouts but replaces only the training rewards with a recorded
per-prompt alternating 0/1 pattern. This prevents a one-step GRPO transport
canary from becoming a zero-advantage no-op; it is not a quality benchmark.
Evaluation retains the multiplication environment's real reward.

The launcher refuses reused paths and unexpected image, model, dataset, Git,
GPU, container, or process state. Runtime patches are content-addressed and
mounted read-only over the qualified r1 image. All vLLM development endpoints,
the router, and the transient NCCL rendezvous bind/advertise on loopback.

Run only after live-state inspection and approval:

```bash
platform/b300/lifecycle/run_lifecycle.sh \
  --execute \
  --dataset-dir /shared/datasets/skyrl-multiply-lifecycle-ebcf5477cdd43cf4 \
  --gpus 0,1
```

The hard runtime bound is 30 minutes. PASS requires two accounted NCCL
transfers, weight versions 1 then 2, a finite nonzero policy gradient, changed
trainer parameters, post-update inference above the measured unchanged-weight
noise floor, a complete step-1 checkpoint, and run-owned cleanup. Evidence and
checkpoint paths are made read-only after final checksums are written. Failed
attempts are also checksummed and frozen by the exit trap. The verifier accepts
both wrapped `model.*` state and Megatron-Core's direct `embedding.*` plus
`decoder.*` distributed-checkpoint layout; `--output` permits a verification
record to be written to a separate audit directory without mutating a frozen
source run.

After the lifecycle passes, independently reload its frozen step-1 Hugging Face
export in a fresh, one-GPU vLLM process and replay the two recorded greedy eval
prompts:

```bash
platform/b300/lifecycle/run_fresh_vllm_replay.sh --execute --gpu 0
```

This follow-up does not train or synchronize weights. It verifies the source
run and checkpoint checksums before GPU allocation, mounts `/shared` read-only,
binds vLLM only to loopback, requires exact response token IDs on two replay
passes, bounds aligned logprob drift to `2e-3`, proves cleanup, and freezes a
separate evidence directory for the fresh process.

The next scale gate keeps the same bounded dataset and single policy rank but
uses two independent one-GPU vLLM engines. GPU order is rollout engines first,
policy last:

```bash
platform/b300/lifecycle/run_lifecycle.sh \
  --execute \
  --dataset-dir /shared/datasets/skyrl-multiply-lifecycle-ebcf5477cdd43cf4 \
  --num-engines 2 \
  --gpus 0,1,2
```

In addition to the original lifecycle assertions, the verifier requires a
three-rank trainer/receiver NCCL group, two distinct loopback server URLs, and
router evidence that both engine indexes handled requests.

After that scale gate passes, the bounded restart gate loads its exact frozen
step-1 distributed checkpoint into a fresh process, writes only to a new run
tree, advances one step, and synchronizes both engines again:

```bash
platform/b300/lifecycle/run_lifecycle.sh \
  --execute \
  --dataset-dir /shared/datasets/skyrl-multiply-lifecycle-ebcf5477cdd43cf4 \
  --num-engines 2 \
  --gpus 0,1,2 \
  --resume-from /shared/checkpoints/qualifications/20260827T000439Z-skyrl-lifecycle-nccl-dense-2eng-r1/checkpoints/global_step_1
```

The launcher accepts no other resume source for this qualification. It verifies
the frozen source evidence and checkpoint before, after, and following artifact
verification. PASS additionally requires the fresh trainer's step-1 export to
match the source tensors exactly, explicit trainer/optimizer/dataloader restore
markers, a real step-2 update and checkpoint, two three-rank NCCL transfers,
traffic through both engines, repeatable step-2 inference, and clean teardown.
The source dataloader state was saved on the final batch while its iterator was
still marked unfinished. The gate therefore records that state and allows one
empty restored outer iteration before the next epoch; `max_training_steps=2`
still limits execution to exactly one new optimizer step.
