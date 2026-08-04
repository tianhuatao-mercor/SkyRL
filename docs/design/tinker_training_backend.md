# Hosted Tinker training backend

## Scope

This backend makes the normal SkyRL trainer use Tinker's hosted LoRA training and
sampling APIs. It does not implement the existing, opposite direction where
SkyRL exposes a Tinker-compatible API.

The implementation keeps the SkyRL synchronous and fully-async trainer loops.
The Tinker adapter owns the hosted training client, immutable sampling clients,
optimizer calls, checkpoints, usage accounting, and teardown. The validated
scope is policy-only GRPO with `rollout_is`, LoRA rank 16, one update epoch per
batch, no critic, and no KL loss.

The two retained production examples are:

- `examples/train/gsm8k/run_gsm8k_tinker_qwen3_8b_lr1e4_nonthinking_1024.sh`
- `examples/train/fully_async/fully_async_run_gsm8k_tinker_qwen3_8b_lr5e5_nonthinking_1024.sh`

Both use Qwen3-8B, 64 prompts x 5 samples, a 1,024-token generation cap, and 50
optimizer steps. The async example uses `max_staleness_steps=4` and 320
generation workers. Launchers have a paid-run confirmation guard, a wall-clock
timeout, isolated checkpoint/W&B identities, and cumulative token, cost, wall
time, service-time, failure, and checkpoint metrics.

## Validation scope

The backend has bounded synchronous and fully-async GSM8K coverage, including
optimizer updates, sampler publication, usage accounting, checkpoint creation,
and checkpoint resume. Tinker checkpoints contain provider-native weights and
optimizer state; they are resumable through SkyRL but are not Hugging Face
exports. Unit tests use provider fakes and do not require credentials.

## Async semantic gap from Fireworks

The Fireworks async path can hot-load weights while an active response remains
in the same provider stream, preserving that stream's KV state. The current
Tinker SDK instead returns a complete sample from one immutable
`SamplingClient`; it does not expose partial-token retrieval or cross-client KV
continuation.

The Tinker backend therefore swaps the active sampling client atomically:

- an in-flight sampling request finishes entirely on the client/version it
  captured;
- a newly admitted request uses the latest published client; and
- no sequence is truncated, resumed on new weights, or given a SkyRL-recomputed
  KV cache.

This is fully-async training with stale rollouts, but it is not the Fireworks
mid-response weight-update behavior. `max_staleness_steps` bounds admission
capacity rather than enforcing a hard rejection threshold. In the retained
async run, 131 consumed groups exceeded the target of 4 and the maximum observed
staleness was 6. Exact sampler versions and rollout logprobs are retained so the
trainer can measure staleness and apply `rollout_is`.

## Multi-turn follow-up

GSM8K is single-turn, so it does not exercise the main multi-turn risk. A tool
or environment trajectory makes several sampling requests
separated by observations. Today, each request captures the latest Tinker
client, while the generator rejects a trajectory that mixes sampler versions.
A publication between two turns can therefore fail that trajectory.

The recommended first multi-turn design is to pin one immutable sampling client
to the trajectory `session_id` on its first request and release it in
`finish_session`. That preserves one behavior policy for the whole trajectory
and fits the existing one-version-per-trajectory metadata. It also requires
timeouts and cleanup for abandoned sessions, and long tool trajectories will
increase the lifetime and staleness of old clients.

Allowing each turn to use a newer client is possible, but would require
per-turn or per-token policy-version metadata, an oldest-version staleness rule,
and updates to merge, retry, and resume bookkeeping. Rollout logprobs make the
learning correction possible, but the current scalar trajectory version is not
enough.

Before a multi-turn run, also validate:

- exact token/logprob and loss-mask alignment around tool observations;
- a token-in/token-out chat template, because custom-template retokenization
  currently cannot preserve hosted response logprobs;
- Tinker model and context-length availability for the selected agent model;
- provider rate limits, retries, and long-trajectory timeouts; and
- environment isolation and credential handling.

The backend is ready for the retained single-turn GSM8K examples. Multi-turn
hosted training is not production-ready until the trajectory-version design is
implemented and tested.
