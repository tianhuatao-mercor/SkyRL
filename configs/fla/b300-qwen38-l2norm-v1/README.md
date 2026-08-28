# B300 Qwen3.8 L2Norm control configs

These configs bypass FLA's per-shape Triton autotuning for the L2Norm kernels
used by Qwen3.8 GDN SFT. They are an experimental B300 control, not a generic
FLA default and not yet part of a qualified OCI image.

The selected launch values are the modal winners observed across the 64-rank
`20260828T040155Z-qwen38-64g-balanced-fla-cuda-resume90-5step` run:

- forward: `BT=16`, `num_warps=4`;
- backward: `BT=8`, `num_warps=4`.

Use them with `FLA_CACHE_MODE=default` and `FLA_CONFIG_DIR` pointing to this
directory. Revalidate numerical behavior and throughput whenever the GPU,
Triton, FLA, model geometry, or dtype changes.
