# Reduction Fusion

Reduction fusion keeps column/row reductions (`sum`, `mean`, etc.) on the GPU so tall slices avoid CPU ping-pong.

## What Qualifies

- **Single reduction builtin.** The planner currently groups each reduction node on its own (`FusionKind::Reduction`). Producers/consumers remain separate but reuse the GPU output.
- **Supported builtins.** `sum`, `mean`, and NLMS-style custom reductions that register fusion specs in `runmat-runtime/src/builtins/math/reduction`.
- **Known reduction axis.** The runtime needs `reduce_len` (size along the reduction axis) and `num_slices` (remaining dimensions) to configure tiles.
- **Provider support.** The active acceleration provider must expose `fused_reduction`; otherwise the builtin falls back to host code.

## Why Reduction?

Reduction is a common pattern in linear algebra and statistics. It shows up in covariance matrix construction, principal component analysis, and other linear algebra operations. By fusing this pattern, we can keep the reduced tensor resident on the GPU, avoiding the overhead of uploading and downloading it to the CPU, and allows us to compute it without launching a second kernel.

## Not Supported

- Reductions embedded inside elementwise chains; today they always stand alone.
- Multi-output statistics (variance, std) unless the builtin tags them with an appropriate fusion template. Work to generalise via `ReductionFlavor` is in progress.
- Host-only reductions, e.g. when the input tensor already lives on the CPU or the provider rejects double precision.

If a workload should fuse but does not, enable `RUNMAT_DEBUG_FUSION=1` to have the planner print why a node was rejected, then compare against the criteria above.