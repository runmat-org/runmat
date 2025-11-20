# Matmul Epilogue Fusion

Matmul epilogue fusion keeps the product of `mtimes(A, B)` on the GPU while folding scalar/vector epilogues into the same provider call.

## What Qualifies

- **One matmul root.** The planner looks for a `mtimes` node whose output feeds a single-consumer elementwise chain.
- **Allowed epilogue ops.** Addition, subtraction, multiplication, division (elementwise or scalar), broadcast multiply/divide by row/column vectors, clamp via `min`/`max`, power (`.^` or `pow`), and optional final `diag`.
- **Single consumer requirement.** Branched matmul outputs break fusion so each consumer can run independently.
- **Tensor residency.** Both inputs must be GPU-friendly (the executor uploads as needed) and match provider precision.

## Why Matmul Epilogue?

Matmul epilogue is a common pattern in linear algebra and statistics. It shows up in affine transforms, normalisation, gating, and other linear algebra operations. By fusing this pattern, we can keep the epilogue operations resident on the GPU, avoiding the overhead of uploading and downloading it to the CPU, and allows us to compute it without launching a second kernel.

## Scenarios that Fuse

- Affine transforms: `C = A*B .* scale + bias`, where `scale`/`bias` can be scalars or row/column vectors.
- Normalisation / gating: `C = clamp((A*B - mean) ./ sigma, lo, hi)`.
- Power heads: `C = (A*B).^gamma` for scalar `gamma`.
- Gram-diagonal extraction: `diag(A*B)` (after optional scaling) for explained-variance style metrics.

## Not Supported

- Non-elementwise consumers such as reshapes, reductions, or additional matmuls.
- Multiple chained matmuls (e.g. `(A*B)*C`)—only the first `mtimes` and its immediate elementwise epilogue fuse today.
- Data-dependent epilogues that require branching.

## Troubleshooting

- **Missing fusion:** Ensure the epilogue operations appear consecutively in the instruction stream with no intervening consumers. Broadcasting mismatches will also break detection.
- **Row/column scale inference:** The planner recognises column vectors (shape `[1, N]`) as column scales and row vectors (`[M, 1]`) as row scales. Higher-rank tensors will not be treated as scale factors.
- **Precision issues:** GPU matmul epilogues follow the provider’s precision. If you request `double` on a device without FP64, RunMat will fall back to the CPU path.

If a workload should fuse but does not, enable `RUNMAT_DEBUG_FUSION=1` to have the planner print why a node was rejected, then compare against the criteria above.