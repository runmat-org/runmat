# Explained Variance Fusion

Explained-variance fusion accelerates diagnostics of the form `diag(Q' * G * Q)`, commonly used to measure how much energy an orthogonal basis captures against a covariance/Gram matrix.

## What Qualifies

- **Outer `diag` builtin.** The fused group must end with `diag(some_matrix)`.
- **Matrix product structure.** The `diag` argument must be `mtimes(mtimes(Q', G), Q)` where:
  - One `mtimes` input is a transpose of `Q`.
  - The other operands are `G` (Gram/covariance matrix) and `Q` (eigenvector estimates).
- **Matching dimensions.** `G` must be square with the same row count as `Q`’s leading dimension.
- **Exclusive chain.** The nodes forming the products and transpose cannot feed other consumers.

## Why Explained Variance?

Explained variance is a common metric in linear algebra and statistics. It shows up in principal component analysis, eigenvalue decomposition, and other linear algebra operations. By fusing this pattern, we can keep the explained variance vector resident on the GPU, avoiding the overhead of uploading and downloading it to the CPU, and allows us to compute it without launching a second kernel.

## Benefits

- **Fewer GPU launches.** We can consolidate multiple explained variance computations into a single kernel, reducing the number of GPU launches.

## Limitations

- Only the exact nested matmul + diag layout is recognised. If you compute variance via alternative formulas (e.g. elementwise multiply + sum), it will not fuse yet.
- The fusion still emits multiple provider matmuls internally; if the matrices exceed provider guardrails the execution may return an error and fall back.
- Debug printing can be enabled with `RUNMAT_DEBUG_EXPLAINED=1` to inspect shapes and sample data when parity issues arise.

If a workload should fuse but does not, enable `RUNMAT_DEBUG_FUSION=1` to have the planner print why a node was rejected, then compare against the criteria above.