# Power-Step Normalisation Fusion

Power-step fusion targets iterative solvers that repeatedly multiply by a matrix and renormalise the result (e.g. block power iteration, Krylov updates, orthogonal projection loops).

## What Qualifies

- **Matmul numerator.** The numerator must be a builtin `mtimes(lhs, rhs)`.
- **Denominator structure.** The denominator must follow `sqrt(sum((numerator.^2)) + epsilon)` or `sqrt(sum((numerator.^2)))` with an optional scalar epsilon add.
- **Shared input.** The `.^2` node must receive the same value ID as the matmul output; extra reshapes/transposes are not recognised.
- **Constant epsilon.** If an epsilon is present it must be a scalar literal so the planner can embed it.
- **Exclusive ownership.** Nodes in the pattern (matmul, pow, sum, add, sqrt, div) cannot be part of another fusion group.

## Why Power-Step Normalisation?

Power-step normalisation is a common pattern in iterative solvers. It shows up in block power iteration, Krylov updates, orthogonal projection loops, and other iterative solvers. By fusing this pattern, we can keep the normalised result resident on the GPU, avoiding the overhead of uploading and downloading it to the CPU, and allows us to compute it without launching a second kernel.

## Benefits

- **Fewer GPU launches.** We can consolidate multiple power-step normalisation computations into a single kernel, reducing the number of GPU launches.

## Limitations & Tips

- The detector expects the epsilon add (if any) to appear either inside or outside the `sqrt` exactly once. Reordering the scalar add will break fusion.
- Different normalisation schemes (e.g. per-column scaling or norms other than 2-norm) are not yet fused.
- Very large `lhs`/`rhs` shapes may trip the provider’s matmul guardrails and force a fallback.

If a workload should fuse but does not, enable `RUNMAT_DEBUG_FUSION=1` to have the planner print why a node was rejected, then compare against the criteria above.