# Elementwise Chain Fusion

Elementwise fusion collapses straight-line arithmetic / transcendental expressions into a single GPU kernel.

## What Qualifies

- **Elementwise-only nodes.** All participating primitives or builtins must be marked elementwise in the acceleration graph (`+`, `-`, `.*`, `./`, `.^`, trig, exp/log, etc.).
- **Single-consumer frontier (with safe fan-out).** The primary detection pass builds a straight chain, but the planner can pull in branchy elementwise nodes that only feed sinks (casts, gathers, stores) via *fan-out expansion*. This lets us fuse producer chains even when the final value fans out to e.g. `single(...)` and `gather(...)` as long as each branch terminates in an elementwise-only leaf.
- **Shape compatibility.** The planner verifies that shapes unify through broadcasting. Unknown shapes (e.g. runtime-only sizes) break the chain.
- **Minimum length.** Chains must contain at least two elementwise nodes; single operations execute through their normal builtin path.

## Why Elementwise Chains?

Elementwise chains represent one of the most common patterns in numerical computing. They appear in nearly every numeric workload and are the easiest way to collapse dozens of scalar ops into one dispatch. Keeping them fused prevents bandwidth blowups from repeated reads/writes of the same tensor.

## Supported Operations

- MATLAB primitives: `+`, `-`, `.*`, `./`, `.\`, `.ˆ`, unary `-`, unary `+`, `abs`, `sign`, comparison ops, and many others registered via `register_builtin_fusion_spec`.
- Builtins covering transcendental math: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `log10`, `log1p`, `sqrt`, `rsqrt`, `pow2`, `pow10`, etc.
- Elementwise logical ops (`~`, `&`, `|`) as long as the inputs resolve to numeric/logical tensors.
- Scalar constants (`single`, `double`, literal numbers) are folded directly into the generated expression.

## Not Supported

- Any node that performs reduction, matrix multiply, reshape, or other non-elementwise behaviour.
- Elementwise nodes that have more than one consumer (e.g. shared subexpressions). Duplicating work safely would require SSA rewrites that RunMat does not attempt today.
- Operations that require runtime data-dependent control flow (e.g. `if` statements inside arrayfun-like constructs).

## Troubleshooting Tips

- **Chain breaks unexpectedly:** Inspect telemetry for `fusion_kind=Reduction` or missing entries—usually indicates an intermediate value feeds multiple consumers or shapes are unknown. If the only consumers are terminal elementwise leaves, ensure `RUNMAT_DEBUG_FUSION=1` shows the fan-out expansion is running; nodes scheduled *before* the chain start are still ignored.
- **Broadcast mismatch errors:** When shapes can only be resolved at runtime, ensure your script provides concrete `Size` metadata (preallocate or annotate) so the planner can unify shapes statically.
- **Scalar options:** Remember that `single(x)`, `double(x)`, and logical casts are respected, but if you mix precisions the active provider must support the wider type (double on GPUs that expose it).

If a workload should fuse but does not, enable `RUNMAT_DEBUG_FUSION=1` to have the planner print why a node was rejected, then compare against the criteria above.