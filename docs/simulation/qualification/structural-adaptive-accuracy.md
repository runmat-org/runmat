# Structural adaptive iteration authority

## One accuracy owner

`StructuralAdaptationPolicy` and its local tolerance comparisons are removed.
An iteration requires an authored shared `AccuracyPlan`, evaluates through
`results::accuracy`, and retains that exact evaluation. Required mandatory roles are:

| Contribution | Evidence | Physical measure |
| --- | --- | --- |
| Recovery | A posteriori estimator | Frobenius volume-L2 discrepancy / original stress volume-L2 norm |
| Recovery | Convergence study | Current / source recovery discrepancy |
| Spatial discretization | Convergence study | Change of the paired, typed target quantity |
| Field transfer | Transfer bound | Every original transferred/reference field comparison |
| Algebraic | Residual bound | Retained target solver's normalized residual |

Recovery and transfer measures are dimensionless; a raw `Pa * m^(3/2)` discrepancy
is never mislabeled as a stress quantity. Target change uses either the authored
relative criterion or its matching physical absolute quantity. The reduction
criterion must require an actual reduction (relative ratio limit below one).
Extra mandatory criteria remain mandatory: unavailable evidence blocks the
iteration even when other contributions pass. If any required transferred field
lacks a relative measure, the complete transfer contribution is indeterminate.

`Converged` requires shared `Satisfied` and both solvers' convergence. A shared
`NotSatisfied` report permits `Continue` only when the authored action is `Refine`
and another cycle remains. Indeterminate evidence rejects and cannot produce adaptive
convergence. The shared refinement ceiling counts
completed mesh changes, including the first source-to-target pair. No separate
adaptive tolerance defaults or private publication evaluator remain.

## Original paired evidence and constraint lineage

Recovery retains the physical reference norm integrated from original stress
samples with the same element geometry and weights as the recovery calculation.
Element and point layouts both carry it. Raw indicators, marking and original
numerical criteria remain unchanged. Independent constant-tensor references and
rotation tests check the normalization for both element orders and layouts.

Iteration schema/domain V3 retains both source and target estimators, target
quantities, typed solver identities, transfer evidence, constraint-plan lineage,
authored accuracy plan and evaluation. Cold validation recomputes the shared
report and decisions; a predecessor check binds its exact digest, target-to-source
mesh, estimates, solver facts, quantity, accuracy plan and constraint-plan digest.
Legacy records without the reference norm or shared plan reject.

The existing canonical constraint compiler rebuilds source and target plans from
one model. Warm construction uses those fresh facts once; imported validation
recompiles and compares their digests. A persistent face clamp test crosses a
Tet4-to-refined-Tet10 change: three boundary corners become six constrained nodes,
with nine versus eighteen displacement rows. The old mesh-bound plan and a lost
persistent target reject. This is constraint recompilation, not reuse of numeric
node aliases.

## Qualification boundary

The cold two-refinement test verifies actual mesh/map/lineage and retained contract
chaining with synthetic numerical facts. It is not a solved adaptive physics or
native-process witness. Imported digests and reports do not independently certify
stress samples, result identities or target values. Existing manufactured,
interface, curved and original cantilever solves remain the scientific evidence;
the original cantilever convergence gate still fails.

This foundation does not qualify Stage 11. Authoring/planning the adaptive graph,
whole-task admission and measured memory, frozen physical source receipt binding,
actual adaptive re-solve and target convergence, durable/remote/retry/cancellation,
irreversible state transfer and semantic publication remain required. Public adaptive
execution must additionally admit a scientifically validated family estimator.
