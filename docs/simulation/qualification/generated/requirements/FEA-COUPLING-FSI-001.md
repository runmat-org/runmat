# FEA-COUPLING-FSI-001: Dynamics/fluid interaction with stable conservative ALE execution

0 PASS, 0 PARTIAL, 1 FAIL, 8 OPEN, 0 TAGGED. Generated; do not edit.

## interface

Evidence kind: PhysicalBalance. Criterion: Resolve exact interface sides and conservatively transfer traction and velocity with interface action/reaction and work balances; original structural and fluid equations must actually be re-solved.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| fluid_structure_interaction | canonical_mixed_topology | public_study | FAIL | `runmat[test:check_cli]::fsi_compliant_tube_example_runs_as_a_coupled_study` (crates/runmat-cli/tests/check_cli.rs:1237) - The compliant-tube example runs steady two-way FSI from `runmat run` to a completed, numerically acceptable run. |

## motion

Evidence kind: OriginalStateComparison. Criterion: Document supported mesh-motion/ALE and quality-triggered remeshing, validate transferred temporal/interface state and reject unsupported large motion or topology change explicitly.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| fluid_structure_interaction | canonical_mixed_topology | public_study | OPEN |  |

## iteration

Evidence kind: IndependentReference. Criterion: Use documented stable partitioned iteration, relaxation and added-mass stabilization; qualify spatial, temporal and coupling-iteration convergence and added-mass-sensitive independent reference cases.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| fluid_structure_interaction | canonical_mixed_topology | public_study | OPEN |  |

## orchestration

Evidence kind: ProcessLifecycle. Criterion: Checkpoint distributed coupling orchestration with whole-task resource admission, cancellation, bounded retry, interface history and publication fencing; release evidence and documentation describe the actually qualified envelope.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| fluid_structure_interaction | canonical_mixed_topology | public_study | OPEN |  |
| fluid_structure_interaction | canonical_mixed_topology | durable | OPEN |  |
| fluid_structure_interaction | canonical_mixed_topology | remote_execution | OPEN |  |
| fluid_structure_interaction | canonical_mixed_topology | parallel_cpu | OPEN |  |
| fluid_structure_interaction | canonical_mixed_topology | domain_decomposed | OPEN |  |
| fluid_structure_interaction | canonical_mixed_topology | claimed_accelerator | OPEN |  |

