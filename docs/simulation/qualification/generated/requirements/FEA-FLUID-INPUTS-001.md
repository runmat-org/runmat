# FEA-FLUID-INPUTS-001: Exact typed fluid boundaries, sources and initial conditions

3 PASS, 0 FAIL, 25 OPEN, 0 TAGGED. Generated; do not edit.

## boundaries

Evidence kind: SemanticRealization. Criterion: Implement component velocity, tangential moving-wall, symmetry, periodic and pressure-reference constraints plus inlet velocity, outlet pressure and mass-flow boundaries on exact regions. Reject unsupported normal wall motion unless a qualified ALE formulation governs it.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## signals

Evidence kind: SemanticRealization. Criterion: Use shared typed signals for inlet velocity, pressure, mass flow, wall motion and body acceleration. Realize space/time dependence conservatively, preserving declared mass/momentum resultants over whole boundaries, cells and partitions with explicit sign/frame conventions and event sidedness.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PASS | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::mass_flow::signed_mass_sources_preserve_exact_faces_partitions_and_cold_original_fields` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/mass_flow.rs:12) - Signed rates across rotated exact faces; three modulo-index pieces demonstrate additive realization only, not distributed cell assembly. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | PASS | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::time_signals::body_force_time_sampling_preserves_independent_volume_resultant` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/time_signals.rs:135) - Independent volume-integrated time sampling; no partitioned algebra credit. |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## initial

Evidence kind: SemanticRealization. Criterion: Resolve model-owned reusable initial velocity and pressure/reference fields on exact cell scope from one typed finite uniform value, exact non-frequency signal sample with coordinate/discontinuity side, or immutable complete field artifact. Check content, topology, quantity, shape and explicit vector frame. Study binding selects the consuming step; compatible upstream state takes precedence unless...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## rejection

Evidence kind: RejectionInvariant. Criterion: Reject missing/multiply matched/wrong-dimension material, cell, source or boundary authority, quantity conflicts, stale input and exceeded input realization budgets before numerical execution.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PASS | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::mass_flow::source_authority_conflicts_quantity_and_budget_reject_before_solve` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/mass_flow.rs:78) - Selected duplicate/wrong-quantity/source authority and budget rejection only. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

