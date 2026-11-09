# FEA-FLUID-INPUTS-001: Exact typed fluid boundaries, sources and initial conditions

0 PASS, 4 PARTIAL, 0 FAIL, 24 OPEN, 0 TAGGED. Generated; do not edit.

## boundaries

Evidence kind: SemanticRealization. Criterion: Implement component velocity, tangential moving-wall, symmetry, periodic and pressure-reference constraints plus inlet velocity, outlet pressure and mass-flow boundaries on exact regions. Reject unsupported normal wall motion unless a qualified ALE formulation governs it.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::kinematic::inlet_velocity_and_wall_motion_reproduce_the_constraint_driven_couette_flow` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/kinematic.rs:62) - An inlet_velocity and wall_motion constraint pair reproduces constraint-driven Couette flow in the kernel.<br>`runmat-analysis-fea[lib]::physics::cfd::canonical::tests::symmetry::closed_component_with_symmetry_planes_reproduces_couette_flow` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/symmetry.rs:190) - a closed component bounded by velocity walls and symmetry planes (a spanwise slab of a driven cavity) takes its pressure level from the authored gauge and repro...<br>`runmat-analysis-fea[lib]::physics::cfd::canonical::tests::tangential_moving_wall_supported_but_normal_motion_requires_ale` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests.rs:389) - A tangential moving wall is supported and normal wall motion is rejected unless an ALE formulation governs it; velocity components, symmetry, periodic and press... |
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
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::kinematic::inlet_velocity_and_wall_motion_reproduce_the_constraint_driven_couette_flow` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/kinematic.rs:62) - inlet_velocity (vector signal) and wall_motion (scalar speed with unit direction) bind the same face law as velocity constraints and reproduce Couette flow.<br>`runmat-analysis-fea[lib]::physics::cfd::canonical::tests::mass_flow::signed_mass_sources_preserve_exact_faces_partitions_and_cold_original_fields` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/mass_flow.rs:12) - Signed rates across rotated exact faces; three modulo-index pieces demonstrate additive realization only, not distributed cell assembly. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::time_signals::body_force_time_sampling_preserves_independent_volume_resultant` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/time_signals.rs:166) - Independent volume-integrated time sampling; no partitioned algebra credit.<br>`runmat-analysis-fea[lib]::physics::cfd::canonical::tests::time_signals::one_sided_velocity_jump_is_realized_on_original_boundary_faces` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/time_signals.rs:266) - A tabulated vector velocity signal with a one-sided jump is realized on the original boundary faces; pressure, mass-flow and body-force signals are separate tes... |
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
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::mass_flow::source_authority_conflicts_quantity_and_budget_reject_before_solve` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/mass_flow.rs:85) - Selected duplicate/wrong-quantity/source authority and budget rejection only. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

