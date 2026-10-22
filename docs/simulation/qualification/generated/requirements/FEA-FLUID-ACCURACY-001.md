# FEA-FLUID-ACCURACY-001: Independent flow benchmark and convergence acceptance

0 PASS, 2 PARTIAL, 0 FAIL, 6 OPEN, 0 TAGGED. Generated; do not edit.

## steady_references

Evidence kind: IndependentReference. Criterion: Validate Poiseuille and Couette plus lid-driven cavity and backward-facing step across the documented Reynolds range, with velocity/pressure/flow/wall quantities and conservative closure.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::affine_couette_uses_original_geometry_and_balanced_physical_fields` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests.rs:153) - Independent affine Couette velocity, wall shear and original balance; not cavity/backward-facing step. |

## spatial

Evidence kind: SpatialConvergence. Criterion: Demonstrate spatial convergence and absolute error for non-affine flow, pressure, integrated flow and wall stress within the documented quality/Reynolds envelope; affine exactness alone does not establish convergence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::rectangular_duct_compares_fourier_reference_with_controlled_tail` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests.rs:262) - Controlled-tail non-affine Fourier duct: velocity, flow and wall shear refinement; not pressure/cavity/backward-facing-step envelope. |

## acceptance

Evidence kind: AccuracyAcceptance. Criterion: Shared accuracy assessment consumes original result, receipts and applicable independent spatial/time evidence. Missing, failed or indeterminate mandatory criteria block completed semantic publication; imported provenance cannot create numerical proof.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

