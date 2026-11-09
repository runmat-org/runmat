# FEA-FLUID-TOPOLOGY-001: Canonical conservative finite-volume projection

0 PASS, 4 PARTIAL, 0 FAIL, 28 OPEN, 0 TAGGED. Generated; do not edit.

## inventory

Evidence kind: PhysicalBalance. Criterion: Retain canonical cell identities, complete reciprocal cell-face incidence, owner/neighbour, oriented area vectors, cell volumes/centroids, centroid distances, interpolation weights, non-orthogonality/skewness metrics and exact original boundary provenance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::topology::finite_volume::tests::exact_two_cell_geometry_is_conservative_and_retains_original_boundary_mapping` (crates/runmat-analysis/fea/src/topology/finite_volume/tests.rs:80) - Two affine Tet4 cells; oriented surface closure, volume and original face mapping. No skew/curved mesh envelope proof. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## geometry

Evidence kind: ExecutionEquivalence. Criterion: Projection is deterministic under reordered source access and placement; oriented surfaces satisfy the divergence theorem and exact source/region ownership. Cache identity binds the original mesh and projection policy.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::topology::finite_volume::tests::stable_cell_face_physics_is_invariant_to_reordered_source_access_indices` (crates/runmat-analysis/fea/src/topology/finite_volume/tests.rs:140) - Two-cell source access reordering, not execution placement equivalence. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## quality

Evidence kind: RejectionInvariant. Criterion: Admit supported tetrahedral order/geometry and governed non-orthogonality/skewness limits before work. Reject inversion, degeneracy, unsupported curvature, nonmanifold/missing/ambiguous boundaries or mutated source geometry; never silently reduce the model.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::topology::finite_volume::tests::rejects_nonmanifold_missing_boundary_inversion_curvature_and_mutated_record` (crates/runmat-analysis/fea/src/topology/finite_volume/tests.rs:206) - Selected malformed source cases only, not complete quality-envelope convergence.<br>`runmat-analysis-fea[lib]::topology::finite_volume::tests::flat_study_cells_close_within_the_rounding_bound_and_keep_face_errors_detectable` (crates/runmat-analysis/fea/src/topology/finite_volume/tests.rs:475) - The flattest cells of three study meshes at their study offsets; the closure check admits them and still resolves a reversed face (V/2) by more than four orders... |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## budget

Evidence kind: ResourceAdmission. Criterion: Count construction inventory and checked scratch/traversal work before allocation; exact ceilings, overflow and cancellation govern topology work.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::topology::finite_volume::tests::exact_admission_bounds_succeed_and_each_one_below_rejects` (crates/runmat-analysis/fea/src/topology/finite_volume/tests.rs:253) - Exact finite-volume construction counts and one-below limits; requested bytes are not RSS. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

