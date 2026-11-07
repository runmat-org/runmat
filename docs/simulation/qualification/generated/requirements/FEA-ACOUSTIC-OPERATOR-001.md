# FEA-ACOUSTIC-OPERATOR-001: Canonical material-region Helmholtz operator

0 PASS, 3 PARTIAL, 0 FAIL, 5 OPEN, 0 TAGGED. Generated; do not edit.

## integrals

Evidence kind: IndependentReference. Criterion: Independent exact simplex or manufactured references verify volume gradient/mass and boundary terms for both element orders, orientations and physical material scaling.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | OPEN |  |
| acoustic_harmonic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::radiation::exact_operators::tet10_and_tri6_all_entries_match_independent_simplex_moments_on_every_face` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/radiation/exact_operators.rs:252) - Every Tet10/Tri6 matrix entry on all faces; independent simplex moments. No Tet4 or public topology audit credit. |

## regions

Evidence kind: SpatialConvergence. Criterion: Material-interface pressure, normal velocity and power continuity agree with independent references under spatial refinement on exact assigned regions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::interface::two_fluid_interface_pressure_mean_velocity_and_power_refine` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/interface.rs:280) - Aligned two-medium slab, phase rotation and spatial refinement; not arbitrary heterogeneous or curved-interface qualification. |
| acoustic_harmonic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::tests::interface::two_fluid_interface_pressure_mean_velocity_and_power_refine` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/tests/interface.rs:280) - Aligned two-medium slab, phase rotation and spatial refinement; not arbitrary heterogeneous or curved-interface qualification. |

## topology

Evidence kind: SemanticRealization. Criterion: Public execution consumes the authored canonical mesh without generated solver-graph substitution, and retains original field topology through native recovery and cold transport.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_harmonic | tet4 | native_process | OPEN |  |
| acoustic_harmonic | tet10 | native_process | OPEN |  |
| acoustic_harmonic | tet4 | public_study | OPEN |  |
| acoustic_harmonic | tet10 | public_study | OPEN |  |

