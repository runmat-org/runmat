# FEA-FLUID-EQUATIONS-001: Original incompressible laminar mass and momentum equations

0 PASS, 4 PARTIAL, 0 FAIL, 20 OPEN, 0 TAGGED. Generated; do not edit.

## flux

Evidence kind: PhysicalBalance. Criterion: Evaluate one original oriented owner-face mass flux and conservative advective, viscous and pressure momentum actions with equal/opposite internal-face contributions. Document interpolation, stabilization and non-orthogonal/skewness corrections within validated mesh limits.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::affine_skew_face_velocity_and_nonzero_convection_are_consistent` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests.rs:94) - Independent affine skew-face flux and integrated nonzero advection; no general nonlinear field-convergence credit. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## coupling

Evidence kind: PhysicalBalance. Criterion: Pressure-velocity coupling solves original incompressibility and momentum equations. Component gauges/nullspaces are explicit, invalid or incompatible references reject, and physical residual checks cannot be replaced by solver convergence. Incompressibility is part of the mixed fluid formulation, never a generic user-authored equality constraint.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::closed_reference::closed_gauge_admission_rejects_missing_duplicate_outlet_and_incompatible_flux` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/closed_reference.rs:91) - Closed gauge/reference and net flux rejection; not full pressure-velocity envelope proof. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

## pressure_state

Evidence kind: OriginalStateComparison. Criterion: Keep original reference-relative pressure as governing state through iteration, recovery and cold execution. Absolute pressure is a projection; converting it back must not erase fluctuations. Bind pressure references and retain owner-face flux rather than reconstructing missing original state.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::pressure_reference::independent_uniform_translation_is_balanced_at_both_pressure_references` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/pressure_reference.rs:13) - A uniform translation is balanced at both pressure references.<br>`runmat-analysis-fea[lib]::physics::cfd::canonical::tests::pressure_reference::original_fluctuation_survives_absolute_rounding_fresh_solve_and_cold_recovery` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/pressure_reference.rs:60) - Fourier duct scaling and two gauges retain bitwise original velocity/fluctuation/flux with and without acceleration; cold JSON replay only. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | PARTIAL | `runmat-execution-runner-native[lib]::analysis_kernel::fluid_steady::tests::pressure_state::native_pressure_projection_preserves_original_governing_state` (crates/runmat-execution-runner-native/src/analysis_kernel/fluid_steady/tests/pressure_state.rs:9) - Native result/receipt preserves original governing fields under gauge shift; not distributed algebra or producer-death transient evidence. |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

