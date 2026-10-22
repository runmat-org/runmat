# FEA-THERMAL-DOMAIN-001: Independent thermal domain and regional heat-equation assembly

0 PASS, 5 PARTIAL, 0 FAIL, 7 OPEN, 0 TAGGED. Generated; do not edit.

## domain

Evidence kind: SemanticRealization. Criterion: Declare thermal reference temperature, initial temperature and compatible material capability explicitly; reject missing, ambiguous or incompatible domains, targets and initial-field lineage.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | public_study | OPEN |  |
| thermal_steady | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

## assembly

Evidence kind: IndependentReference. Criterion: Integrate conductivity, density times heat capacity, and volumetric sources per element and material region using shared topology, properties, algebra and constraints; preserve conductivity nullspace and integrated capacity.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::thermal::assembly::convergence_tests::two_material_slab_matches_series_resistance_and_authored_order_is_invariant` (crates/runmat-analysis/fea/src/physics/thermal/assembly/convergence_tests.rs:23) - Aligned Tet4 two-material series resistance, temperature and flux plus property-order invariance; no arbitrary nonuniform interface claim. |
| thermal_steady | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::numerical::canonical_tet10_thermal_matrices_pass_constant_patch_and_capacity_integral` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/numerical.rs:72) - Single Tet10 conductivity constant patch, integrated capacity, linear gradient and flux; no general curved-mesh convergence. |
| thermal_transient | tet4 | numerical_kernel | OPEN |  |
| thermal_transient | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::numerical::canonical_tet10_thermal_matrices_pass_constant_patch_and_capacity_integral` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/numerical.rs:72) - Single Tet10 conductivity constant patch, integrated capacity, linear gradient and flux; no general curved-mesh convergence. |

## constraints

Evidence kind: SemanticRealization. Criterion: Apply prescribed, tied and periodic temperature constraints and supported affine temperature jumps through the shared exact constraint authority. Steady components require physical anchoring or an explicitly supported nullspace treatment.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermal_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::steady::disconnected_bodies_have_independent_exact_steady_equilibria_and_require_each_anchor` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/steady.rs:7) - Three components with prescribed/convection anchoring; removal of an anchor rejects coercivity. Does not exercise tied, periodic or affine jumps. |
| thermal_steady | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::thermal::assembly::tests::steady::disconnected_bodies_have_independent_exact_steady_equilibria_and_require_each_anchor` (crates/runmat-analysis/fea/src/physics/thermal/assembly/tests/steady.rs:7) - Three components with prescribed/convection anchoring; removal of an anchor rejects coercivity. Does not exercise tied, periodic or affine jumps. |
| thermal_transient | tet4 | numerical_kernel | OPEN |  |
| thermal_transient | tet10 | numerical_kernel | OPEN |  |

