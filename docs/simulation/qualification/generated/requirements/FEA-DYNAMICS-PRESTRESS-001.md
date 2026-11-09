# FEA-DYNAMICS-PRESTRESS-001: Validated prestressed base states and stability perturbation

0 PASS, 4 PARTIAL, 0 FAIL, 108 OPEN, 0 TAGGED. Generated; do not edit.

## complete_base_operator

Evidence kind: IndependentReference. Criterion: Consume only a complete, converged structural base state from an explicit sequence transition. Support a documented initial envelope of linear static preload and selected converged nonlinear preload states for which a consistent tangent and geometric stiffness are available. Assemble and retain material, tangent, and geometric-stiffness contributions with separate identities and sign conventions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| prestressed_structural | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::tests::prestressed_frequency::prestressed_direct_harmonic_uses_the_exact_tangent_and_rejects_state_substitution` (crates/runmat-analysis/fea/src/tests/prestressed_frequency.rs:175) - Fixed direct entry fixture compares preload response ordering and residual, canonical result roundtrip, and substituted topology rejection. No independent buckl... |
| prestressed_structural | tet10 | numerical_kernel | OPEN |  |
| prestressed_structural | beam | numerical_kernel | OPEN |  |
| prestressed_structural | shell | numerical_kernel | OPEN |  |
| prestressed_structural | tet4 | public_study | OPEN |  |
| prestressed_structural | tet10 | public_study | OPEN |  |
| prestressed_structural | beam | public_study | OPEN |  |
| prestressed_structural | shell | public_study | OPEN |  |
| prestressed_structural | canonical_mixed_topology | native_process | OPEN |  |
| prestressed_structural | canonical_mixed_topology | durable | OPEN |  |
| prestressed_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| prestressed_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| prestressed_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| prestressed_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |
| structural_stability | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::tests::physical_preload_publishes_signed_load_linear_geometric_stiffness` (crates/runmat-analysis/fea/src/tests.rs:2041) - A physical preload publishes a signed-load linear geometric stiffness on a Tet4 fixture. |
| structural_stability | tet10 | numerical_kernel | OPEN |  |
| structural_stability | beam | numerical_kernel | OPEN |  |
| structural_stability | shell | numerical_kernel | OPEN |  |
| structural_stability | tet4 | public_study | OPEN |  |
| structural_stability | tet10 | public_study | OPEN |  |
| structural_stability | beam | public_study | OPEN |  |
| structural_stability | shell | public_study | OPEN |  |
| structural_stability | canonical_mixed_topology | native_process | OPEN |  |
| structural_stability | canonical_mixed_topology | durable | OPEN |  |
| structural_stability | canonical_mixed_topology | remote_execution | OPEN |  |
| structural_stability | canonical_mixed_topology | parallel_cpu | OPEN |  |
| structural_stability | canonical_mixed_topology | domain_decomposed | OPEN |  |
| structural_stability | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## buckling_modal_lineage

Evidence kind: IndependentReference. Criterion: Implement eigenvalue buckling as a linear perturbation about the base state, with typed load-pattern scaling and explicit treatment of fixed, proportional, follower, and unsupported loads. Report signed load factors, normalized buckling modes, eigen-residuals, clustered/repeated eigenspaces, base-load attribution, and the assumptions that prevent interpreting the result as nonlinear post-buckling ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| prestressed_structural | tet4 | numerical_kernel | OPEN |  |
| prestressed_structural | tet10 | numerical_kernel | OPEN |  |
| prestressed_structural | beam | numerical_kernel | OPEN |  |
| prestressed_structural | shell | numerical_kernel | OPEN |  |
| prestressed_structural | tet4 | public_study | OPEN |  |
| prestressed_structural | tet10 | public_study | OPEN |  |
| prestressed_structural | beam | public_study | OPEN |  |
| prestressed_structural | shell | public_study | OPEN |  |
| prestressed_structural | canonical_mixed_topology | native_process | OPEN |  |
| prestressed_structural | canonical_mixed_topology | durable | OPEN |  |
| prestressed_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| prestressed_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| prestressed_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| prestressed_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |
| structural_stability | tet4 | numerical_kernel | OPEN |  |
| structural_stability | tet10 | numerical_kernel | OPEN |  |
| structural_stability | beam | numerical_kernel | OPEN |  |
| structural_stability | shell | numerical_kernel | OPEN |  |
| structural_stability | tet4 | public_study | OPEN |  |
| structural_stability | tet10 | public_study | OPEN |  |
| structural_stability | beam | public_study | OPEN |  |
| structural_stability | shell | public_study | OPEN |  |
| structural_stability | canonical_mixed_topology | native_process | OPEN |  |
| structural_stability | canonical_mixed_topology | durable | OPEN |  |
| structural_stability | canonical_mixed_topology | remote_execution | OPEN |  |
| structural_stability | canonical_mixed_topology | parallel_cpu | OPEN |  |
| structural_stability | canonical_mixed_topology | domain_decomposed | OPEN |  |
| structural_stability | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## unsupported_linearization

Evidence kind: IndependentReference. Criterion: Permit prestressed direct or modal harmonic response only after both the base-state transition and the corresponding harmonic formulation are independently verified. Reject unsupported material history, changing contact status, follower-load linearization, topology change, loss of tangent symmetry, or unstable base states rather than silently reverting to an unstressed operator. Keep geometric imp...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| prestressed_structural | tet4 | numerical_kernel | OPEN |  |
| prestressed_structural | tet10 | numerical_kernel | OPEN |  |
| prestressed_structural | beam | numerical_kernel | OPEN |  |
| prestressed_structural | shell | numerical_kernel | OPEN |  |
| prestressed_structural | tet4 | public_study | OPEN |  |
| prestressed_structural | tet10 | public_study | OPEN |  |
| prestressed_structural | beam | public_study | OPEN |  |
| prestressed_structural | shell | public_study | OPEN |  |
| prestressed_structural | canonical_mixed_topology | native_process | OPEN |  |
| prestressed_structural | canonical_mixed_topology | durable | OPEN |  |
| prestressed_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| prestressed_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| prestressed_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| prestressed_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |
| structural_stability | tet4 | numerical_kernel | OPEN |  |
| structural_stability | tet10 | numerical_kernel | OPEN |  |
| structural_stability | beam | numerical_kernel | OPEN |  |
| structural_stability | shell | numerical_kernel | OPEN |  |
| structural_stability | tet4 | public_study | OPEN |  |
| structural_stability | tet10 | public_study | OPEN |  |
| structural_stability | beam | public_study | OPEN |  |
| structural_stability | shell | public_study | OPEN |  |
| structural_stability | canonical_mixed_topology | native_process | OPEN |  |
| structural_stability | canonical_mixed_topology | durable | OPEN |  |
| structural_stability | canonical_mixed_topology | remote_execution | OPEN |  |
| structural_stability | canonical_mixed_topology | parallel_cpu | OPEN |  |
| structural_stability | canonical_mixed_topology | domain_decomposed | OPEN |  |
| structural_stability | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## references

Evidence kind: IndependentReference. Criterion: Evidence includes column and frame buckling references, mesh convergence, sign and load-scaling cases, prestress-induced modal frequency shifts, tension stiffening, compression softening, repeated modes, tangent/geometric-stiffness checks, and sequence/restart equivalence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| prestressed_structural | tet4 | numerical_kernel | OPEN |  |
| prestressed_structural | tet10 | numerical_kernel | OPEN |  |
| prestressed_structural | beam | numerical_kernel | OPEN |  |
| prestressed_structural | shell | numerical_kernel | OPEN |  |
| prestressed_structural | tet4 | public_study | OPEN |  |
| prestressed_structural | tet10 | public_study | OPEN |  |
| prestressed_structural | beam | public_study | OPEN |  |
| prestressed_structural | shell | public_study | OPEN |  |
| prestressed_structural | canonical_mixed_topology | native_process | OPEN |  |
| prestressed_structural | canonical_mixed_topology | durable | OPEN |  |
| prestressed_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| prestressed_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| prestressed_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| prestressed_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |
| structural_stability | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::tests::physical_compressive_preload_buckling_has_exact_residual_and_inverse_load_scaling` (crates/runmat-analysis/fea/src/tests.rs:2134) - A single-tetrahedron compressive preload buckling problem has an exact residual and load factors that scale inversely with load; column and frame buckling, mesh... |
| structural_stability | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::tests::column_buckling::fixed_free_column_buckles_at_the_euler_load_with_converged_eigenpairs` (crates/runmat-analysis/fea/src/tests/column_buckling.rs:116) - A fixed-free Tet10 solid column (1 m, 100 x 50 mm, 250 kN) converges its buckling eigenpairs to the authored relative tolerance and matches Euler's fixed-free l... |
| structural_stability | beam | numerical_kernel | OPEN |  |
| structural_stability | shell | numerical_kernel | OPEN |  |
| structural_stability | tet4 | public_study | OPEN |  |
| structural_stability | tet10 | public_study | OPEN |  |
| structural_stability | beam | public_study | OPEN |  |
| structural_stability | shell | public_study | OPEN |  |
| structural_stability | canonical_mixed_topology | native_process | OPEN |  |
| structural_stability | canonical_mixed_topology | durable | OPEN |  |
| structural_stability | canonical_mixed_topology | remote_execution | OPEN |  |
| structural_stability | canonical_mixed_topology | parallel_cpu | OPEN |  |
| structural_stability | canonical_mixed_topology | domain_decomposed | OPEN |  |
| structural_stability | canonical_mixed_topology | claimed_accelerator | OPEN |  |

