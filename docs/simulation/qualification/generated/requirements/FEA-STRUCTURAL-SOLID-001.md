# FEA-STRUCTURAL-SOLID-001: Physical linear solid operators, loads and assigned-material recovery

0 PASS, 7 PARTIAL, 0 FAIL, 49 OPEN, 0 TAGGED. Generated; do not edit.

## quadrature_mass

Evidence kind: IndependentReference. Criterion: Complete Tet4 verification and retain it as the initial serial reference. Complete Tet10 volume and face quadrature, recovery, stress extrapolation/projection, pressure loading, and estimator support. Assemble consistent mass and a documented optional lumped mass from density and geometry.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::topology::tests::oblique_both_order_volume_quadrature_reproduces_all_world_affine_gradients` (crates/runmat-analysis/fea/src/topology/tests.rs:10) - Oblique Tet4 and Tet10 volume quadrature reproduces all world-affine gradients; mass matrices and face quadrature are other tests. |
| linear_structural | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::assembly::elements::solid::tetrahedron10_mass::tests::affine_mass_preserves_all_quadratic_kinetic_cross_moments_at_scale_and_translation` (crates/runmat-analysis/fea/src/assembly/elements/solid/tetrahedron10_mass/tests.rs:59) - Affine Tet10 consistent mass preserves independent quadratic kinetic cross moments across scale/translation. No lumped mass, body-load, damping or mesh-converge...<br>`runmat-analysis-fea[lib]::assembly::elements::solid::tetrahedron10_mass::tests::reciprocal_density_and_geometry_scales_preserve_physical_rigid_translation_mass` (crates/runmat-analysis/fea/src/assembly/elements/solid/tetrahedron10_mass/tests.rs:154) - Tet10 consistent mass preserves physical rigid-translation mass under reciprocal density and geometry scaling.<br>`runmat-analysis-fea[lib]::topology::tests::oblique_both_order_volume_quadrature_reproduces_all_world_affine_gradients` (crates/runmat-analysis/fea/src/topology/tests.rs:10) - Oblique Tet4 and Tet10 volume quadrature reproduces all world-affine gradients; mass matrices and face quadrature are other tests. |
| linear_structural | beam | numerical_kernel | OPEN |  |
| linear_structural | shell | numerical_kernel | OPEN |  |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |
| linear_structural | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural | canonical_mixed_topology | durable | OPEN |  |
| linear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## damping_body_loads

Evidence kind: IndependentReference. Criterion: Implement documented damping inputs, initially Rayleigh damping where applicable. Integrate body forces by region, material density, shape function, and volume. Implement nonzero prescribed displacement with correct elimination or lifting and reaction recovery.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::assembly::tests::tet10_body_force_quadrature_preserves_integrated_volume_resultant` (crates/runmat-analysis/fea/src/assembly/mod.rs:3653) - Tet10 body-force quadrature preserves the integrated volume resultant. |
| linear_structural | beam | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::assembly::tests::explicit_body_force_density_integrates_beam_and_shell_volume` (crates/runmat-analysis/fea/src/assembly/mod.rs:4132) - An explicit body-force density integrates over beam and shell volumes. |
| linear_structural | shell | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::assembly::tests::shell_assembles_authored_rayleigh_damping` (crates/runmat-analysis/fea/src/assembly/mod.rs:3100) - A shell assembles authored Rayleigh damping; solid damping is not asserted here.<br>`runmat-analysis-fea[lib]::assembly::tests::explicit_body_force_density_integrates_beam_and_shell_volume` (crates/runmat-analysis/fea/src/assembly/mod.rs:4132) - An explicit body-force density integrates over beam and shell volumes. |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |
| linear_structural | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural | canonical_mixed_topology | durable | OPEN |  |
| linear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## material_recovery

Evidence kind: IndependentReference. Criterion: Make stress, strain, energy, and derived-field recovery use each element's assigned material. Define treatment of discontinuous stress at material interfaces. Preserve exact force and moment resultants for distributed surface and wrench loads.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::post::solid::tests::recovery_uses_each_elements_assigned_material` (crates/runmat-analysis/fea/src/post/solid.rs:470) - Two Tet4 elements with different Young's moduli recover different stresses from the same strain, in serial and with two threads; Tet10 recovery is covered by th... |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | beam | numerical_kernel | OPEN |  |
| linear_structural | shell | numerical_kernel | OPEN |  |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |
| linear_structural | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural | canonical_mixed_topology | durable | OPEN |  |
| linear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## load_excitation_diagnostics

Evidence kind: IndependentReference. Criterion: Realize constant and parameterized static excitations through the shared signal/excitation path rather than a formulation-private load vector. Detect rigid modes, inadequate constraint, inversion, poor conditioning, and unsupported element types with typed diagnostics.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | beam | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::tests::large_load_sweep_does_not_fabricate_dofs_from_excitation_count` (crates/runmat-analysis/fea/src/tests.rs:4767) - A large load sweep does not fabricate degrees of freedom from the excitation count; diagnostic completeness is not asserted. |
| linear_structural | shell | numerical_kernel | OPEN |  |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |
| linear_structural | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural | canonical_mixed_topology | durable | OPEN |  |
| linear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

