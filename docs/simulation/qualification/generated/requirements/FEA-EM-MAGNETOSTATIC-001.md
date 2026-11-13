# FEA-EM-MAGNETOSTATIC-001: Documented magnetostatic formulation and physical recovery

0 PASS, 4 PARTIAL, 0 FAIL, 0 OPEN, 0 TAGGED. Generated; do not edit.

## formulation

Evidence kind: IndependentReference. Criterion: Integrated magnetic operator and gauge enforce the documented physical equations and source compatibility, with exact edge/region identities and validated material restrictions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| magnetostatic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::magnetic::tests::square_uniform_current_has_independent_potential_field_and_energy_convergence` (crates/runmat-analysis/fea/src/physics/magnetic/tests.rs:84) - Affine Tet4 and Tet10 extruded unit square, uniform axial current and grounded boundary; independent Fourier-series potential, curl field and energy with tail b... |
| magnetostatic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::magnetic::tests::square_uniform_current_has_independent_potential_field_and_energy_convergence` (crates/runmat-analysis/fea/src/physics/magnetic/tests.rs:84) - Affine Tet4 and Tet10 extruded unit square, uniform axial current and grounded boundary; independent Fourier-series potential, curl field and energy with tail b... |

## fields

Evidence kind: PhysicalBalance. Criterion: Recover magnetic field, flux density and energy from the solved state, report physical residual, gauge residual, source work and divergence/flux closure; magnetic force needs separate formulation and accuracy evidence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| magnetostatic | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::magnetic::tests::sign_permeability_and_affine_geometry_order_preserve_physical_scaling` (crates/runmat-analysis/fea/src/physics/magnetic/tests.rs:148) - Affine Tet4 and Tet10 geometry with lowest-order edge basis: sign/permeability scaling against a Tet4 solve; Tet10 geometry is not higher-order magnetic approxi...<br>`runmat-analysis-fea[lib]::physics::magnetic::tests::recovered_curl_has_shared_face_normal_flux_continuity_and_global_flux_closure` (crates/runmat-analysis/fea/src/physics/magnetic/tests.rs:441) - Affine Tet4 shared-face normal curl continuity and global boundary flux closure; no general magnetic force recovery. |
| magnetostatic | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::magnetic::tests::sign_permeability_and_affine_geometry_order_preserve_physical_scaling` (crates/runmat-analysis/fea/src/physics/magnetic/tests.rs:148) - Affine Tet4 and Tet10 geometry with lowest-order edge basis: sign/permeability scaling against a Tet4 solve; Tet10 geometry is not higher-order magnetic approxi... |

