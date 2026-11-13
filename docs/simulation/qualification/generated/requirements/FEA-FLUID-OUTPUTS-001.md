# FEA-FLUID-OUTPUTS-001: Flow outputs engineers ask for

0 PASS, 3 PARTIAL, 0 FAIL, 49 OPEN, 0 TAGGED. Generated; do not edit.

## forces_and_coefficients

Evidence kind: IndependentReference. Criterion: Forces, moments and drag and lift coefficients on named surfaces reproduce sphere Stokes drag 6 pi mu a U and C_D about 1.09 at Re 100, and cylinder C_D about 1.52 and wake length L/D about 2.3 at Re 40.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet4 | public_study | OPEN |  |
| incompressible_laminar_steady | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet10 | public_study | OPEN |  |
| incompressible_laminar_transient | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet4 | public_study | OPEN |  |
| incompressible_laminar_transient | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet10 | public_study | OPEN |  |
| conjugate_heat_transfer | tet4 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet4 | public_study | OPEN |  |
| conjugate_heat_transfer | tet10 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet10 | public_study | OPEN |  |

## wall_shear_stress_fields

Evidence kind: PhysicalBalance. Criterion: A wall shear stress field, time-averaged wall shear stress and the oscillatory shear index publish for pulsatile flow; the carotid study reports minimum time-averaged wall shear stress in the sinus.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::couette_vorticity_and_wall_shear_match_the_linear_profile` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests.rs:560) - Couette vorticity and wall shear stress equal the linear profile's; time-averaged WSS and the oscillatory shear index are not covered. |
| incompressible_laminar_steady | tet4 | public_study | OPEN |  |
| incompressible_laminar_steady | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet10 | public_study | OPEN |  |
| incompressible_laminar_transient | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet4 | public_study | OPEN |  |
| incompressible_laminar_transient | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet10 | public_study | OPEN |  |
| conjugate_heat_transfer | tet4 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet4 | public_study | OPEN |  |
| conjugate_heat_transfer | tet10 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet10 | public_study | OPEN |  |

## pressure_drop_and_flow_rate

Evidence kind: PhysicalBalance. Criterion: `pressure_drop` between named faces and flow rate through named faces publish and satisfy mass conservation to the declared tolerance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::execution::fluid::tests::pressure_drop::pressure_drop_of_plane_poiseuille_flow_is_twelve_mu_u_l_over_h_squared` (crates/runmat-analysis/fea/src/execution/fluid/tests/pressure_drop.rs:180) - the face-pressure mean difference between the inlet and outlet of plane Poiseuille flow approaches (−dp/dx)·L = 12 µ Ū L / h² under refinement; developing and t... |
| incompressible_laminar_steady | tet4 | public_study | PARTIAL | `runmat[test:check_cli]::plane_poiseuille_example_publishes_the_analytic_pressure_drop` (crates/runmat-cli/tests/check_cli.rs:1182) - pressure_drop on the public plane Poiseuille example (L/h = 30, Re = 1) lies within 5% of the fully developed 12 mu U L / h^2 = 36 Pa; the uniform inlet's entra... |
| incompressible_laminar_steady | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet10 | public_study | OPEN |  |
| incompressible_laminar_transient | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet4 | public_study | OPEN |  |
| incompressible_laminar_transient | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet10 | public_study | OPEN |  |
| conjugate_heat_transfer | tet4 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet4 | public_study | OPEN |  |
| conjugate_heat_transfer | tet10 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet10 | public_study | OPEN |  |

## heat_transfer_quantities

Evidence kind: PhysicalBalance. Criterion: Nusselt number, heat-transfer coefficient and bulk temperature publish for heated walls and agree with the energy balance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| conjugate_heat_transfer | tet4 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet4 | public_study | OPEN |  |
| conjugate_heat_transfer | tet10 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet10 | public_study | OPEN |  |

## vorticity_and_streamlines

Evidence kind: PublicationIntegrity. Criterion: Vorticity, Q-criterion and nodal velocity (for streamlines) publish with the declared shapes and locations.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet4 | public_study | OPEN |  |
| incompressible_laminar_steady | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet10 | public_study | OPEN |  |
| incompressible_laminar_transient | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet4 | public_study | OPEN |  |
| incompressible_laminar_transient | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet10 | public_study | OPEN |  |
| conjugate_heat_transfer | tet4 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet4 | public_study | OPEN |  |
| conjugate_heat_transfer | tet10 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet10 | public_study | OPEN |  |

