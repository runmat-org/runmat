# FEA-FLUID-HEAT-001: Heat in the fluid: forced and natural convection, CHT and low-Mach flow

0 PASS, 0 PARTIAL, 1 FAIL, 23 OPEN, 0 TAGGED. Generated; do not edit.

## forced_convection

Evidence kind: IndependentReference. Criterion: Fluid energy transport reproduces the pipe Nusselt numbers 3.66 (constant wall temperature) and 4.364 (constant heat flux), and the Graetz thermal entry length.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| conjugate_heat_transfer | tet4 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet4 | public_study | OPEN |  |
| conjugate_heat_transfer | tet10 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet10 | public_study | OPEN |  |

## natural_convection

Evidence kind: IndependentReference. Criterion: Two-way Boussinesq natural convection reproduces the de Vahl Davis differentially heated cavity Nusselt numbers 1.118, 2.243, 4.519 and 8.800 at Ra 1e3 to 1e6 within 1 percent, the Tric cube values 1.0700, 2.0542, 4.3371 and 8.6407, and Rayleigh-Benard onset at Ra_c 1707.76; `conjugate_heat_transfer steady { buoyancy = boussinesq; gravity = ... }` runs the sealed enclosure study.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| conjugate_heat_transfer | tet4 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet4 | public_study | OPEN |  |
| conjugate_heat_transfer | tet10 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet10 | public_study | OPEN |  |

## viscous_heating_and_properties

Evidence kind: PhysicalBalance. Criterion: Viscous heating and temperature-dependent density, viscosity, conductivity and specific heat run and close the energy balance to the declared tolerance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| conjugate_heat_transfer | tet4 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet4 | public_study | OPEN |  |
| conjugate_heat_transfer | tet10 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet10 | public_study | OPEN |  |

## two_way_cht

Evidence kind: PhysicalBalance. Criterion: Two-way conjugate heat transfer with thermal solids converges with conservative interface flux continuity (residual published) and reproduces series conduction and plug-flow references; the liquid cold plate study reports pressure drop and peak plate temperature at 0.03 kg/s and 1.2 kW.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| conjugate_heat_transfer | tet4 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet4 | public_study | FAIL | `runmat[test:check_cli]::cht_plate_channel_example_runs_as_a_coupled_study` (crates/runmat-cli/tests/check_cli.rs:1223) - One public two-way CHT example on a linear mesh converges. |
| conjugate_heat_transfer | tet10 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet10 | public_study | OPEN |  |

## radiation_coupling

Evidence kind: PhysicalBalance. Criterion: Surface radiation couples to the fluid and solid energy balances with emissivity and ambient or enclosure temperature and closes the balance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| conjugate_heat_transfer | tet4 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet4 | public_study | OPEN |  |
| conjugate_heat_transfer | tet10 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet10 | public_study | OPEN |  |

## low_mach_variable_density

Evidence kind: IndependentReference. Criterion: Low-Mach flow with density from temperature and pressure reproduces the heated vertical channel variable-density closed form.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| low_mach_flow | tet4 | numerical_kernel | OPEN |  |
| low_mach_flow | tet4 | public_study | OPEN |  |
| low_mach_flow | tet10 | numerical_kernel | OPEN |  |
| low_mach_flow | tet10 | public_study | OPEN |  |

