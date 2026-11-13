# FEA-FLUID-MATERIALS-001: Rheology, porous media and species transport

0 PASS, 2 PARTIAL, 0 FAIL, 14 OPEN, 0 TAGGED. Generated; do not edit.

## generalized_newtonian

Evidence kind: IndependentReference. Criterion: Power law, Carreau and Carreau-Yasuda, Cross, Casson, regularized Bingham and Herschel-Bulkley with Arrhenius and WLF temperature dependence run, publish shear-rate and apparent-viscosity fields, reproduce the power-law channel profile and the Buckingham-Reiner pipe flow rate.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| generalized_newtonian_flow | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::rheology::shear_thinning_power_law_channel_converges_to_the_closed_form_profile` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/rheology.rs:20) - shear-thinning power-law (n = 0.5) channel converges to the closed-form profile; Carreau, Cross and yield-stress laws, temperature dependence and the shear-rate... |
| generalized_newtonian_flow | tet4 | public_study | OPEN |  |
| generalized_newtonian_flow | tet10 | numerical_kernel | OPEN |  |
| generalized_newtonian_flow | tet10 | public_study | OPEN |  |

## viscoelastic

Evidence kind: IndependentReference. Criterion: Oldroyd-B, Giesekus and PTT with log-conformation run, and the Oldroyd-B cylinder-in-channel drag against Weissenberg number matches Alves et al.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| viscoelastic_flow | tet4 | numerical_kernel | OPEN |  |
| viscoelastic_flow | tet4 | public_study | OPEN |  |
| viscoelastic_flow | tet10 | numerical_kernel | OPEN |  |
| viscoelastic_flow | tet10 | public_study | OPEN |  |

## porous_zones

Evidence kind: IndependentReference. Criterion: Darcy, Brinkman and Forchheimer porous zones with anisotropic permeability and local thermal equilibrium run: Darcy u = (K/mu) G, the Brinkman channel profile and the Ergun packed-bed pressure drop; the filter study reports its housing-plus-cartridge pressure drop.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| porous_media_flow | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::cfd::canonical::tests::porous::uniform_flow_through_a_porous_zone_has_the_darcy_forchheimer_gradient` (crates/runmat-analysis/fea/src/physics/cfd/canonical/tests/porous.rs:11) - uniform flow through a porous slab has the pressure gradient μU/K + ½ρC₂U² (Darcy and Forchheimer terms); Brinkman layers, anisotropy, local thermal equilibrium... |
| porous_media_flow | tet4 | public_study | OPEN |  |
| porous_media_flow | tet10 | numerical_kernel | OPEN |  |
| porous_media_flow | tet10 | public_study | OPEN |  |

## species_and_scalars

Evidence kind: IndependentReference. Criterion: Passive scalar advection-diffusion, mass-action and Arrhenius reactions, residence time and a mixing index run: Graetz-type mass transfer and laminar tubular reactor conversion match their closed forms; the static mixer reports outlet concentration coefficient of variation and pressure drop.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| scalar_transport | tet4 | numerical_kernel | OPEN |  |
| scalar_transport | tet4 | public_study | OPEN |  |
| scalar_transport | tet10 | numerical_kernel | OPEN |  |
| scalar_transport | tet10 | public_study | OPEN |  |

