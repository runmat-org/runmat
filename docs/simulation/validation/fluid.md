---
description: Reference checks for RunMat fluid flow analyses: each problem, its closed-form, manufactured or published reference, the compared quantity, the tolerance and the latest result.
keywords: [fluid flow validation, verification, reference solution, mesh convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Fluid flow validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each check compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. The tests listed under a check run on every change and in the evidence report; receipts record a public study run with its build, commit and mesh. The [validation methodology](./methodology.md) explains how checks are built and how tolerances are set, and [Status](./status.md) summarizes every family.

## Steady

### Plane Poiseuille pressure drop

**Problem.** Fully developed laminar flow between parallel plates, refined in the kernel test. The public example has L/h = 30 at Re = 1.

**Reference.** Fully developed pressure drop $\Delta p = 12\mu\bar UL/h^2$; 36 Pa for the public example. (F. M. White, *Viscous Fluid Flow*, 3rd ed., McGraw-Hill, 2006.)

**Compared.** Difference of the face-mean pressure between inlet and outlet.

**Tolerance.** Kernel: relative error below 5 % and decreasing with refinement. Public example: between 34.2 and 37.8 Pa (5 %), which contains the uniform inlet's entrance loss of about +3 %.

**Result.** Passed (2 tests) in the evidence run of 2026-11-08.

Tests: `pressure_drop_of_plane_poiseuille_flow_is_twelve_mu_u_l_over_h_squared`, `plane_poiseuille_example_publishes_the_analytic_pressure_drop`.

### Couette flow

**Problem.** Flow between a moving and a fixed wall on canonical tetrahedra.

**Reference.** The linear Couette profile, its uniform vorticity and its wall shear stress $\mu U/h$. (F. M. White, *Viscous Fluid Flow*, 3rd ed., McGraw-Hill, 2006.)

**Compared.** Velocity field, wall shear on both walls, vorticity, global momentum.

**Tolerance.** Velocity relative L2 error below 0.25; wall shear within 0.25; global momentum below 1e-6. For the exact linear field, vorticity and wall shear within 1e-6.

**Result.** Passed (2 tests) in the evidence run of 2026-11-08.

Tests: `affine_couette_uses_original_geometry_and_balanced_physical_fields`, `couette_vorticity_and_wall_shear_match_the_linear_profile`.

### Fully developed flow in a rectangular duct

**Problem.** Pressure-driven laminar flow in a rectangular duct, three refinements.

**Reference.** Fourier-series solution for the velocity, flow rate and wall shear; the series tail is below 1e-5 at 255 terms. (F. M. White, *Viscous Fluid Flow*, 3rd ed., McGraw-Hill, 2006.)

**Compared.** Velocity L2 error, flow rate, wall shear.

**Tolerance.** Each error decreases with refinement; finest velocity error below 0.35 and flow-rate error below 0.25; boundary flux below 1e-8.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `rectangular_duct_compares_fourier_reference_with_controlled_tail`.

### Uniform flow through a porous zone

**Problem.** Uniform flow through a porous slab.

**Reference.** Darcy–Forchheimer pressure gradient $\mu U/K + \tfrac12\rho C_2U^2$. (D. A. Nield and A. Bejan, *Convection in Porous Media*, 5th ed., Springer, 2017.)

**Compared.** Velocity and pressure gradient.

**Tolerance.** Relative 1e-6.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `uniform_flow_through_a_porous_zone_has_the_darcy_forchheimer_gradient`.

### Shear-thinning channel flow

**Problem.** Pressure-driven channel flow of a power-law fluid with n = 0.5, three refinements.

**Reference.** The closed-form power-law channel velocity profile.

**Compared.** Velocity profile error.

**Tolerance.** Error decreases with refinement and is below 5 % on the finest mesh.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `shear_thinning_power_law_channel_converges_to_the_closed_form_profile`.

## Transient

### Uniformly accelerating flow

**Problem.** A uniform flow accelerated at a constant rate, at two reference pressures.

**Reference.** The exact inertia-driven pressure gradient $-\rho\,dU/dt$.

**Compared.** Velocity and pressure gradient.

**Tolerance.** Velocity within 1e-8; pressure gradient within 1e-6.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `original_uniform_acceleration_has_exact_inertia_pressure_and_cold_binding`.

### Exponentially decaying translation

**Problem.** A uniform translation decaying as $e^{-t}$, solved with three time steps.

**Reference.** The exact pressure gradient of the decaying translation.

**Compared.** Volume-weighted pressure-gradient error.

**Tolerance.** Error falls below 0.6 of its previous value at each halving of the step (first order) and is below 0.02 at the finest step.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `exponential_translation_pressure_shows_first_order_time_convergence`.
