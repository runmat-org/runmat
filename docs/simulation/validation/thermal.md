---
description: Reference checks for RunMat heat transfer analyses: each problem, its closed-form, manufactured or published reference, the compared quantity, the tolerance and the latest result.
keywords: [heat transfer validation, verification, reference solution, mesh convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Heat transfer validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each check compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. The tests listed under a check run on every change and in the evidence report; receipts record a public study run with its build, commit and mesh. The [validation methodology](./methodology.md) explains how checks are built and how tolerances are set, and [Status](./status.md) summarizes every family.

## Steady

### Two-layer slab in series

**Problem.** A slab of two half-meter layers with k = 1 and 4 W/(m·K), held at fixed temperatures on its outer faces; Tet4.

**Reference.** One-dimensional series thermal resistance: piecewise-linear temperature and a uniform flux of 160 W/m².

**Compared.** Nodal temperature; element heat flux; dependence on the order in which the materials are authored.

**Tolerance.** Temperature within 1e-7 K; flux within 1e-6 W/m²; reordering the materials changes the result by less than 1e-10.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `two_material_slab_matches_series_resistance_and_authored_order_is_invariant`.

### Bar between two fixed temperatures

**Problem.** A bar held at a hot and a cold end temperature with insulated sides.

**Reference.** Fourier's law through the bar, $Q = kA\,\Delta T/L$.

**Compared.** Heat entering and leaving through the prescribed ends; net prescribed reaction.

**Tolerance.** Heat flow within relative 1e-8 of $kA\,\Delta T/L$; the net reaction cancels to 1e-8 of it.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `bar_between_two_fixed_temperatures_reports_its_closed_form_heat_flow`.

### Tet10 conduction patch test

**Problem.** A single quadratic tetrahedron under a linear temperature field.

**Reference.** Constant-gradient patch: conductivity rows sum to zero, the gradient is exactly (2, 3, −1), the flux is $-k\nabla T$, and the integrated capacity equals $\rho c_p V$.

**Compared.** Conductivity row sums, temperature gradient, heat flux, integrated capacity.

**Tolerance.** Relative 1e-12.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `canonical_tet10_thermal_matrices_pass_constant_patch_and_capacity_integral`.

### Radiating bar

**Problem.** A 1 m steel bar ($k$ = 45 W/m/K) takes 1000 W/m² on one end and radiates from the other ($\varepsilon$ = 0.8, 300 K surroundings). The public example is [radiating-bar.fea](../examples/radiating-bar.fea); the test runs the same bar on Tet4 and Tet10 meshes.

**Reference.** Gray-surface radiation at the radiating end, $q = \varepsilon\sigma_{SB}(T^4 - T_\infty^4)$, and conduction $qL/k$ across the bar.

**Compared.** Temperatures at both ends and along the bar; applied and radiated power.

**Tolerance.** Nodal temperature within relative 1e-8 of the closed form; applied power equals radiated power.

**Result.** Passed (1 test) in the evidence run of 2026-11-08. RunMat gives 416.6790 K at the radiating end and 438.9013 K at the heated end; the closed form gives the same values to 1e-12. 10 W applied, 10 W radiated.

Tests: `radiating_bar_matches_the_gray_surface_closed_form_and_balances_power`.

### Slab radiating from a fixed hot face

**Problem.** A slab held at 400 K on one face radiates from the opposite face; Tet4 and Tet10.

**Reference.** The scalar gray-surface equilibrium of the slab, solved independently by bisection.

**Compared.** Nodal temperature, gradient and flux; radiated power against conducted power.

**Tolerance.** Temperature, gradient and flux within relative 1e-8; radiated power equals the prescribed-face power within 1e-8.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `fixed_hot_slab_matches_independent_gray_surface_scalar_equilibrium`.

## Transient

### Uniformly heated insulated body

**Problem.** An insulated body with uniform volumetric heating, integrated with backward Euler on Tet4 and Tet10 meshes.

**Reference.** The continuum temperature history of the heated body and its stored-energy law: storage rate equals applied power.

**Compared.** Temperature at every step; applied, stored and boundary power.

**Tolerance.** Discrete solution within 1e-8; error equals the backward-Euler term $B\,t_f\,\Delta t$ within 1e-8 and falls at first order (ratio below 0.56 per halving); storage rate equals applied power within 1e-8; energy residual below 1e-10.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `uniform_ramp_heating_has_continuum_backward_euler_time_convergence`.

### Manufactured radiative cooling

**Problem.** A body cooling by surface radiation under a manufactured solution, on Tet4 and Tet10 meshes with 8, 16 and 32 backward-Euler steps.

**Reference.** The manufactured temperature history and its gradient, flux and stored heat.

**Compared.** Temperature, gradient, heat flux and storage; power closure.

**Tolerance.** Every error falls by more than 1.8× per halving of the step (first order); power closes.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `manufactured_continuum_cooling_retains_first_order_backward_euler_and_power_closure`.
