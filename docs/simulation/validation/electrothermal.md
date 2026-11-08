---
description: Reference checks for RunMat electrothermal analyses: each problem, its closed-form, manufactured or published reference, the compared quantity, the tolerance and the latest result.
keywords: [electrothermal validation, verification, reference solution, mesh convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Electrothermal validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each check compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. The tests listed under a check run on every change and in the evidence report; receipts record a public study run with its build, commit and mesh. The [validation methodology](./methodology.md) explains how checks are built and how tolerances are set, and [Status](./status.md) summarizes every family.

## Steady

### Two-way Joule heating of a bar

**Problem.** A bar with resistivity $\rho(T) = \rho_0(1 + \alpha(T - T_0))$ carries a fixed current with its ends held at $T_0$ and its sides insulated: $\sigma_0$ = 3 S/m, $\alpha$ = 0.004 /K, $k$ = 1 W/m/K, $I$ = 34.64 A on a unit cube, Tet10 with 2, 4 and 8 cells.

**Reference.** The temperature rise $\theta$ solves $\theta'' + \beta(1 + \alpha\theta) = 0$ with $\beta = J^2\rho_0/k$; closed form $\theta(x) = \frac{1}{\alpha}\left(\frac{\cos(m(x - L/2))}{\cos(mL/2)} - 1\right)$ with $m^2 = \alpha\beta$, and power $\int I^2\rho(T)/A\,dx$.

**Compared.** Nodal temperature; Joule power against terminal power and against $I^2R(T)$.

**Tolerance.** At least three coupling sweeps; temperature error falls at second order; terminal power equals the Joule integral within 1e-8; power within 0.5 % of $I^2R(T)$.

**Result.** Passed (1 test) in the evidence run of 2026-11-08. Largest temperature error 2.21, 0.575 and 0.150 K at 2, 4 and 8 cells, against a 59.95 K peak rise; Joule power 461.47, 462.88 and 463.35 W against 463.52 W.

Tests: `two_way_joule_heating_matches_the_closed_form_bar`.

### Joule heating of a slab, one way

**Problem.** A slab of constant conductivity dissipates 12 W, which drives a steady thermal solve on the same mesh; Tet4 and Tet10.

**Reference.** The quadratic temperature profile of uniform heat generation, and 12 W at the terminals, in the transfer and at the thermal boundary.

**Compared.** Terminal, transferred and boundary power; nodal temperature.

**Tolerance.** Power within 1e-8 W; Tet10 temperature within 1e-8; Tet4 interior error below 0.35 of the coarse mesh after one refinement.

**Result.** Passed (2 tests) in the evidence run of 2026-11-08.

Tests: `retained_conduction_drives_independent_quadratic_thermal_patch_and_power`, `tet4_refinement_and_voltage_sign_gauge_preserve_independent_heat_and_power`.
