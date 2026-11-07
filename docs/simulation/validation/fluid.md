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

**Reference.** Fully developed pressure drop $\Delta p = 12\mu\bar UL/h^2$; 36 Pa for the public example.

**Compared.** Difference of the face-mean pressure between inlet and outlet.

**Tolerance.** Kernel: relative error below 5 % and decreasing with refinement. Public example: between 34.2 and 37.8 Pa (5 %), which contains the uniform inlet's entrance loss of about +3 %.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Couette flow

**Problem.** Flow between a moving and a fixed wall on canonical tetrahedra.

**Reference.** The linear Couette profile, its uniform vorticity and its wall shear stress $\mu U/h$.

**Compared.** Velocity field, wall shear on both walls, vorticity, global momentum.

**Tolerance.** Velocity relative L2 error below 0.25; wall shear within 0.25; global momentum below 1e-6. For the exact linear field, vorticity and wall shear within 1e-6.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Fully developed flow in a rectangular duct

**Problem.** Pressure-driven laminar flow in a rectangular duct, three refinements.

**Reference.** Fourier-series solution for the velocity, flow rate and wall shear; the series tail is below 1e-5 at 255 terms.

**Compared.** Velocity L2 error, flow rate, wall shear.

**Tolerance.** Each error decreases with refinement; finest velocity error below 0.35 and flow-rate error below 0.25; boundary flux below 1e-8.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Uniform flow through a porous zone

**Problem.** Uniform flow through a porous slab.

**Reference.** Darcy–Forchheimer pressure gradient $\mu U/K + \tfrac12\rho C_2U^2$.

**Compared.** Velocity and pressure gradient.

**Tolerance.** Relative 1e-6.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Shear-thinning channel flow

**Problem.** Pressure-driven channel flow of a power-law fluid with n = 0.5, three refinements.

**Reference.** The closed-form power-law channel velocity profile.

**Compared.** Velocity profile error.

**Tolerance.** Error decreases with refinement and is below 5 % on the finest mesh.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

## Transient

### Uniformly accelerating flow

**Problem.** A uniform flow accelerated at a constant rate, at two reference pressures.

**Reference.** The exact inertia-driven pressure gradient $-\rho\,dU/dt$.

**Compared.** Velocity and pressure gradient.

**Tolerance.** Velocity within 1e-8; pressure gradient within 1e-6.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Exponentially decaying translation

**Problem.** A uniform translation decaying as $e^{-t}$, solved with three time steps.

**Reference.** The exact pressure gradient of the decaying translation.

**Compared.** Volume-weighted pressure-gradient error.

**Tolerance.** Error falls below 0.6 of its previous value at each halving of the step (first order) and is below 0.02 at the finest step.

**Result.** Not yet in the evidence catalog; run `evidence_report`.
