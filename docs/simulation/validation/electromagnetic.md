---
description: Reference checks for RunMat electromagnetics analyses: each problem, its closed-form, manufactured or published reference, the compared quantity, the tolerance and the latest result.
keywords: [electromagnetics validation, verification, reference solution, mesh convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Electromagnetics validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each check compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. The tests listed under a check run on every change and in the evidence report; receipts record a public study run with its build, commit and mesh. The [validation methodology](./methodology.md) explains how checks are built and how tolerances are set, and [Status](./status.md) summarizes every family.

## Electrostatics and steady current

### Homogeneous slab capacitor and conductor

**Problem.** A homogeneous slab between two terminals, solved as a capacitor (electrostatics) and as a conductor (steady current) on Tet4 and Tet10 meshes.

**Reference.** Parallel-plate closed form: linear potential, uniform field, terminal charge or current, stored energy and Joule power.

**Compared.** Nodal potential, field, terminal flux, energy, Joule power.

**Tolerance.** Potential and field within 1e-8; flux imbalance and energy identity below 1e-8; energy and displacement field within relative 1e-8.

**Result.** 

### Two-material slab in series

**Problem.** A slab of two aligned materials between two terminals, on Tet4 and Tet10 meshes.

**Reference.** Series capacitance and series resistance, with continuous normal flux across the interface.

**Compared.** Potential, energy, Joule power, interface flux.

**Tolerance.** Relative 1e-8.

**Result.** 

### Harmonic polynomial potential

**Problem.** The harmonic potential $x^2 - y^2$, solved with both the electrostatic and the conduction equation; Tet10 on one mesh and Tet4 on three.

**Reference.** The exact field $(-2x, 2y, 0)$ and energy of the harmonic polynomial.

**Compared.** Field and energy.

**Tolerance.** Tet10: field and energy exact to relative 1e-8. Tet4: field error falls below 0.6 of the previous mesh at each refinement.

**Result.** 

### Uniformly charged dielectric slab

**Problem.** A dielectric slab with uniform volume charge on three Tet4 meshes.

**Reference.** Closed form: quadratic potential, linear field, conserved charge and energy $\varepsilon/6$ in the fixture's units.

**Compared.** Potential, field and energy errors; flux imbalance.

**Tolerance.** Per refinement: potential error order above 1.7, field order above 0.8, energy error below 0.35 of the previous mesh. Finest mesh: potential below 0.005, field below 0.1, energy below 0.025. Flux imbalance below 1e-9.

**Result.** Potential error 4.6e-2, 1.1e-2, 2.9e-3; field error 0.29, 0.14, 0.072; energy error 0.25, 0.063, 0.016.

### Steady current in a copper bar

**Problem.** A 1 m copper bar ($\sigma$ = 59.6 MS/m) with 1 mV across its ends; Tet10.

**Reference.** Uniform current density $J = \sigma V/L$ and Joule power $P = V^2/R$.

**Compared.** Current density; Joule power from the loss density and from the terminal current.

**Tolerance.** Relative 1e-9.

**Result.** Peak current density 59 600.000 A/m² against 59 600 A/m² (relative difference 1e-13); Joule power 0.596000 W against 0.596 W (1e-14).

## Magnetostatics

### Magnetic field of a square conductor

**Problem.** A square conductor carries a uniform axial current density and is grounded on every exterior face; Tet4 and Tet10 geometry with the lowest-order edge basis, three meshes per order.

**Reference.** An independent Fourier-series solution for the vector potential, flux density and energy; the series tail is bounded by $8/(3\pi^4N^3)$ at N = 255 terms.

**Compared.** Vector potential, flux density and magnetic energy.

**Tolerance.** Each error falls by more than 1.4× per refinement. Finest mesh: potential below 0.02, flux density below 0.10, energy below 0.25. Physical and gauge residuals below 8e-10.

**Result.** Flux-density error 0.166, 0.116, 0.064 and energy error 0.80, 0.39, 0.12 on 1, 2 and 4 cells.

## Harmonic

### Skin effect in a driven conductor

**Problem.** A conducting strip ($W$ = 1 mm, $L$ = 2 mm, $t$ = 1 mm, $\sigma$ = 1 MS/m) under an air gap $g$ = 0.5 mm, returned through a grounded plane, is driven by a 2 A terminal current at $t/\delta$ = 0.5, 2 and 4 with 8, 16 and 32 cells through the thickness and along the current.

**Reference.** Parallel-plate line impedance $Z = \frac{L}{W}\left[\frac{k\coth(kt)}{\sigma} + i\omega\mu_0 g\right]$ with $k = (1+i)/\delta$ and $\delta = \sqrt{2/\omega\mu_0\sigma}$.

**Compared.** Terminal impedance $Z = U/I$; Joule loss against delivered power; Poynting balance and charge continuity.

**Tolerance.** Impedance error below 2e-3 at 32 cells and reduced by more than 3× from 8 to 32 cells; Joule loss equals $\tfrac12\mathrm{Re}(U\bar I)$ within 1e-8; Poynting and charge-continuity residuals below 1e-8.

**Result.** Relative impedance error at 8, 16 and 32 cells: 4.2e-4, 1.0e-4, 2.6e-5 at $t/\delta$ = 0.5; 7.3e-3, 1.8e-3, 4.5e-4 at 2; 1.6e-2, 3.9e-3, 9.8e-4 at 4. The error falls fourfold per refinement.

### Low-frequency limit of a driven strip

**Problem.** The strip of the skin-effect check, voltage driven at 1 Hz.

**Reference.** DC resistance $L/(\sigma W t)$ and the low-frequency gap inductance.

**Compared.** Real and imaginary parts of the terminal impedance.

**Tolerance.** Resistance within relative 1e-6; reactance within relative 2e-2.

**Result.** 
