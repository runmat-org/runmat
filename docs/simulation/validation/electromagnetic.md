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

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `homogeneous_capacitor_and_conductor_match_independent_slab_field_terminals_and_energy`.

### Two-material slab in series

**Problem.** A slab of two aligned materials between two terminals, on Tet4 and Tet10 meshes.

**Reference.** Series capacitance and series resistance, with continuous normal flux across the interface.

**Compared.** Potential, energy, Joule power, interface flux.

**Tolerance.** Relative 1e-8.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `layered_materials_match_series_capacitance_and_resistance_with_interface_flux_continuity`.

### Harmonic polynomial potential

**Problem.** The harmonic potential $x^2 - y^2$, solved with both the electrostatic and the conduction equation; Tet10 on one mesh and Tet4 on three.

**Reference.** The exact field $(-2x, 2y, 0)$ and energy of the harmonic polynomial.

**Compared.** Field and energy.

**Tolerance.** Tet10: field and energy exact to relative 1e-8. Tet4: field error falls below 0.6 of the previous mesh at each refinement.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `harmonic_quadratic_potential_has_tet10_exact_field_and_tet4_convergence`.

### Uniformly charged dielectric slab

**Problem.** A dielectric slab with uniform volume charge on three Tet4 meshes.

**Reference.** Closed form: quadratic potential, linear field, conserved charge and energy $\varepsilon/6$ in the fixture's units.

**Compared.** Potential, field and energy errors; flux imbalance.

**Tolerance.** Per refinement: potential error order above 1.7, field order above 0.8, energy error below 0.35 of the previous mesh. Finest mesh: potential below 0.005, field below 0.1, energy below 0.025. Flux imbalance below 1e-9.

**Result.** Passed (1 test) in the evidence run of 2026-11-08. Potential error 4.6e-2, 1.1e-2, 2.9e-3; field error 0.29, 0.14, 0.072; energy error 0.25, 0.063, 0.016.

Tests: `tet4_charged_slab_has_independent_spatial_field_and_energy_convergence`.

### Steady current in a copper bar

**Problem.** A 1 m copper bar ($\sigma$ = 59.6 MS/m) with 1 mV across its ends; Tet10.

**Reference.** Uniform current density $J = \sigma V/L$ and Joule power $P = V^2/R$.

**Compared.** Current density; Joule power from the loss density and from the terminal current.

**Tolerance.** Relative 1e-9.

**Result.** Passed (1 test) in the evidence run of 2026-11-08. Peak current density 59 600.000 A/m² against 59 600 A/m² (relative difference 1e-13); Joule power 0.596000 W against 0.596 W (1e-14).

Tests: `copper_bar_carries_sigma_v_over_l_and_dissipates_v_squared_over_r`.

## Magnetostatics

### Magnetic field of a square conductor

**Problem.** A square conductor carries a uniform axial current density and is grounded on every exterior face; Tet4 and Tet10 geometry with the lowest-order edge basis, three meshes per order.

**Reference.** An independent Fourier-series solution for the vector potential, flux density and energy; the series tail is bounded by $8/(3\pi^4N^3)$ at N = 255 terms.

**Compared.** Vector potential, flux density and magnetic energy.

**Tolerance.** Each error falls by more than 1.4× per refinement. Finest mesh: potential below 0.02, flux density below 0.10, energy below 0.25. Physical and gauge residuals below 8e-10.

**Result.** Passed (1 test) in the evidence run of 2026-11-08. Flux-density error 0.166, 0.116, 0.064 and energy error 0.80, 0.39, 0.12 on 1, 2 and 4 cells.

Tests: `square_uniform_current_has_independent_potential_field_and_energy_convergence`.

## Harmonic

### Skin effect in a driven conductor

**Problem.** A conducting strip ($W$ = 1 mm, $L$ = 2 mm, $t$ = 1 mm, $\sigma$ = 1 MS/m) under an air gap $g$ = 0.5 mm, returned through a grounded plane, is driven by a 2 A terminal current at $t/\delta$ = 0.5, 2 and 4 with 8, 16 and 32 cells through the thickness and along the current.

**Reference.** Parallel-plate line impedance $Z = \frac{L}{W}\left[\frac{k\coth(kt)}{\sigma} + i\omega\mu_0 g\right]$ with $k = (1+i)/\delta$ and $\delta = \sqrt{2/\omega\mu_0\sigma}$. (S. Ramo, J. R. Whinnery and T. Van Duzer, *Fields and Waves in Communication Electronics*, 3rd ed., Wiley, 1994.)

**Compared.** Terminal impedance $Z = U/I$; Joule loss against delivered power; Poynting balance and charge continuity.

**Tolerance.** Impedance error below 2e-3 at 32 cells and reduced by more than 3× from 8 to 32 cells; Joule loss equals $\tfrac12\mathrm{Re}(U\bar I)$ within 1e-8; Poynting and charge-continuity residuals below 1e-8.

**Result.** Passed (1 test) in the evidence run of 2026-11-08. Relative impedance error at 8, 16 and 32 cells: 4.2e-4, 1.0e-4, 2.6e-5 at $t/\delta$ = 0.5; 7.3e-3, 1.8e-3, 4.5e-4 at 2; 1.6e-2, 3.9e-3, 9.8e-4 at 4. The error falls fourfold per refinement.

Tests: `current_driven_strip_impedance_converges_to_the_skin_effect_solution`.

### Low-frequency limit of a driven strip

**Problem.** The strip of the skin-effect check, voltage driven at 1 Hz.

**Reference.** DC resistance $L/(\sigma W t)$ and the low-frequency gap inductance.

**Compared.** Real and imaginary parts of the terminal impedance.

**Tolerance.** Resistance within relative 1e-6; reactance within relative 2e-2.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `voltage_drive_and_low_frequency_limit_match_dc_resistance_and_inductance`.

## Transmission-line parameters

### Microstrip impedance against Hammerstad–Jensen

**Problem.** A zero-thickness surface microstrip on a 0.2 mm substrate over a reference plane, open above, with w/h = 0.5, 1 and 2 on εr = 4.4, w/h = 4 on εr = 2.2 and w/h = 1 on εr = 9.8, on the default graded mesh.

**Reference.** Hammerstad and Jensen's closed forms for Z0 and εeff of a zero-thickness microstrip; the authors state Z0 in air within 0.01 % (w/h ≤ 1) and 0.03 % (w/h ≤ 1000) and εeff within 0.2 % (εr ≤ 128, 0.01 ≤ w/h ≤ 100) of exact quasi-static values. E. Hammerstad and Ø. Jensen, "Accurate models for microstrip computer-aided design," IEEE MTT-S International Microwave Symposium Digest, 1980, pp. 407–409.

**Compared.** Z0, εeff and delay per length of the line.

**Tolerance.** 0.5 % on Z0 and εeff; the delay follows from εeff.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `microstrip_matches_hammerstad_jensen`.

### Stripline and coupled stripline against Cohn's exact forms

**Problem.** Zero-thickness strips centred between planes 0.5 mm apart in εr = 3.0: single strips with w/b = 0.2, 0.5 and 1.0, and edge-coupled pairs with (w/b, s/b) = (0.5, 0.2), (0.5, 0.5) and (0.3, 1.0).

**Reference.** Cohn's conformal-mapping solutions, exact for zero-thickness strips between infinite planes: Z0 = (30π/√εr) K(k)/K(k′) with k = sech(πw/2b), and the even- and odd-mode impedances with k_e = tanh(πw/2b) tanh(π(w+s)/2b), k_o = tanh(πw/2b) coth(π(w+s)/2b). S. B. Cohn, "Characteristic impedance of the shielded-strip transmission line," IRE Trans. Microwave Theory Tech., vol. MTT-2, pp. 52–57, 1954; S. B. Cohn, "Shielded coupled-strip transmission line," IRE Trans. Microwave Theory Tech., vol. MTT-3, pp. 29–38, 1955.

**Compared.** Z0 of single strips; even- and odd-mode impedances of pairs, with Zdiff = 2 Zodd, Zcommon = Zeven / 2, εeff = εr and zero far-end coupling in the homogeneous dielectric.

**Tolerance.** 0.3 % on each impedance; the identities hold to 1e-9.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `stripline_matches_cohn_exact_forms`.

### Microstrip and stripline against IPC-2141A

**Problem.** 35 µm copper. Surface microstrips (w, h, εr) = (0.35 mm, 0.2 mm, 4.1), (0.2 mm, 0.15 mm, 4.3) and (0.15 mm, 0.25 mm, 3.5); centred striplines (w, b, εr) = (0.15 mm, 0.6 mm, 4.2), (0.1 mm, 0.5 mm, 3.6) and (0.2 mm, 0.8 mm, 4.4). Every case is inside the forms' stated ranges.

**Reference.** IPC-2141A surface microstrip Z0 = 87/√(εr + 1.41) · ln(5.98h/(0.8w + t)) for 0.1 < w/h < 2.0 and 1 < εr < 15, and symmetric stripline Z0 = 60/√εr · ln(1.9b/(0.8w + t)) for w/b < 0.35 and t/b < 0.25. IPC-2141A, Design Guide for High-Speed Controlled Impedance Circuit Boards, IPC, 2004.

**Compared.** Z0 of each line.

**Tolerance.** 5 %: the IPC-2141A forms are empirical fits, so this check bounds agreement with the board-industry forms; Hammerstad–Jensen and Cohn are the tight references.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `microstrip_and_stripline_agree_with_ipc_2141a_inside_their_ranges`.

### Mesh and enclosure convergence of Z0

**Problem.** A 0.35 mm, 35 µm-thick surface microstrip on 0.2 mm of εr = 4.1, meshed with edge sizes of 7, 3.5, 1.75 and 0.875 µm, then with the default mesh in an enclosure twice the default width and height.

**Reference.** Self-convergence: the change in Z0 between successive refinements, and the change from doubling the enclosure.

**Compared.** Z0.

**Tolerance.** Successive changes decrease and the last is below 0.1 %; the enclosure changes Z0 by under 0.05 %.

**Result.** Passed (1 test) in the evidence run of 2026-11-08.

Tests: `microstrip_impedance_converges_with_the_mesh_and_enclosure`.
