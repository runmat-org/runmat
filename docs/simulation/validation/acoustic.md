---
description: Reference checks for RunMat acoustics analyses: each problem, its closed-form, manufactured or published reference, the compared quantity, the tolerance and the latest result.
keywords: [acoustics validation, verification, reference solution, mesh convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Acoustics validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each check compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. The tests listed under a check run on every change and in the evidence report; receipts record a public study run with its build, commit and mesh. The [validation methodology](./methodology.md) explains how checks are built and how tolerances are set, and [Status](./status.md) summarizes every family.

## Modal

### Rigid rectangular cavity modes

**Problem.** A rigid-walled cube cavity. The kernel test uses quadratic meshes with 2, 3 and 4 cells per side and a linear mesh with 2 and 4; the public-study test runs a 1 m cube through the command line.

**Reference.** Rectangular-cavity frequencies $f_{lmn} = \frac{c}{2}\sqrt{(l/L_x)^2 + (m/L_y)^2 + (n/L_z)^2}$, including the zero-frequency uniform mode and degenerate triplets; for the 1 m cube the first axial mode is $c/2L$ = 171.5 Hz.

**Compared.** First eight natural frequencies; residuals and mass orthogonality.

**Tolerance.** Quadratic: error ratio above 2.5 per refinement and below 4e-3 at 4 cells, residuals at most 1e-8, mass orthogonality below 1e-8. Linear: error ratio above 3 from 2 to 4 cells. Command line: uniform mode below 1e-3 Hz and the three axial modes within 2 % of 171.5 Hz.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Open–open duct modes

**Problem.** A duct with pressure-release ends; Tet10.

**Reference.** Open–open duct frequencies $f_n = nc/2L$, with no rigid mode and zero pressure on the released ends.

**Compared.** First five natural frequencies; modal pressure on the released nodes.

**Tolerance.** Relative 4e-3; released-node pressure exactly zero.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

## Harmonic

### Sound power through a matched duct end

**Problem.** A plane wave travels down a duct to an anechoic (matched) end.

**Reference.** Plane-wave power $|p|^2A/2\rho c$.

**Compared.** Power through the matched end; accounted input power.

**Tolerance.** Relative 1e-3 (0.1 %).

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Transmission loss of a plane layer

**Problem.** A plane layer between an incident port and an anechoic port, at several frequencies, on two meshes.

**Reference.** Closed-form plane-wave transmission loss of the layer at each frequency.

**Compared.** Transmission loss.

**Tolerance.** Below 0.05 dB on the finer mesh at every frequency, and no larger than on the coarse mesh.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Plane wave across a two-fluid interface

**Problem.** A plane wave crosses the interface between two fluids in an aligned slab. The public example runs the same interface with impedance ratio m = 4 at 500, 1000 and 2000 Hz.

**Reference.** Plane-wave reflection and transmission at the interface; for m = 4 the transmission loss is 1.938 dB.

**Compared.** Pressure, mean velocity and transmitted power; transmission loss of the public example.

**Tolerance.** Kernel: power residual below 1e-8; finest-mesh pressure error below 0.004 of the incident amplitude and velocity error below 0.01. Public example: within 0.1 dB of 1.938 dB.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Lossy duct standing wave

**Problem.** A duct with loss factor 2ζ driven at one end.

**Reference.** The complex-wavenumber standing wave and the power it dissipates.

**Compared.** Nodal complex pressure; dissipated against input power.

**Tolerance.** Pressure error ratio above 3 per refinement and below 0.004 on the finest mesh; power error below 1e-9.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Duct with a complex impedance termination

**Problem.** A duct terminated by a tabulated complex impedance; Tet4.

**Reference.** The two-wave duct solution for the termination's reflection coefficient.

**Compared.** Nodal complex pressure; dissipated against input power.

**Tolerance.** Pressure error ratio above 3 per refinement and below 0.004 on the finest mesh; power error below 1e-9.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Plane traveling wave

**Problem.** A Tet10 slab carrying a plane traveling wave at two frequencies, three refinements.

**Reference.** The exact traveling wave: pressure phase, particle velocity, intensity and energy.

**Compared.** Pressure, velocity, intensity and energy; power balance.

**Tolerance.** Pressure error below 0.004; velocity RMS error below 0.01; intensity and energy RMS errors below 0.005; residual below 1e-10 and power error below 1e-8.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Forced axial cavity near resonance

**Problem.** A rigid cavity forced axially at 0.9 and 1.1 times its first resonance, on Tet4 and Tet10 meshes.

**Reference.** The closed-form forced standing wave and its acoustic energy.

**Compared.** Pressure phase and amplitude; acoustic energy.

**Tolerance.** Finest-mesh pressure error below 0.10; energy RMS error below 0.15; power error below 1e-8.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Incident wave on an impedance boundary

**Problem.** A planar incident wave meets an impedance boundary with reflection coefficient 7/13.

**Reference.** Plane-wave reflection: net power $\frac{|p|^2}{2\rho c}(1 - R^2)$, with the reflected pressure and velocity.

**Compared.** Pressure, velocity and net power.

**Tolerance.** Pressure error below 0.004 of the incident amplitude; velocity below 0.01; intensity terms below 0.005; power error below 1e-8.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Plane wave across periodic boundaries

**Problem.** An anchored slab carries a plane wave across declared pairs of periodic planes.

**Reference.** The exact plane wave with Bloch phase across each periodic pair.

**Compared.** Pressure, mean velocity and signed intensity.

**Tolerance.** Pressure error below 0.004; velocity below 0.01; intensity below 0.005.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Plane-wave radiation boundary

**Problem.** A slab closed by a local plane-wave radiation boundary.

**Reference.** Plane-wave impedance $\rho c$: inlet and outlet power $|p|^2/2\rho c$ with the exact phase.

**Compared.** Pressure phase, mean velocity and signed power at inlet and outlet.

**Tolerance.** Power within 0.005; velocity error below 0.01; residual below 1e-10 and power error below 1e-8.

**Result.** Not yet in the evidence catalog; run `evidence_report`.

### Volume-velocity source

**Problem.** Slabs of several cross-sectional areas driven by a total volume flow.

**Reference.** The plane wave set by the volume velocity, with pressure scaling as 1/A.

**Compared.** Pressure, velocity and power against area and phase.

**Tolerance.** Pressure error below 0.004·√1.16/A; velocity below 0.01/A; power error below 1e-8.

**Result.** Not yet in the evidence catalog; run `evidence_report`.
