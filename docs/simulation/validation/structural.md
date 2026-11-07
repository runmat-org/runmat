---
description: Reference checks for RunMat structural mechanics analyses: each problem, its closed-form, manufactured or published reference, the compared quantity, the tolerance and the latest result.
keywords: [structural mechanics validation, verification, reference solution, mesh convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Structural mechanics validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each check compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. The tests listed under a check run on every change and in the evidence report; receipts record a public study run with its build, commit and mesh. The [validation methodology](./methodology.md) explains how checks are built and how tolerances are set, and [Status](./status.md) summarizes every family.

## Static

### Cantilever bending against beam theory

**Problem.** A 1 m steel cantilever (E = 200 GPa, ν = 0.3) with a 100 mm × 100 mm section is fully fixed at x = 0 and carries a 1,000 N downward traction resultant on the free end face. It is the cantilever of the [linear static](../guides/linear-static.md) and [mesh refinement](../guides/mesh-refinement.md) guides, run as a public study on Tet10 meshes at 150, 75, 50 and 25 mm and Tet4 meshes at 150 and 50 mm.

**Reference.** Euler–Bernoulli beam theory: tip deflection $\delta = FL^3/3EI$ = 0.2000 mm and top-surface bending stress $\sigma = Mc/I$ with $M = F(L - x)$, 5.40 MPa at x = 100 mm. The support reaction equals the applied 1,000 N. Timoshenko theory adds the shear deflection $FL/\kappa GA \approx$ 0.0016 mm (0.8 %), so a converged solid model lies between the two theories.

**Compared.** Largest displacement on the loaded end face; nodal von Mises stress on the top surface at x = 100 mm, one section depth from the clamp; support reaction. The largest element stress at the clamped edge is recorded to show the singularity of the fully fixed face.

**Tolerance.** None set; the receipts record each value. Tet10 tip deflection is expected between the Euler–Bernoulli and Timoshenko values.

**Result.** Run receipts dated 2026-10-29. Tet10 tip deflection converges to 0.2007 mm, 0.34 % above Euler–Bernoulli theory and below the Timoshenko value. The stress at x = 100 mm settles at 5.41 MPa, 0.2 % above beam theory. The largest element stress at the clamped edge does not settle (5.39, 5.45, 5.27, 5.76 MPa), as expected at a singularity. The reaction balances 1,000 N on every mesh. Tet4 locks in bending: −50 % at 150 mm and −19 % at 50 mm.

| Quantity | Mesh | RunMat | Reference | Difference |
| --- | --- | ---: | ---: | ---: |
| Tip deflection: largest displacement magnitude over the loaded end face | Tet10, 150 mm, 2,727 nodes | [0.200034 mm](receipt:cantilever-static#tip-deflection) | 0.2 mm | +0.02 % |
|  | Tet10, 75 mm, 2,758 nodes | [0.200295 mm](receipt:cantilever-static#tip-deflection-tet10-75) | 0.2 mm | +0.15 % |
|  | Tet10, 50 mm, 4,679 nodes | [0.200474 mm](receipt:cantilever-static#tip-deflection-tet10-50) | 0.2 mm | +0.24 % |
|  | Tet10, 25 mm, 32,075 nodes | [0.200678 mm](receipt:cantilever-static#tip-deflection-tet10-25) | 0.2 mm | +0.34 % |
|  | Tet4, 150 mm, 431 nodes | [0.10069 mm](receipt:cantilever-static#tip-deflection-tet4-150) | 0.2 mm | −49.66 % |
|  | Tet4, 50 mm, 729 nodes | [0.162496 mm](receipt:cantilever-static#tip-deflection-tet4-50) | 0.2 mm | −18.75 % |
| Nodal von Mises stress on the top surface at x = 100 mm, y = 50 mm | Tet10, 150 mm, 2,727 nodes | [5.43303 MPa](receipt:cantilever-static#top-von-mises) | 5.4 MPa | +0.61 % |
|  | Tet10, 75 mm, 2,758 nodes | [5.42193 MPa](receipt:cantilever-static#top-von-mises-tet10-75) | 5.4 MPa | +0.41 % |
|  | Tet10, 50 mm, 4,679 nodes | [5.41287 MPa](receipt:cantilever-static#top-von-mises-tet10-50) | 5.4 MPa | +0.24 % |
|  | Tet10, 25 mm, 32,075 nodes | [5.41221 MPa](receipt:cantilever-static#top-von-mises-tet10-25) | 5.4 MPa | +0.23 % |
|  | Tet4, 150 mm, 431 nodes | [1.92644 MPa](receipt:cantilever-static#top-von-mises-tet4-150) | 5.4 MPa | −64.33 % |
|  | Tet4, 50 mm, 729 nodes | [3.51563 MPa](receipt:cantilever-static#top-von-mises-tet4-50) | 5.4 MPa | −34.90 % |
| Largest element von Mises stress in the beam (at the clamped edge; does not converge) | Tet10, 150 mm, 2,727 nodes | [5.39247 MPa](receipt:cantilever-static#peak-element-von-mises) | none |  |
|  | Tet10, 75 mm, 2,758 nodes | [5.45246 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-75) | none |  |
|  | Tet10, 50 mm, 4,679 nodes | [5.26615 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-50) | none |  |
|  | Tet10, 25 mm, 32,075 nodes | [5.76473 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-25) | none |  |
|  | Tet4, 150 mm, 431 nodes | [24.80245 MPa](receipt:cantilever-static#peak-element-von-mises-tet4-150) | none |  |
|  | Tet4, 50 mm, 729 nodes | [9.19027 MPa](receipt:cantilever-static#peak-element-von-mises-tet4-50) | none |  |
| Resultant reaction force at the fixed end, z component | Tet10, 150 mm, 2,727 nodes | [1000 N](receipt:cantilever-static#support-reaction) | 1000 N | +0.00 % |
|  | Tet10, 75 mm, 2,758 nodes | [1000 N](receipt:cantilever-static#support-reaction-tet10-75) | 1000 N | +0.00 % |
|  | Tet10, 50 mm, 4,679 nodes | [1000 N](receipt:cantilever-static#support-reaction-tet10-50) | 1000 N | +0.00 % |
|  | Tet10, 25 mm, 32,075 nodes | [1000 N](receipt:cantilever-static#support-reaction-tet10-25) | 1000 N | +0.00 % |
|  | Tet4, 150 mm, 431 nodes | [1000 N](receipt:cantilever-static#support-reaction-tet4-150) | 1000 N | +0.00 % |
|  | Tet4, 50 mm, 729 nodes | [1000 N](receipt:cantilever-static#support-reaction-tet4-50) | 1000 N | +0.00 % |

### Tet4 patch test under an affine displacement field

**Problem.** Connected patches of linear tetrahedra (Tet4) under an affine displacement field, the standard patch test for solid elements.

**Reference.** The exact uniform strain and stress of the affine field (strain 0.1, 0.2 and 0.3 in x, y and z; stress 32, 40 and 48 in the fixture's units), its strain energy, and nodal forces equal to the independent boundary tractions.

**Compared.** Recovered strain and stress at elements and integration points, strain energy, internal nodal forces.

**Tolerance.** Absolute 1e-12 on strain and 1e-11 on stress and energy; relative 1e-12 on nodal forces and energy.

**Result.** 

### Tet10 nodal stress recovery under a linear stress field

**Problem.** Two quadratic tetrahedra (Tet10) sharing a face carry a linear stress field; the two elements belong to different regions.

**Reference.** The linear stress field evaluated exactly at every corner and midside node, with no averaging across the region interface node.

**Compared.** Recovered nodal stress components.

**Tolerance.** Absolute error at most 1e-12 × (1 + |σ|) per component.

**Result.** 

### Shell membrane patch under biaxial plane stress

**Problem.** A one-layer shell patch in uniform biaxial membrane stress.

**Reference.** Plane-stress membrane resultants $N_x = \sigma_x t$, $N_y = \sigma_y t$, $N_{xy} = 0$, no bending or shear resultants, and von Mises stress $\sqrt{\sigma_x^2 - \sigma_x\sigma_y + \sigma_y^2}$.

**Compared.** Membrane force resultants, bending and shear resultants, von Mises stress.

**Tolerance.** Relative 1e-11 on resultants; bending and shear resultants below 1e-12.

**Result.** 

### Simply supported plate under sinusoidal load

**Problem.** A simply supported plate meshed with triangular shell elements carries a sinusoidal transverse pressure; the mesh is refined.

**Reference.** Navier's series solution for the center deflection of a simply supported plate.

**Compared.** Center transverse displacement.

**Tolerance.** The center displacement approaches the Navier value with refinement; the finest-mesh error is below 8 %.

**Result.** 

## Modal

### Fundamental frequency of an axial rod

**Problem.** A solid slab rod, fixed at one end and free at the other, in axial vibration on Tet4 meshes (2, 4 and 8 cells) and Tet10 meshes (1, 2 and 4 cells). The retained one-mode response is also compared with a direct harmonic solve.

**Reference.** Fixed–free rod fundamental frequency $f_1 = c/4L$ with $c = \sqrt{E/\rho}$.

**Compared.** First natural frequency; modal-truncation response against the direct harmonic response.

**Tolerance.** Frequency error falls by more than 3× (Tet4) or 6× (Tet10) per refinement; finest-mesh error below 2e-3 (Tet4) and 2e-4 (Tet10).

**Result.** 

## Harmonic

### Axial rod under harmonic end traction

**Problem.** A Tet4 slab rod, fixed at one end, carries a harmonic axial traction on the free end at three frequencies on both sides of the first resonance; 2, 4 and 8 cells.

**Reference.** The continuum complex displacement of the driven rod and the support reaction $-F/\cos kL$.

**Compared.** Complex displacement field and support reaction; mean input power of the undamped response.

**Tolerance.** Displacement and reaction errors fall by more than 2.5× per refinement and are below 2e-2 on the finest mesh; mean input power below 1e-9.

**Result.** 

## Transient

### Undamped free vibration under average-acceleration integration

**Problem.** An undamped oscillator in free vibration, integrated with the Newmark average-acceleration rule, both directly and in modal coordinates, with the time step halved twice.

**Reference.** The exact harmonic free response.

**Compared.** Displacement and velocity at the final time.

**Tolerance.** Error ratio above 3.8 per halving of the time step (second order) on both paths; finest direct Newmark error below 2e-3; modal relative residuals at most 2e-15.

**Result.** 

### Rod under affine forcing in time

**Problem.** A Tet4 and a Tet10 rod under a physical forcing that is affine in space, integrated with 20, 40 and 80 time intervals, with and without damping.

**Reference.** The independent continuum response of the rod and its work–energy balance.

**Compared.** Displacement, velocity and acceleration fields; work–energy balance.

**Tolerance.** Spatially exact quantities below 1e-12; time-discretization errors fall by more than 3× per halving and are below 2e-4 at 80 intervals.

**Result.** 

## Buckling

### Fixed–free column against Euler's buckling load

**Problem.** A solid column, 1 m long with a 100 × 50 mm section, fixed at the base and free at the top, carries a 250 kN axial preload; Tet10 mesh.

**Reference.** Euler's critical load for a fixed–free column, $P_{cr} = \pi^2EI/4L^2$.

**Compared.** First buckling load factor; eigenpair residuals.

**Tolerance.** First load factor within 1 % of $P_{cr}$/250 kN; eigenpairs converged to a relative tolerance of 1e-8.

**Result.** 

## Response spectrum

### Modal combination rules

**Problem.** Two signed modal responses with set frequencies and damping are combined by SRSS, absolute sum and CQC.

**Reference.** The SRSS, absolute-sum and CQC formulas evaluated independently, including an independently computed CQC correlation coefficient.

**Compared.** Combined peak response.

**Tolerance.** Relative 2e-13.

**Result.** 

## Nonlinear (plasticity)

### Uniaxial bar beyond yield

**Problem.** A bar in uniaxial tension loaded past yield, with J2 plasticity and linear hardening, solved by Newton iteration on Tet4 and Tet10 meshes and restarted cold from its persisted state.

**Reference.** The closed-form uniaxial elastoplastic stress at the final load (5.6 in the fixture's units).

**Compared.** Axial stress at every integration point; restart state.

**Tolerance.** Axial stress within 2e-10 of the closed form.

**Result.** 

### Manufactured fully yielded cube

**Problem.** A cube under body force and surface traction yields completely; J2 plasticity, Tet4 and Tet10 meshes, applied in one increment and in eight.

**Reference.** A manufactured polynomial J2 solution: quadratic terminal strain with an exact radial return on its proportional path.

**Compared.** Displacement, stress, hardening variable, plastic history, energies and reaction.

**Tolerance.** Tet10 in one increment: displacement, stress, hardening variable and energy errors below 1e-7 under the Keast15, Duffy5 and Duffy8 rules. In eight increments the yield front crosses elements mid-step; that path error falls at order 1.5 or better with refinement. Tet4: every quantity refines.

**Result.** 
