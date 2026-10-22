# FEA-FLUID-PHYSICS-001: Free surface, rotating frames, lubrication, electrokinetics, particles, compressibility and reduced dimensions

0 PASS, 0 PARTIAL, 0 FAIL, 26 OPEN, 0 TAGGED. Generated; do not edit.

## free_surface_two_phase

Evidence kind: IndependentReference. Criterion: Volume-of-fluid or level-set free surface with surface tension, contact angle and Marangoni stress reproduces the static drop Laplace pressure 2 sigma / R with bounded spurious currents and Jurin capillary rise.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| free_surface_flow | tet4 | numerical_kernel | OPEN |  |
| free_surface_flow | tet4 | public_study | OPEN |  |
| free_surface_flow | tet10 | numerical_kernel | OPEN |  |
| free_surface_flow | tet10 | public_study | OPEN |  |

## rotating_frames_and_sliding_mesh

Evidence kind: IndependentReference. Criterion: Rotating reference frames (MRF and SRF) and sliding mesh reproduce solid-body rotation in a rotating annulus and Taylor vortex onset at Re_c about 68.2 for radius ratio 0.5.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| rotating_frame_flow | tet4 | numerical_kernel | OPEN |  |
| rotating_frame_flow | tet4 | public_study | OPEN |  |
| rotating_frame_flow | tet10 | numerical_kernel | OPEN |  |
| rotating_frame_flow | tet10 | public_study | OPEN |  |

## thin_film_lubrication

Evidence kind: IndependentReference. Criterion: The Reynolds equation with Elrod-JFO cavitation, journal and thrust bearings, squeeze film and coupling to structural deformation reproduces long and short journal bearing load and attitude angle (Sommerfeld); the journal bearing study reports load capacity at 3000 rpm and eccentricity ratio 0.6.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thin_film_lubrication | canonical_mixed_topology | numerical_kernel | OPEN |  |
| thin_film_lubrication | canonical_mixed_topology | public_study | OPEN |  |

## electrokinetics

Evidence kind: IndependentReference. Criterion: Electroosmotic slip (Helmholtz-Smoluchowski), Poisson-Nernst-Planck and dielectrophoresis run; electroosmotic plug flow equals -epsilon zeta E / mu.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electrokinetics | tet4 | numerical_kernel | OPEN |  |
| electrokinetics | tet4 | public_study | OPEN |  |
| electrokinetics | tet10 | numerical_kernel | OPEN |  |
| electrokinetics | tet10 | public_study | OPEN |  |

## particle_tracing

Evidence kind: IndependentReference. Criterion: Massless particles (residence time, mixing) and inertial particles with drag are traced one-way through a solved flow, and Stokes settling velocity is reproduced.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| particle_tracing | tet4 | numerical_kernel | OPEN |  |
| particle_tracing | tet4 | public_study | OPEN |  |
| particle_tracing | tet10 | numerical_kernel | OPEN |  |
| particle_tracing | tet10 | public_study | OPEN |  |

## compressible_and_networks

Evidence kind: IndependentReference. Criterion: Compressible laminar flow and rarefied slip reproduce microchannel gas flow with slip against the closed form, and one-dimensional pipe networks coupled to three-dimensional regions reproduce Darcy-Weisbach network pressure drop.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| compressible_laminar_flow | canonical_mixed_topology | numerical_kernel | OPEN |  |
| compressible_laminar_flow | canonical_mixed_topology | public_study | OPEN |  |
| pipe_network | canonical_mixed_topology | numerical_kernel | OPEN |  |
| pipe_network | canonical_mixed_topology | public_study | OPEN |  |

## two_dimensional_and_axisymmetric

Evidence kind: IndependentReference. Criterion: Two-dimensional and axisymmetric studies run through the same study language and reproduce the three-dimensional pipe and sphere results.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_mixed_topology | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_mixed_topology | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_mixed_topology | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | canonical_mixed_topology | public_study | OPEN |  |

