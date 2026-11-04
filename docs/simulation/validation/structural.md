---
description: How RunMat checks its structural mechanics results: reference cases with their errors and receipts, and the tests behind each requirement, generated from the capability ledger.
keywords: [structural mechanics validation, verification, reference cases, convergence, FEA accuracy]
doc_type: validation
generated: node scripts/docs/generate-pages.mjs
---
# Structural mechanics validation

<!-- Generated from docs/simulation/ledger.json by scripts/docs/generate-pages.mjs; do not edit by hand. -->

Each case below compares a RunMat result with an independent reference: a closed-form solution, a manufactured solution or a published benchmark. Values link to their run receipts. The [validation methodology](./methodology.md) explains how to read them, and [Status](./status.md) lists what the release runs.

## Capabilities

| Capability | In this release | Cases |
| --- | --- | --- |
| Structural static | yes | [Cantilever bending against beam theory](#cantilever-bending-against-beam-theory) |
| Structural modal | yes | — |
| Structural prestressed modal | yes | — |
| Structural buckling | yes | — |
| Structural transient | yes | — |
| Structural harmonic | yes | — |
| Structural prestressed harmonic | yes | — |
| Structural nonlinear | yes | — |
| Structural transient (modal superposition) | yes | — |
| Structural harmonic (modal superposition) | yes | — |
| Structural prestressed harmonic (modal superposition) | yes | — |
| Structural response spectrum (modal superposition) | yes | — |
| Structural random response (modal superposition) | yes | — |
| Nonlinear structural steps inside procedure graphs (preload → nonlinear → modal; thermal prefix → nonlinear) | [not yet in the current release](pending:structural.nonlinear.sequence) | — |

## Reference cases

### Cantilever bending against beam theory

The cantilever in the [linear static](../guides/linear-static.md) and [mesh refinement](../guides/mesh-refinement.md) guides is a closed-form check of the solid elements in bending. A 1 m steel beam (E = 200 GPa, ν = 0.3) with a 100 mm × 100 mm section is fully fixed at x = 0 and carries a 1,000 N downward traction resultant on the x = 1 m face.

**References.** Euler–Bernoulli beam theory gives the tip deflection $\delta = FL^3/3EI = 0.2000$ mm and the top-surface bending stress $\sigma = Mc/I$ with $M = F(L - x)$: 5.40 MPa at x = 100 mm.

**Where stress is compared.** At x = 100 mm on the top surface, one section depth from the support. The clamped edge itself is a stress singularity of the idealized fully fixed face: its largest element value does not converge, so it is not a comparison point. One section depth away, the Saint-Venant end effect of the clamp has decayed and beam theory applies.

| Tet10 target size | Nodes | Tip deflection | vs 0.2000 mm | Stress at x = 100 mm | vs 5.40 MPa | Largest element stress (clamped edge) | Reaction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150 mm | 2,727 | [0.20003 mm](receipt:cantilever-static#tip-deflection) | +0.02 % | [5.433 MPa](receipt:cantilever-static#top-von-mises) | +0.6 % | [5.39 MPa](receipt:cantilever-static#peak-element-von-mises) | [1,000 N](receipt:cantilever-static#support-reaction) |
| 75 mm | 2,758 | [0.20030 mm](receipt:cantilever-static#tip-deflection-tet10-75) | +0.15 % | [5.422 MPa](receipt:cantilever-static#top-von-mises-tet10-75) | +0.4 % | [5.45 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-75) | [1,000 N](receipt:cantilever-static#support-reaction-tet10-75) |
| 50 mm | 4,679 | [0.20047 mm](receipt:cantilever-static#tip-deflection-tet10-50) | +0.24 % | [5.413 MPa](receipt:cantilever-static#top-von-mises-tet10-50) | +0.2 % | [5.27 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-50) | [1,000 N](receipt:cantilever-static#support-reaction-tet10-50) |
| 25 mm | 32,075 | [0.20068 mm](receipt:cantilever-static#tip-deflection-tet10-25) | +0.34 % | [5.412 MPa](receipt:cantilever-static#top-von-mises-tet10-25) | +0.2 % | [5.76 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-25) | [1,000 N](receipt:cantilever-static#support-reaction-tet10-25) |

| Tet4 target size | Nodes | Tip deflection | vs 0.2000 mm | Stress at x = 100 mm | Largest element stress | Reaction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 150 mm | 431 | [0.1007 mm](receipt:cantilever-static#tip-deflection-tet4-150) | −50 % | [1.93 MPa](receipt:cantilever-static#top-von-mises-tet4-150) | [24.8 MPa](receipt:cantilever-static#peak-element-von-mises-tet4-150) | [1,000 N](receipt:cantilever-static#support-reaction-tet4-150) |
| 50 mm | 729 | [0.1625 mm](receipt:cantilever-static#tip-deflection-tet4-50) | −19 % | [3.52 MPa](receipt:cantilever-static#top-von-mises-tet4-50) | [9.19 MPa](receipt:cantilever-static#peak-element-von-mises-tet4-50) | [1,000 N](receipt:cantilever-static#support-reaction-tet4-50) |

**Reading.**

- The Tet10 tip deflection converges to 0.2007 mm, 0.34 % above Euler–Bernoulli theory. Euler–Bernoulli theory neglects shear deformation; Timoshenko theory adds $FL/\kappa GA \approx 0.0016$ mm (0.8 %) for this 10:1 beam, while the fully fixed face, which also stops the section's Poisson contraction at the root, stiffens it slightly. The finite-element value is expected between the two theories, and it is.
- The stress at x = 100 mm settles at 5.41 MPa, 0.2 % above beam theory, with less than 0.4 % change across the sequence.
- The largest element stress at the clamped edge does not settle (5.39, 5.45, 5.27, 5.76 MPa), as expected at a singularity.
- The reaction balances the applied 1,000 N on every mesh.
- Tet4 locks in bending: 50 % short at 150 mm and 19 % short at 50 mm. Bending-dominated parts use Tet10.

Each value links to its run receipt: CLI build, commit, mesh, date and result id.

- References: Euler–Bernoulli beam theory: tip deflection FL³/3EI, bending stress Mc/I; Timoshenko beam theory: shear deflection FL/κGA.

## Test evidence by requirement

Each requirement lists acceptance obligations, and each obligation must hold in stated envelopes: formulation, element order and execution provider. Tests declare the obligation and envelope they support at the test; the counts are envelope cells from the latest evidence run. PASS covers the obligation fully, PARTIAL passes with a narrower boundary than the criterion, and OPEN has no supporting test yet.

| Requirement | Title | Obligations | PASS | PARTIAL | FAIL | OPEN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FEA-STRUCTURAL-SOLID-001 | Physical linear solid operators, loads and assigned-material recovery | 4 | 0 | 7 | 0 | 49 |
| FEA-STRUCTURAL-BEAM-001 | Physical three-dimensional beams and explicit mixed-model coupling | 2 | 0 | 2 | 0 | 26 |
| FEA-STRUCTURAL-SHELL-001 | Physical membrane and bending shells | 2 | 0 | 1 | 0 | 27 |
| FEA-STRUCTURAL-REFERENCE-001 | Linear structural independent references and physical balances | 2 | 0 | 3 | 0 | 25 |
| FEA-STRUCTURAL-CONSTRAINT-001 | Complete typed structural constraints and constitutive connectors | 6 | 0 | 8 | 0 | 76 |
| FEA-STRUCTURAL-EXECUTION-001 | Governed linear structures and execution equivalence | 7 | 0 | 2 | 0 | 96 |
| FEA-PUBLIC-STUDIES-001 | Published studies run from the public command | 9 | 0 | 0 | 1 | 68 |
| FEA-SCALE-PARALLEL-001 | Every family saturates the machine through one thread plan | 6 | 0 | 2 | 0 | 114 |
| FEA-SCALE-DISTRIBUTED-001 | Remote and cooperative execution for every family | 5 | 0 | 0 | 0 | 118 |
| FEA-SCALE-SIZE-001 | Millions of degrees of freedom at stated time and memory | 5 | 0 | 0 | 0 | 94 |
| FEA-STRUCTURAL-PUBLIC-001 | Structural capabilities the page claims | 10 | 0 | 0 | 0 | 100 |
| FEA-DYNAMICS-MODAL-001 | Physical modal extraction, eigenspaces and reusable participation state | 4 | 0 | 3 | 0 | 53 |
| FEA-COUPLING-CHAIN-001 | Chained thermal-structural studies | 2 | 0 | 0 | 0 | 24 |
| FEA-DYNAMICS-PRESTRESS-001 | Validated prestressed base states and stability perturbation | 4 | 0 | 3 | 0 | 109 |
| FEA-DYNAMICS-RESPONSE-001 | Distinct second-order direct and modal dynamic procedures | 5 | 0 | 8 | 0 | 67 |
| FEA-DYNAMICS-GOVERNANCE-001 | Governed dynamic procedures, checkpoints and distributed evidence | 9 | 0 | 2 | 0 | 133 |
| FEA-NONLINEAR-STATE-001 | Complete original accepted nonlinear history through cold continuation | 6 | 0 | 22 | 0 | 2 |
| FEA-NONLINEAR-OPERATOR-001 | State-dependent nonlinear mechanics and supported plastic laws | 4 | 0 | 5 | 1 | 34 |
| FEA-CONTACT-OPERATOR-001 | Authored surface contact, search, friction and coupled inequality equilibrium | 8 | 0 | 11 | 0 | 69 |
| FEA-NONLINEAR-GOVERNANCE-001 | Converged nonlinear states, supported laws and independent qualification | 5 | 0 | 2 | 0 | 48 |
