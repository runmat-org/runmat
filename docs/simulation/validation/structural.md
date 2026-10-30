## Structural mechanics coverage

Structural validation checks the connection between material response, element formulation, physical predictions, and the complete saved-result workflow. The criteria below state the required scope for each check.

## Cantilever bending against beam theory

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

## Element formulation and integration

Checks cover strain and stress recovery, stiffness, and load integration for supported solid element orders. Read the individual criteria for material laws, topology, and implementation boundaries.

## Physical references and mesh convergence

Reference problems and refinement examine displacement, stress, support reactions, and balance. Accuracy is assessed for the tested model and comparison quantities, including locations away from idealized support singularities.

## Execution and durable results

Workflow checks cover the authored study, admission, execution, original fields, and preservation after reopening. Their evidence is separate from a physical convergence assessment.

## Criteria and assessment records

<!-- fea:verification -->
