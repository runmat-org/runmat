---
description: Worked linear static analysis in RunMat. Check a steel cantilever's tip deflection against a span limit and its bending stress against an allowable, read reactions, and run the study from the CLI.
keywords: [linear static analysis, structural FEA, cantilever, tip deflection, von Mises stress, support reaction, Tet10, stress singularity, deflection limit, allowable stress]
doc_type: guide
family: [structural]
examples: [cantilever-static]
---
A linear static analysis finds how far a part deflects, how stressed it is and what its supports carry under a steady load. Use it to check a deflection against a stiffness limit, a stress against an allowable, or a reaction against a fastener or foundation rating. It applies while strains stay small, the material stays elastic and contact does not open or close; inertia, plasticity, large rotation and changing contact belong to the [transient, nonlinear and contact procedures](../language/reference/procedures.md).

## The question

A 1 m steel cantilever with a 100 mm × 100 mm square section carries a 1,000 N downward load at its free end.

- Does the tip deflect less than **L/2,000 = 0.5 mm**?
- Does the bending stress near the support stay below an allowable of **167 MPa** (250 MPa yield with a factor of safety of 1.5)?
- What force must the support carry?

<!-- fea:example:cantilever-static:setup -->

| Input | Value | Where |
| --- | --- | --- |
| Length × section | 1 m × 100 mm × 100 mm | `cantilever.step`, millimeters |
| Support | all displacement fixed | the x = 0 face, "Fixed end" |
| Load | 100 kPa traction in −z, a 1,000 N resultant | the x = 1 m face, "Loaded end" |
| Steel | E = 200 GPa, ν = 0.3, ρ = 7,850 kg/m³ | the body, "Beam" |
| Mesh | quadratic tetrahedra (Tet10), 150 mm target size, 1 mm surface deviation | |

## Inspect the model

<!-- fea:example:cantilever-static:geometry -->

The fixed end is the x = 0 face and the loaded end the x = 1 m face; both names come from the STEP file. The mesh is coarse along the beam because Tet10 elements represent the linear variation of bending stress through the depth exactly, so a few elements through the section capture bending well.

## Results

### Tip deflection

<!-- fea:example:cantilever-static:displacement -->

Color shows displacement magnitude in meters on the deformed shape; the shape is exaggerated, and the scale under the figure applies to the shape only. The free end moves down by **[0.2000 mm](receipt:cantilever-static#tip-deflection)**, the study's `tip_deflection` output: the largest displacement magnitude over the loaded face.

### Bending stress near the support

<!-- fea:example:cantilever-static:stress -->

Color shows the element von Mises stress in pascals: each element's own value, without smoothing between elements. The stress rises toward the support, where the bending moment is largest, and peaks on the top and bottom surfaces.

Read the stress one section depth from the support, at x = 100 mm on the top surface: the nodal von Mises stress there is **[5.43 MPa](receipt:cantilever-static#top-von-mises)**. Don't read it at the clamped edge itself. A fully fixed face meeting a free surface is a stress singularity: the theoretical stress there is unbounded, so the largest element value near that edge does not settle as the mesh is refined ([5.39 MPa](receipt:cantilever-static#peak-element-von-mises), [5.27 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-50) and [5.76 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-25) at 150, 50 and 25 mm), while the value at x = 100 mm settles ([5.43 MPa](receipt:cantilever-static#top-von-mises), [5.41 MPa](receipt:cantilever-static#top-von-mises-tet10-50), [5.41 MPa](receipt:cantilever-static#top-von-mises-tet10-25)). The [mesh refinement guide](./mesh-refinement.md) shows this sequence.

### Support reaction

The reaction at the fixed face sums to **[1,000 N](receipt:cantilever-static#support-reaction)** in +z, equal and opposite to the applied 1,000 N. A reaction that does not balance the applied load points to a load applied to the wrong region or a support that does not restrain what was intended.

## Check against the criteria

| Criterion | Limit | Result | Margin |
| --- | --- | --- | --- |
| Tip deflection | 0.5 mm (L/2,000) | [0.2000 mm](receipt:cantilever-static#tip-deflection) | 40 % of the limit |
| Von Mises stress, top surface at x = 100 mm | 167 MPa | [5.43 MPa](receipt:cantilever-static#top-von-mises) | 3.3 % of the allowable |
| Support reaction | balances 1,000 N | [1,000 N](receipt:cantilever-static#support-reaction) | balanced |

Both criteria pass: the beam is stiffness-governed long before it is strength-governed. As a check on the model, Euler–Bernoulli beam theory gives a tip deflection of $FL^3/3EI = 0.2000$ mm and a top-surface stress of $M c / I = 5.40$ MPa at x = 100 mm, where $M = F(L - x)$.

## Build and modify the study

The complete example opens in RunMat Desktop with its source, the STEP file and the saved results. The [`.fea` language overview](../language/index.md#a-complete-study) walks through this file line by line.

<!-- fea:example:cantilever-static:edit -->

To state the criteria in the study itself, add outputs and assessments to the `study` block. Each run then reports a pass or fail verdict for each, computed from the authored output values (`evidence = authored_samples`):

```fea fragment
output top_stress = interpolate(analysis.static.field.node.von_mises_stress, at = [100 mm, 50 mm, 100 mm])
output support_reaction = resultant(analysis.static.field.node.reaction_force, over = geometry.beam.region.fixed_end)

assessment deflection_limit {
    value = output.tip_deflection
    maximum = 0.5 mm
    evidence = authored_samples
}
assessment stress_allowable {
    value = output.top_stress
    maximum = 167 MPa
    evidence = authored_samples
}
```

## Try a change

- **Double the traction** to 200 kPa. The analysis is linear, so deflection, stress and reaction all double.
- **Use aluminum** (`youngs_modulus = 68.9 GPa`). Deflection grows by the stiffness ratio, 200 / 68.9 ≈ 2.9 times; stress is essentially unchanged, because the beam is statically determinate and the moment at each section is set by the load alone.
- **Refine the mesh** to `target_size = 50 mm` and compare the tip deflection and the stress at x = 100 mm. The deflection changes by +0.2 % and the stress by −0.4 %; the peak at the clamped edge does not settle.
- **Switch to linear tetrahedra** (`order = linear`, Tet4). Linear tetrahedra are too stiff in bending: at the same size the tip deflection falls well short of beam theory ([0.1625 mm](receipt:cantilever-static#tip-deflection-tet4-50) at 50 mm, 19 % short). Use Tet10 for bending-dominated parts.

## Run from the command line

From the downloaded example folder:

```sh
runmat check cantilever-static.fea
runmat plan cantilever-static.fea
runmat run cantilever-static.fea
```

`check` verifies units, references and the CAD region names in a few seconds. `plan` prints the geometry, mesh, analysis and output work. `run` imports the STEP file, meshes, solves, and publishes the fields, the outputs and the assessment verdict. Add `--json` for structured output. See [check, plan and run](../language/index.md#check-plan-and-run).

## How RunMat computes this

RunMat solves the small-strain elasticity equations in weak form: find the displacement $\mathbf u$ such that

$$
\int_\Omega \boldsymbol\varepsilon(\mathbf v) : \mathsf C : \boldsymbol\varepsilon(\mathbf u)\, d\Omega = \int_{\Gamma_N} \mathbf v \cdot \mathbf t\, d\Gamma \quad \text{for all admissible } \mathbf v,
$$

with $\boldsymbol\varepsilon = \tfrac12(\nabla\mathbf u + \nabla\mathbf u^{\mathsf T})$ and $\mathbf t$ the applied traction. Each Tet10 element contributes a stiffness matrix $\mathbf K^e = \int_{\Omega_e} \mathbf B^{\mathsf T}\mathbf D\,\mathbf B\, d\Omega$ evaluated by Gauss quadrature, and the element matrices are assembled into the global system $\mathbf K\mathbf u = \mathbf f$. The fixed face is applied by eliminating its degrees of freedom, and the symmetric positive-definite system is solved with a sparse Cholesky factorization. Stress is computed in each element from the displacement gradient; the nodal von Mises stress averages the stress tensors of the elements that share a node and takes von Mises from the averaged tensor. Reactions are the internal forces at the constrained degrees of freedom.

The full derivation, element formulation and solver options are in the [structural physics reference](../physics/structural.md).

## Validation

The structural formulation is checked against closed-form beam, plate and bar solutions with mesh sequences and stated tolerances; see [structural validation](../validation/structural.md).
