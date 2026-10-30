---
description: A worked example of linear static analysis. A steel cantilever under an end load is solved by beam theory and by the finite-element method in RunMat, from the governing equations and Tet10 discretization to deflection, stress, reactions and a design check.
keywords: [linear static analysis, structural FEA, cantilever, beam bending, Euler-Bernoulli, weak form, Tet10, shape functions, stiffness matrix, Ku = f, tip deflection, von Mises stress, support reaction, stress singularity, deflection limit]
doc_type: guide
family: [structural]
examples: [cantilever-static]
---
A linear static analysis finds the displacement, strain and stress in a supported part under loads that do not change with time. It applies while strains stay small, the material stays elastic and contacts neither open nor close. This page works one example in full: the theory a mechanics-of-materials course teaches, the finite-element method that generalizes it, and the solution in RunMat.

## Example 1: A steel cantilever under an end load

A steel cantilever is built into a rigid wall at one end and loaded downward at the other. The design must keep the tip deflection within L/2,000 and the bending stress below an allowable of 167 MPa (yield strength 250 MPa with a factor of safety of 1.5).

**Given**

<!-- fea:example:cantilever-static:setup -->

| Quantity | Value |
| --- | --- |
| Length | L = 1 m, along x from the wall |
| Cross-section | square, b = h = 100 mm |
| Material | steel, E = 200 GPa, ν = 0.3 |
| Support | the face at x = 0 is fixed in all directions |
| Load | P = 1,000 N downward (−z), spread uniformly over the end face at x = L as a traction of 100 kPa |
| Deflection limit | L/2,000 = 0.5 mm |
| Allowable stress | 167 MPa |

**Find**

1. The tip deflection.
2. The bending stress near the support.
3. The force and moment the wall carries.
4. Whether the beam meets the deflection limit and the allowable stress.

## Bending of a cantilever beam

### Assumptions

Euler–Bernoulli beam theory assumes that plane cross-sections stay plane and normal to the deflected axis, that deflections and rotations are small, and that the material is linear elastic. The first assumption neglects shear deformation, which is reasonable for a slender beam: this one has L/h = 10, and shear adds under 1 % to its deflection.

### Governing equations

With $v(x)$ the downward deflection, the bending moment and the curvature are related by $M = EI\,v''$, and equilibrium of a beam element with no distributed load gives

$$
EI\,\frac{d^4 v}{dx^4} = 0, \qquad 0 < x < L.
$$

The wall fixes the deflection and the slope, and the free end carries the shear force P and no moment:

$$
v(0) = 0, \quad v'(0) = 0, \quad EI\,v''(L) = 0, \quad EI\,v'''(L) = -P.
$$

Integrating four times and applying these conditions gives the deflected shape, the tip deflection and the moment:

$$
v(x) = \frac{P x^2 (3L - x)}{6EI}, \qquad \delta = v(L) = \frac{PL^3}{3EI}, \qquad M(x) = P\,(L - x).
$$

The bending stress varies linearly through the depth and is largest on the top and bottom surfaces, at a distance $c = h/2$ from the neutral axis:

$$
\sigma_x = \frac{M\,c}{I}, \qquad I = \frac{b h^3}{12}.
$$

For this beam, $I = 8.33 \times 10^{-6}\ \text{m}^4$, so $\delta = 0.200$ mm, the stress at the wall is 6.0 MPa, and the wall carries a force of 1,000 N and a moment of 1,000 N·m. These are the sizes of answer to expect.

### Finite-element formulation

Beam theory needs a slender, prismatic member. The finite-element method solves the equilibrium of the full three-dimensional solid, so the same steps apply to a bracket or a casting where no formula exists.

**Field equations.** Linear elasticity relates the displacement $\mathbf u$, the strain and the stress:

$$
\boldsymbol\varepsilon = \tfrac12\left(\nabla\mathbf u + \nabla\mathbf u^{\mathsf T}\right), \qquad
\boldsymbol\sigma = \mathsf C : \boldsymbol\varepsilon, \qquad
\nabla \cdot \boldsymbol\sigma = \mathbf 0 \ \text{in } \Omega,
$$

with $\mathbf u = \mathbf 0$ on the fixed face $\Gamma_u$ and $\boldsymbol\sigma\mathbf n = \bar{\mathbf t}$ on the loaded face $\Gamma_t$. For an isotropic material, $\mathsf C$ depends only on $E$ and $\nu$.

**Weak form.** Multiplying the equilibrium equation by a virtual displacement $\mathbf w$ that vanishes on $\Gamma_u$ and integrating by parts gives the principle of virtual work:

$$
\int_\Omega \boldsymbol\varepsilon(\mathbf w) : \mathsf C : \boldsymbol\varepsilon(\mathbf u)\, d\Omega
= \int_{\Gamma_t} \mathbf w \cdot \bar{\mathbf t}\, d\Gamma
\quad \text{for every admissible } \mathbf w.
$$

**Shape functions.** The solid is divided into ten-node tetrahedra (Tet10): four corner nodes and six mid-edge nodes. In volume coordinates $L_1, \dots, L_4$ (with $\sum L_i = 1$), the shape functions are quadratic:

$$
N_i = L_i\,(2L_i - 1) \ \text{(corners)}, \qquad N_{ij} = 4\,L_i L_j \ \text{(mid-edges)},
$$

and the displacement inside an element is interpolated from its nodal values, $\mathbf u^h = \sum_a N_a\,\mathbf u_a$. A quadratic displacement gives a strain that varies linearly across the element, which is exactly how bending strain varies through a beam's depth.

**Element matrices.** Writing the strain in terms of nodal displacements, $\boldsymbol\varepsilon = \mathbf B_e \mathbf u_e$, turns the weak form into element stiffness matrices and load vectors:

$$
\mathbf K_e = \int_{\Omega_e} \mathbf B_e^{\mathsf T}\, \mathbf D\, \mathbf B_e \, d\Omega, \qquad
\mathbf f_e = \int_{\Gamma_{t,e}} \mathbf N^{\mathsf T}\, \bar{\mathbf t}\, d\Gamma,
$$

where $\mathbf D$ is the elasticity matrix. Each $\mathbf K_e$ is 30 × 30 (ten nodes, three displacements each) and is integrated with four Gauss points.

**Assembly and supports.** Summing the element contributions into global positions gives

$$
\mathbf K\,\mathbf u = \mathbf f .
$$

The fixed face prescribes $\mathbf u_c = \mathbf 0$ on its nodes. Partitioning the unknowns into free (f) and constrained (c) sets,

$$
\mathbf K_{ff}\,\mathbf u_f = \mathbf f_f, \qquad \mathbf r_c = \mathbf K_{cf}\,\mathbf u_f - \mathbf f_c ,
$$

where $\mathbf r_c$ are the support reactions. Once the part is supported, $\mathbf K_{ff}$ is symmetric positive definite, and RunMat solves it with a sparse Cholesky factorization.

**Stress recovery.** Stress follows from the solved displacements, $\boldsymbol\sigma = \mathbf D\,\mathbf B_e\,\mathbf u_e$, at the four Gauss points of each element. The element stress is their volume-weighted average; the nodal stress extrapolates the Gauss-point values to the nodes and averages them between neighboring elements of the same material. The von Mises stress combines the components into one scalar, $\sigma_{vm} = \sqrt{\tfrac32\,\mathbf s : \mathbf s}$, where $\mathbf s$ is the deviatoric stress. The [structural physics reference](../physics/structural.md) covers the other structural procedures.

## Solution

The study file states the geometry, mesh, material, support, load and requested results. Each block below is an excerpt of `cantilever-static.fea`.

### Geometry

The beam is imported from a STEP file. Its faces carry names in the CAD, so the study refers to "Fixed end" and "Loaded end" instead of face numbers that change when the CAD is edited.

```fea source=cantilever-static.fea
geometry beam {
    file = "./geometry/cantilever.step"
    units = mm
    region body : solid { select name "Beam"; expect exactly_one }
    region fixed_end : face { select name "Fixed end"; expect exactly_one }
    region loaded_end : face { select name "Loaded end"; expect exactly_one }
}
```

`expect exactly_one` makes `runmat check` stop with a located message if a face is renamed or missing.

<!-- fea:example:cantilever-static:geometry -->

### Material

A linear static analysis needs only the elastic constants E and ν. The density is unused here; it serves gravity loads and modal studies of the same part.

```fea source=cantilever-static.fea
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
```

### Support and load

`structural.fixed` sets every displacement component on the end face to zero: the $\mathbf u_c = \mathbf 0$ of the formulation. The 1,000 N load is applied as a traction, 1,000 N over the 0.01 m² end face, or 100 kPa. The traction enters $\mathbf f$ through the face integral above, which spreads it over the face nodes consistently. A force at a single node would instead produce an unbounded stress beneath it.

```fea source=cantilever-static.fea
model loaded_beam {
    assign material.steel to geometry.beam.region.body
    constraint fixed_end {
        structural.fixed { target = geometry.beam.region.fixed_end }
    }
    load end_traction {
        traction on geometry.beam.region.loaded_end = [0, 0, -100000] Pa
    }
    load_case service { include load.end_traction }
}
```

### Mesh

Quadratic tetrahedra are chosen because the part bends: one Tet10 element already represents the linear variation of strain through the depth. A linear tetrahedron (Tet4) has constant strain, cannot bend within one element, and makes a bending part too stiff unless the mesh is very fine. A 150 mm target size gives 1,510 elements and 2,727 nodes, about 8,200 unknowns.

```fea source=cantilever-static.fea
mesh beam on geometry.beam {
    tetrahedra {
        order = quadratic
        target_size = 0.15 m
        maximum_deviation = 0.001 m
    }
}
```

`maximum_deviation` limits how far the mesh surface may depart from the CAD surface. It matters on curved parts and has no effect on this flat-sided beam.

### Study and outputs

The study selects a static structural analysis and names the results to publish: the displacement field, the element von Mises stress, the tip deflection, and a check of the tip deflection against the 0.5 mm limit.

```fea source=cantilever-static.fea
study cantilever_static {
    geometry = geometry.beam
    mesh = mesh.beam
    model = model.loaded_beam
    analysis static { structural static {} }
    output displacement = analysis.static.field.node.displacement
    output von_mises = analysis.static.field.element.von_mises_stress
    output tip_deflection = max(magnitude(analysis.static.field.node.displacement), over = geometry.beam.region.loaded_end)
    assessment deflection_limit {
        value = output.tip_deflection
        maximum = 0.5 mm
        evidence = authored_samples
    }
}
```

### Run

The complete example opens in RunMat Desktop with its source, the STEP file and saved results:

<!-- fea:example:cantilever-static:edit -->

From the command line, in the example folder:

```sh
runmat check cantilever-static.fea
runmat run cantilever-static.fea
```

`check` catches unit mistakes, broken references and missing CAD names in a few seconds, before any meshing. `run` imports the STEP file, meshes, assembles and solves $\mathbf K_{ff}\mathbf u_f = \mathbf f_f$, recovers stress, and publishes the outputs. See [check, plan and run](../language/index.md#check-plan-and-run).

### Reactions

The reactions at the fixed face sum to **[1,000 N](receipt:cantilever-static#support-reaction)** upward, equal and opposite to the applied load. Checking this first catches a load on the wrong face, a unit error or a support that restrains too much. Taking moments about the wall, the same reactions carry a moment of P L = 1,000 N·m.

### Deflection

<!-- fea:example:cantilever-static:displacement -->

Color shows the displacement magnitude, and the shape is exaggerated so the bending is visible. The free end drops and the beam curves smoothly from the wall, the shape $v(x)$ above. The tip moves down by **[0.2000 mm](receipt:cantilever-static#tip-deflection)**, the `tip_deflection` output, and the `deflection_limit` assessment passes.

### Stress

<!-- fea:example:cantilever-static:stress -->

Color shows the element von Mises stress. As beam theory predicts, it is highest near the wall and on the top and bottom surfaces, and falls toward the free end.

The largest element value, [5.39 MPa](receipt:cantilever-static#peak-element-von-mises), lies in an element on the fixed edge, and it is not a design value. Where a perfectly rigid face meets a free surface, linear elasticity predicts a stress singularity: the computed stress there keeps changing as the elements shrink, because no real support is perfectly rigid. A short distance from the wall the field is smooth. On the top surface at x = 100 mm the nodal von Mises stress is **[5.43 MPa](receipt:cantilever-static#top-von-mises)**; beam theory gives M c / I = 5.4 MPa at the same section.

### Answers

1. Tip deflection: [0.2000 mm](receipt:cantilever-static#tip-deflection).
2. Bending stress: [5.43 MPa](receipt:cantilever-static#top-von-mises) on the top surface at x = 100 mm, rising toward about 6 MPa at the wall by the linear moment distribution.
3. Wall reaction: [1,000 N](receipt:cantilever-static#support-reaction) upward and a moment of 1,000 N·m.
4. The deflection is 40 % of the 0.5 mm limit, and the stress is under 4 % of the 167 MPa allowable. The beam meets both criteria.

## Discussion

**Design meaning.** Both criteria pass, but by very different margins: the deflection uses 40 % of its limit, the stress under 4 %. The beam is governed by stiffness, which is typical of steel members sized to a deflection limit. A lighter section would reach the deflection limit long before the stress limit, so a redesign should keep $EI$ and remove material where it contributes little to $I$, for example by changing to an I-section.

**Model limits.** The analysis is linear: a 0.2 mm deflection on a 1 m span and a stress far below yield are well inside small-strain elasticity. The fixed face is an idealization. A real welded or bolted joint is somewhat flexible, which adds deflection at the tip and removes the singular peak at the edge. When the stress at the joint matters, model the joint itself: the weld fillet, the bolts, or the flexible support. The peak then becomes a real value that settles as the mesh is refined.

**Refinement.** A finite-element answer depends on the mesh, so the quantities a decision rests on should be checked on a finer mesh. Refining this beam from 150 mm to 50 mm changes the tip deflection by about 0.2 % and the stress at x = 100 mm by about 0.4 %, far less than the margins above. The [mesh refinement guide](./mesh-refinement.md) works through the full sequence, and [structural validation](../validation/structural.md) compares it with beam theory.

## Try a change

**Problem 1.1.** Replace the steel with aluminum (`youngs_modulus = 68.9 GPa`). Predict the tip deflection and the stress at x = 100 mm before running. Does the beam still meet the deflection limit?

*Answer.* The deflection scales with 1/E, to about 0.2000 × 200 / 68.9 ≈ 0.58 mm, which exceeds the limit. The stress barely changes, because the moment at each section depends only on the load.

**Problem 1.2.** Double the load to 200 kPa. Because the problem is linear, deflection, stress and reactions all double; one solution covers every load level by scaling.

**Problem 1.3.** Change the mesh to linear tetrahedra (`order = linear`) at a 50 mm target size and compare the tip deflection with the Tet10 result. It falls well short, at [0.1625 mm](receipt:cantilever-static#tip-deflection-tet4-50): constant-strain elements cannot represent bending and make the beam too stiff.

**Problem 1.4.** Put the stress criterion in the study, so every run reports both verdicts. The two outputs run today; an assessment on a value sampled at a point is [not yet in the current release](pending:output.point-sample-assessment).

```fea fragment
output top_stress = interpolate(analysis.static.field.node.von_mises_stress, at = [100 mm, 50 mm, 100 mm])
output support_reaction = resultant(analysis.static.field.node.reaction_force, over = geometry.beam.region.fixed_end)

assessment stress_allowable {
    value = output.top_stress
    maximum = 167 MPa
    evidence = authored_samples
}
```

## Next

- [Mesh refinement](./mesh-refinement.md): a convergence study on this beam.
- [Modal analysis](./modal-analysis.md): natural frequencies and mode shapes.
- [Structural physics reference](../physics/structural.md): every structural procedure and its equations.
- [Structural validation](../validation/structural.md): this beam against beam theory across mesh sizes.
