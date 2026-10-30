---
description: Learn to judge mesh convergence in RunMat. Refine a steel cantilever from 150 mm to 25 mm Tet10, see which results settle and which do not, compare element orders, and decide when a mesh is fine enough for a design decision.
keywords: [mesh refinement, mesh convergence, mesh independence, Tet10, Tet4, stress singularity, element order, target size, convergence study]
doc_type: guide
family: [structural]
examples: [cantilever-static]
---
A finite-element result depends on the mesh. A convergence study solves the same model on finer meshes and checks that the quantity the decision rests on stops changing. Displacements, reactions and stresses away from corners and supports settle as the mesh is refined. A stress at a sharp re-entrant corner or at the edge of a fully fixed face does not: it grows without bound as the elements shrink, so it is never a design value.

This guide refines the cantilever from the [linear static guide](./linear-static.md) and decides when its mesh is fine enough.

## Is the cantilever mesh fine enough?

For the cantilever, is the mesh fine enough that:

- the tip deflection changes by less than **1 %** on further refinement, and
- the von Mises stress on the top surface at x = 100 mm changes by less than **2 %**?

<!-- fea:example:cantilever-static:setup -->

## How it's modeled: discretization error and singular points

Each element approximates the displacement with a simple polynomial: linear in a Tet4, quadratic in a Tet10. Over a smooth field, smaller elements follow the true shape more closely, the way more segments approximate a curve better, and the answer settles toward one value. Quadratic elements settle much faster per node, because a quadratic displacement already carries the linear variation of bending strain through a beam's depth.

The exception is a singular point. Linear elasticity predicts unbounded stress at a sharp re-entrant corner or where a perfectly rigid support meets a free surface. A smaller element there samples closer to that point and reports a larger value, so the peak keeps moving. The real part has a fillet radius or a support with some give, and its stress is finite; the singularity belongs to the idealized model.

## Model

The model is the [linear static cantilever](./linear-static.md#model-the-beam) unchanged: a 1 m beam with a 100 mm × 100 mm section, fixed at x = 0, with a 1,000 N downward traction resultant on the x = 1 m face. Only the mesh changes between runs, so any change in a result comes from the mesh alone.

Add outputs for the quantities being compared, so each run reports them directly:

```fea fragment
output top_stress = interpolate(analysis.static.field.node.von_mises_stress, at = [100 mm, 50 mm, 100 mm])
output support_reaction = resultant(analysis.static.field.node.reaction_force, over = geometry.beam.region.fixed_end)
```

## Mesh sequence

Step the target size down from 150 mm to 25 mm on quadratic tetrahedra (Tet10), and run a separate pair on linear tetrahedra (Tet4):

```fea fragment
mesh beam on geometry.beam {
    tetrahedra {
        order = quadratic
        target_size = 50 mm
        maximum_deviation = 0.001 m
    }
}
```

<!-- fea:example:cantilever-static:geometry -->

A target size is a request for an edge length, not an element count. Surface curvature, feature size and quality limits decide the final mesh, so compare node counts, not requests. On this beam, 75 mm gives almost the same mesh as 150 mm (2,758 nodes against 2,727). Halve the size between steps; a step that barely changes the node count tells you little.

## Run

Run each file, or declare the target size as a study input and run them as a [sweep](../language/inputs-sweeps-and-cases.md):

```sh
runmat run cantilever-50mm.fea --json
```

The `--json` output carries the node count and each output value, so a script can tabulate the sequence.

<!-- fea:example:cantilever-static:edit -->

## Results

| Tet10 target size | Nodes | Tip deflection | Change vs 25 mm | Top-surface stress at x = 100 mm | Change vs 25 mm | Largest element stress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 150 mm | 2,727 | [0.20003 mm](receipt:cantilever-static#tip-deflection) | −0.32 % | [5.433 MPa](receipt:cantilever-static#top-von-mises) | +0.38 % | [5.39 MPa](receipt:cantilever-static#peak-element-von-mises) |
| 75 mm | 2,758 | [0.20030 mm](receipt:cantilever-static#tip-deflection-tet10-75) | −0.19 % | [5.422 MPa](receipt:cantilever-static#top-von-mises-tet10-75) | +0.18 % | [5.45 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-75) |
| 50 mm | 4,679 | [0.20047 mm](receipt:cantilever-static#tip-deflection-tet10-50) | −0.10 % | [5.413 MPa](receipt:cantilever-static#top-von-mises-tet10-50) | +0.01 % | [5.27 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-50) |
| 25 mm | 32,075 | [0.20068 mm](receipt:cantilever-static#tip-deflection-tet10-25) | — | [5.412 MPa](receipt:cantilever-static#top-von-mises-tet10-25) | — | [5.76 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-25) |

Read the table column by column:

- **The reaction** is [1,000 N](receipt:cantilever-static#support-reaction) on every mesh. Equilibrium holds at any resolution, so reactions are the first thing to check and the last to worry about.
- **The tip deflection** creeps up slightly as the mesh is refined: a coarser mesh is a little too stiff. The change from 150 mm to 25 mm is 0.32 %.
- **The stress at x = 100 mm** settles to within 0.4 % across the sequence.
- **The largest element stress** moves up and down (5.39, 5.45, 5.27, 5.76 MPa) without settling. It sits at the fixed edge, the singular point described above.

<!-- fea:example:cantilever-static:stress -->

### Element order

| Tet4 target size | Nodes | Tip deflection | Short of the Tet10 25 mm value |
| --- | ---: | ---: | ---: |
| 150 mm | 431 | [0.1007 mm](receipt:cantilever-static#tip-deflection-tet4-150) | 50 % |
| 50 mm | 729 | [0.1625 mm](receipt:cantilever-static#tip-deflection-tet4-50) | 19 % |

Linear tetrahedra have constant strain in each element, so they cannot bend; a coarse Tet4 mesh locks and is far too stiff. The coarsest Tet10 mesh is within 0.4 % of the finest. For bending-dominated parts, change the element order before spending nodes on a finer Tet4 mesh.

## Deciding when to stop

- **Tip deflection** changes by 0.32 % from 150 mm to 25 mm, inside the 1 % criterion. The 150 mm mesh is fine enough for deflection.
- **Stress at x = 100 mm** changes by 0.38 %, inside the 2 % criterion. The 150 mm mesh is fine enough for this stress.
- **The peak at the fixed edge** will never pass a convergence criterion, and refining further would only move it again. Read stress one section depth or more from such an edge, or model the real support (its fillet, bolts or compliance) when the stress there matters.

The general practice: pick the quantity the decision rests on, refine until it changes by less than a tolerance well inside the design margin, and confirm that the quantity is not at a singular point. Spend refinement where it changes the answer: a [`refine` block](../meshing.md) near a fillet or a hole is cheaper than a finer mesh everywhere.

## Try a change

- **Refine only near the support** with a `refine` block on the fixed face. The deflection and the stress at x = 100 mm should not change; the largest element stress at the edge will.
- **Run Tet4 at 25 mm** and compare its deflection and node count with the Tet10 runs: Tet4 needs many more nodes for the same accuracy in bending.
- **Move the stress sample to x = 20 mm.** Close to the singular edge, the sampled stress changes more between meshes.

## Next

- [Linear static analysis](./linear-static.md): the model this guide refines.
- [Meshing](../meshing.md): element orders, sizes and local refinement.
- [Structural validation](../validation/structural.md): this sequence compared with beam theory.
