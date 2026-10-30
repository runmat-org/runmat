---
description: Worked mesh convergence study in RunMat. Refine a steel cantilever from 150 mm to 25 mm Tet10, check that tip deflection and stress away from the support converge, and see why the peak at a clamped edge does not.
keywords: [mesh refinement, mesh convergence, mesh independence, Tet10, Tet4, stress singularity, element order, target size, convergence study]
doc_type: guide
family: [structural]
examples: [cantilever-static]
---
A finite-element result depends on the mesh. A mesh convergence study solves the same model on finer meshes and checks that the quantity the decision rests on stops changing. Converged quantities are typically displacements, reactions, and stresses away from corners and supports. A stress at a sharp re-entrant corner or at the edge of a fully fixed face is a singularity: it grows without bound as the mesh is refined, so it never converges and is not a design value.

## The question

For the cantilever in the [linear static guide](./linear-static.md), is the mesh fine enough that:

- the tip deflection changes by less than **1 %** on further refinement, and
- the von Mises stress on the top surface at x = 100 mm changes by less than **2 %**?

<!-- fea:example:cantilever-static:setup -->

The beam is 1 m long with a 100 mm × 100 mm section, fixed at x = 0, with a 1,000 N downward traction resultant on the x = 1 m face. Only the mesh changes between runs: the target size steps from 150 mm to 25 mm, on quadratic tetrahedra (Tet10). A separate pair of runs uses linear tetrahedra (Tet4).

## Inspect the starting mesh

<!-- fea:example:cantilever-static:geometry -->

The 150 mm request gives 2,727 nodes. A target size is a request for an edge length, not an element count. Surface curvature, feature size and quality limits decide the final mesh, so compare node counts, not requests. At this beam's size, 75 mm gives almost the same mesh as 150 mm (2,758 nodes).

## Results

| Tet10 target size | Nodes | Tip deflection | Change vs 25 mm | Top-surface stress at x = 100 mm | Change vs 25 mm | Largest element stress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 150 mm | 2,727 | [0.20003 mm](receipt:cantilever-static#tip-deflection) | −0.32 % | [5.433 MPa](receipt:cantilever-static#top-von-mises) | +0.38 % | [5.39 MPa](receipt:cantilever-static#peak-element-von-mises) |
| 75 mm | 2,758 | [0.20030 mm](receipt:cantilever-static#tip-deflection-tet10-75) | −0.19 % | [5.422 MPa](receipt:cantilever-static#top-von-mises-tet10-75) | +0.18 % | [5.45 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-75) |
| 50 mm | 4,679 | [0.20047 mm](receipt:cantilever-static#tip-deflection-tet10-50) | −0.10 % | [5.413 MPa](receipt:cantilever-static#top-von-mises-tet10-50) | +0.01 % | [5.27 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-50) |
| 25 mm | 32,075 | [0.20068 mm](receipt:cantilever-static#tip-deflection-tet10-25) | — | [5.412 MPa](receipt:cantilever-static#top-von-mises-tet10-25) | — | [5.76 MPa](receipt:cantilever-static#peak-element-von-mises-tet10-25) |

The support reaction is [1,000 N](receipt:cantilever-static#support-reaction) on every mesh: equilibrium holds exactly at any resolution, so reactions are the first quantity to check and the last to worry about.

<!-- fea:example:cantilever-static:displacement -->

<!-- fea:example:cantilever-static:stress -->

### Element order

| Tet4 target size | Nodes | Tip deflection | Short of the Tet10 25 mm value |
| --- | ---: | ---: | ---: |
| 150 mm | 431 | [0.1007 mm](receipt:cantilever-static#tip-deflection-tet4-150) | 50 % |
| 50 mm | 729 | [0.1625 mm](receipt:cantilever-static#tip-deflection-tet4-50) | 19 % |

Linear tetrahedra have constant strain in each element, so they cannot bend; a coarse Tet4 mesh locks and is far too stiff. Quadratic tetrahedra represent the linear variation of bending strain through the depth, and the coarsest Tet10 mesh is already within 0.4 % of the finest.

## Check against the criteria

- **Tip deflection** changes by 0.32 % from 150 mm to 25 mm, inside the 1 % criterion at every level. The 150 mm mesh is fine enough for deflection.
- **Stress at x = 100 mm** changes by 0.38 %, inside the 2 % criterion. The 150 mm mesh is fine enough for this stress.
- **The largest element stress** moves up and down (5.39, 5.45, 5.27, 5.76 MPa) without settling. It sits at the fixed edge, where a fully fixed face meets a free surface; the exact solution there is singular. Each element reports its average, and the average over an ever-smaller element near the singular edge keeps changing. Read stress one section depth or more from such an edge, or model the real support stiffness.

The converged tip deflection, 0.2007 mm, is 0.34 % above Euler–Bernoulli beam theory's 0.2000 mm. Euler–Bernoulli theory neglects shear deformation, which adds a little deflection in this 10:1 beam.

## Run the sequence

Copy the cantilever study, change only `target_size` (or `order`), and add outputs for the quantities being compared:

```fea fragment
mesh beam on geometry.beam {
    tetrahedra {
        order = quadratic
        target_size = 50 mm
        maximum_deviation = 0.001 m
    }
}
```

```fea fragment
output top_stress = interpolate(analysis.static.field.node.von_mises_stress, at = [100 mm, 50 mm, 100 mm])
output support_reaction = resultant(analysis.static.field.node.reaction_force, over = geometry.beam.region.fixed_end)
```

Run each file, or declare the target size as a study input and run them as a [sweep](../language/inputs-sweeps-and-cases.md):

```sh
runmat run cantilever-50mm.fea --json
```

The `--json` output carries the node count and each output value, so a script can tabulate the sequence.

<!-- fea:example:cantilever-static:edit -->

## Try a change

- **Refine only near the support** with a `refine` block on the fixed face. The deflection and the stress at x = 100 mm should not change; the largest element stress at the edge will.
- **Run Tet4 at 25 mm** and compare its deflection and node count with the Tet10 runs: Tet4 needs many more nodes for the same accuracy in bending.
- **Move the stress sample to x = 20 mm.** Close to the singular edge, the sampled stress changes more between meshes.

## How RunMat computes this

Each run meshes the CAD to the requested size, assembles and solves the same weak form as the [linear static guide](./linear-static.md#how-runmat-computes-this), and recovers stress from each element's displacement gradient. The nodal value at x = 100 mm is interpolated from nodal stresses, each the average of the stress tensors of the elements around that node. The element value is each element's own average. On a smooth field both converge; at a singularity the nodal and element values diverge. See the [structural physics reference](../physics/structural.md) for the element formulation and recovery.

## Validation

[Structural validation](../validation/structural.md) lists the mesh sequences, references and acceptance criteria used to qualify the structural formulation.
