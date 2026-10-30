---
description: Learn linear static analysis in RunMat on a steel cantilever. Model the part, mesh it, run it, read deflection, stress and reactions the way a practicing engineer does, decide whether to refine, and try design changes.
keywords: [linear static analysis, structural FEA, cantilever, tip deflection, von Mises stress, support reaction, Tet10, stress singularity, deflection limit, allowable stress, beam bending]
doc_type: guide
family: [structural]
examples: [cantilever-static]
---
A linear static analysis finds how far a part deflects, how stressed it is and what its supports carry under a steady load. It answers questions such as "is this bracket stiff enough?", "is this stress below the allowable?" and "what does this bolt have to hold?". It applies while strains stay small, the material stays elastic and contacts neither open nor close. Inertia, plasticity, large rotation and changing contact belong to the [transient, nonlinear and contact procedures](../language/reference/procedures.md).

This guide takes one part from the question to a design decision.

## 1. The engineering question

A 1 m steel cantilever with a 100 mm × 100 mm square section carries a 1,000 N downward load at its free end.

- Does the tip deflect less than **L/2,000 = 0.5 mm**?
- Does the bending stress stay below an allowable of **167 MPa** (250 MPa yield with a factor of safety of 1.5)?
- What force must the support carry?

<!-- fea:example:cantilever-static:setup -->

## The physics you already know

This is the beam from a first course in mechanics of materials. Beam theory gives the tip deflection and the bending stress directly:

$$
\delta = \frac{PL^3}{3EI}, \qquad \sigma = \frac{Mc}{I}, \quad M = P\,(L - x).
$$

The moment is largest at the support, so the stress is too, and it is highest on the top and bottom surfaces, where $c$ is largest. A hand estimate like this sets the size of answer to expect before anything runs.

Finite-element analysis solves the same equilibrium without the beam assumptions. It splits the part into small elements, writes the balance of internal and applied forces for each one, and assembles them into one linear system,

$$
\mathbf K\,\mathbf u = \mathbf f,
$$

where $\mathbf K$ is the stiffness of the meshed part, $\mathbf f$ the applied loads and $\mathbf u$ the displacement of every node. Stress follows from how the displacement varies inside each element. Nothing in this system assumes a slender beam, so the same steps work for a bracket, a housing or a casting, where no textbook formula exists. The [structural physics reference](../physics/structural.md) derives it in full.

## 2. Model it

A model is a set of choices: what shape to analyze, what it is made of, how it is held and how it is loaded. Each choice below is one block of the study file.

### Geometry

Import the part from CAD and name the faces that carry loads and supports. The names come from the STEP file, so the study refers to "Fixed end" and "Loaded end" instead of face numbers that change when the CAD is edited.

```fea source=cantilever-static.fea
geometry beam {
    file = "./geometry/cantilever.step"
    units = mm
    region body : solid { select name "Beam"; expect exactly_one }
    region fixed_end : face { select name "Fixed end"; expect exactly_one }
    region loaded_end : face { select name "Loaded end"; expect exactly_one }
}
```

`expect exactly_one` makes `runmat check` stop with a clear message when a face is renamed or missing in the CAD.

<!-- fea:example:cantilever-static:geometry -->

### Material

A linear static analysis needs only stiffness: Young's modulus and Poisson's ratio. The density is there for modal or gravity studies of the same part.

```fea source=cantilever-static.fea
material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}
```

### Supports and loads

Supports and loads are where a model most often departs from the real part, so choose them deliberately.

- **Support.** `structural.fixed` holds every displacement on the end face at zero: a perfectly rigid wall. A real bolted or welded joint is a little flexible, so a fully fixed face makes the part slightly too stiff near the support and creates an artificial stress peak along its edge (see step 5).
- **Load.** The 1,000 N end load is applied as a traction, a force per area, spread over the 0.01 m² end face: 1,000 N / 0.01 m² = 100 kPa. A load spread over a face is closer to how a real load arrives than a force at a single node, which would produce an unbounded local stress.

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

## 3. Mesh it

Choose the element type first, then the size.

- **Element order.** Use quadratic tetrahedra (Tet10) for parts that bend. A quadratic element represents the linear change of bending strain through the depth, so a few elements through the section are enough. A linear tetrahedron (Tet4) has constant strain, cannot bend inside one element, and makes a bending part far too stiff unless the mesh is very fine.
- **Size.** Start coarse. A 150 mm target size on this 1 m beam gives about 2,700 nodes and solves in seconds. Step 6 decides whether that is fine enough.

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

## 4. Run it

The study block ties the pieces together and names the results to publish: the displacement field, the element stress field and a tip-deflection output.

```fea source=cantilever-static.fea
study cantilever_static {
    geometry = geometry.beam
    mesh = mesh.beam
    model = model.loaded_beam
    analysis static { structural static {} }
    output displacement = analysis.static.field.node.displacement
    output von_mises = analysis.static.field.element.von_mises_stress
    output tip_deflection = max(magnitude(analysis.static.field.node.displacement), over = geometry.beam.region.loaded_end)
    assessment tip_deflection_matches_beam_theory {
        value = output.tip_deflection
        minimum = 0.19 mm
        maximum = 0.21 mm
        evidence = authored_samples
    }
}
```

The complete example opens in RunMat Desktop with its source, the STEP file and saved results:

<!-- fea:example:cantilever-static:edit -->

From the command line, in the example folder:

```sh
runmat check cantilever-static.fea
runmat run cantilever-static.fea
```

`check` catches unit mistakes, broken references and missing CAD names in a few seconds, before any meshing. `run` imports the STEP file, meshes, solves, and publishes the fields and outputs. `runmat plan` prints the work a run will do, and `--json` gives structured output for scripts. See [check, plan and run](../language/index.md#check-plan-and-run).

## 5. Read the results

Read the results in this order: reactions, then the deformed shape, then stress.

### Check the reactions first

The reaction at the fixed face sums to **[1,000 N](receipt:cantilever-static#support-reaction)** upward, equal and opposite to the applied load. A reaction that does not balance the load means the load went on the wrong face, a unit is off, or a support restrains something it should not. The reaction also answers the third question: the support carries 1,000 N, plus the bending moment that goes with it.

### Look at the deformed shape

<!-- fea:example:cantilever-static:displacement -->

Color shows the displacement magnitude. The shape is exaggerated so the bending is visible; the scale under the figure applies to the shape only. Check that the part bends the expected way: here the free end drops and the beam curves smoothly from the support. A shape that moves the wrong way, or a part that floats off, points to a load or support mistake.

The free end moves down by **[0.2000 mm](receipt:cantilever-static#tip-deflection)**, the `tip_deflection` output: 40 % of the 0.5 mm limit.

### Read stress where it means something

<!-- fea:example:cantilever-static:stress -->

Color shows the von Mises stress of each element. As beam theory predicts, the stress is highest near the support and on the top and bottom surfaces. The legend maximum, [5.39 MPa](receipt:cantilever-static#peak-element-von-mises), sits in an element right at the clamped edge.

Don't judge the design on that peak. Where a fully fixed face meets a free surface, the idealized model has a stress singularity: the computed stress there keeps changing as the mesh is refined, because a perfectly rigid wall does not exist. It is an artifact of the support model, not a property of the part. Judge the design on the stress a short distance away, about one section depth from the support, where the result is stable. On the top surface at x = 100 mm the stress is **[5.43 MPa](receipt:cantilever-static#top-von-mises)**, about 3 % of the 167 MPa allowable.

When the real stress at a joint matters, model the joint instead: the fillet, the bolt or the flexible support. The peak then becomes a real value that settles under refinement.

### The decision

| Criterion | Limit | Result |
| --- | --- | --- |
| Tip deflection | 0.5 mm (L/2,000) | [0.2000 mm](receipt:cantilever-static#tip-deflection) |
| Von Mises stress, top surface at x = 100 mm | 167 MPa | [5.43 MPa](receipt:cantilever-static#top-von-mises) |
| Support reaction | balances 1,000 N | [1,000 N](receipt:cantilever-static#support-reaction) |

Both criteria pass with a wide margin. The beam is governed by stiffness long before strength, which is typical of steel parts sized to a deflection limit.

## 6. Decide whether to refine

A finite-element answer depends on the mesh, so ask whether a finer mesh would change the quantity the decision rests on. Run the study again at a smaller `target_size` and compare. Here, refining from 150 mm to 50 mm changes the tip deflection by about 0.2 % and the stress at x = 100 mm by about 0.4 %, far less than the margins above, so the 150 mm mesh is fine enough for this decision. The peak at the clamped edge does not settle, as expected at a singularity.

A practical rule: refine until the quantity of interest changes by less than 1 to 2 % between meshes, and compare quantities that converge: displacements, reactions, and stresses away from sharp corners and idealized supports. The [mesh refinement guide](./mesh-refinement.md) walks through a full sequence.

## 7. Try a change

- **Double the load** to 200 kPa. The analysis is linear, so deflection, stress and reaction all double: one run covers every load level by scaling.
- **Use aluminum** (`youngs_modulus = 68.9 GPa`). The deflection grows by the stiffness ratio, 200 / 68.9 ≈ 2.9 times, to about 0.58 mm, which fails the limit. The stress barely changes, because the moment at each section is set by the load alone.
- **Switch to linear tetrahedra** (`order = linear`) and compare the tip deflection. It falls well short of the Tet10 value (Tet4 at a 50 mm target size gives [0.1625 mm](receipt:cantilever-static#tip-deflection-tet4-50)): the too-stiff behavior described in step 3.
- **Put the criteria in the study.** Outputs and assessments report a pass or fail verdict with every run, so a change that breaks a limit shows up at once:

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

## Next

- [Mesh refinement](./mesh-refinement.md): a full convergence sequence on this beam.
- [Modal analysis](./modal-analysis.md): the natural frequencies of a part.
- [Structural physics reference](../physics/structural.md): the equations, elements and solver.
- [Structural validation](../validation/structural.md): this beam against beam theory across mesh sizes.
