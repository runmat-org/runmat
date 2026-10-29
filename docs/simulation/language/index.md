# The `.fea` language

A `.fea` file defines an engineering analysis in source form. It names the geometry, regions,
mesh, material behavior, applied conditions, analysis sequence, requested results, and—when
needed—how independent work may be placed. RunMat checks those declarations before it starts CAD
translation, meshing, or a solver.

The language is declarative. It has typed declarations and expressions, physical quantities with
units, immutable references, and explicit imports. General computation remains in `.m` functions;
a `.fea` module can import one of those functions by file and use it through a bounded signal or
geometry-generation contract.

## The object model

Seven top-level declarations divide the study into parts with clear ownership:

| Declaration | What it defines |
| --- | --- |
| `geometry` | An imported, constructed, derived, or generated geometric model and its named regions |
| `mesh` | A meshing request for one geometry |
| `material` | Constitutive properties and their physical capabilities |
| `model` | Property assignments, constraints, loads, signals, excitations, interfaces, and initial conditions |
| `study` | One bound geometry, mesh, model, ordered analysis sequence, inputs, and outputs |
| `sweep` | Named input cases over one study |
| `execution` | Optional placement, resource, fan-out, partitioning, and resilience preferences |

The declarations are peers. A study refers to a model; the model is not nested inside the study.
This lets several studies reuse one physical model without copying it. Nesting is reserved for
objects owned by a declaration, such as regions inside geometry and loads inside a model.

## A first checked study

The smallest useful structural example still states its physical ownership explicitly:

The included STEP file carries the authored body and end-face names the study selects; `select name` resolves only names authored in the CAD file. The study stays in one readable source file; exporting the project preserves its CAD names.

```fea
geometry bracket {
    file = "./geometry/two-box-assembly.step"
    units = m

    region left_body : solid { select name "left_body"; expect exactly_one }
    region right_body : solid { select name "right_body"; expect exactly_one }
    region left_support : face { select name "left_support"; expect exactly_one }
    region left_load_face : face { select name "left_load_face"; expect exactly_one }
    region right_support : face { select name "right_support"; expect exactly_one }
}

mesh bracket on geometry.bracket {
    tetrahedra {
        order = linear
        target_size = 1.5 m
        maximum_deviation = 0.05 m
    }
}

material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}

model loaded_bracket {
    assign material.steel to geometry.bracket.region.left_body
    assign material.steel to geometry.bracket.region.right_body

    constraint left_support {
        structural.fixed { target = geometry.bracket.region.left_support }
    }
    constraint right_support {
        structural.fixed { target = geometry.bracket.region.right_support }
    }
    load service_traction {
        traction on geometry.bracket.region.left_load_face = [166.6666666667, 0, 0] Pa
    }
    load_case service { include load.service_traction }
}

study bracket_static {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.loaded_bracket

    analysis static { structural static {} }

    output displacement = analysis.static.field.node.displacement
    output von_mises = analysis.static.field.element.von_mises_stress
}
```

The checked-in copy is
[`structural-static.fea`](../examples/structural-static.fea). From the
repository root:

```sh
runmat check docs/simulation/examples/structural-static.fea
runmat plan docs/simulation/examples/structural-static.fea
runmat run docs/simulation/examples/structural-static.fea
```

`check` resolves imports and performs source and semantic checking. `plan` selects a study or sweep
and reports the logical work without doing it. `run` materializes geometry and mesh, validates the
realized model, executes the admitted numerical work, and publishes results and evidence.

## Read next

- [Files, modules, and imports](./files-modules-and-imports)
- [Types, units, and expressions](./types-units-and-expressions)
- [Geometry, models, and assignments](./geometry-models-and-assignments)
- [Studies, analyses, and state](./studies-analyses-and-state)
- [Inputs, sweeps, and cases](./inputs-sweeps-and-cases)
- [Outputs, fields, and observations](./outputs-fields-and-observations)
- [Execution, parallel work, and distributed work](./execution-parallel-and-distributed)
- [RunMat function integration](./runmat-function-integration)
- [Diagnostics and tooling](./diagnostics-and-tooling)

Use the [first static study guide](./guides/first-static-study) for a complete task and the
[declaration reference](./reference/declarations) for exact field rules.
