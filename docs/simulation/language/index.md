---
description: The .fea study language describes a finite-element study as typed declarations with units, which runmat check, plan and run verify, resolve and solve.
keywords: [.fea language, FEA study file, declarations, runmat check, runmat run, finite element study, simulation language]
doc_type: language-reference
examples: [cantilever-static, two-body-static]
---
# The `.fea` study language

A `.fea` file describes a finite-element study in source form: the geometry and its named regions, the mesh, the materials, the loads and supports, the analysis steps, the results to publish, and, when needed, how the work is placed on cores, GPUs and workers. RunMat checks the whole description before it translates CAD, meshes or solves, so an error in a unit, a region name or a procedure is reported in seconds, at its line.

The language is declarative and typed. Every value has a type, every physical value has a unit, and every reference is a path that the checker resolves, such as `geometry.beam.region.fixed_end`. Studies are plain text, so they diff, review and version like code. General computation stays in `.m` functions: a `.fea` module imports a function by file and uses it to generate a signal, a spectrum or a geometry.

RunMat Desktop, the `runmat` command line and the RunMat agent all read and write the same `.fea` files.

## What a `.fea` file declares

A file is a set of top-level declarations. Each one owns one part of the study:

| Declaration | What it defines | Reference |
| --- | --- | --- |
| `geometry` | An imported, constructed, derived or generated geometric model, and its named regions | [geometry](./reference/declarations.md#geometry) |
| `mesh` | The meshing request for one geometry: element shape and order, target size, surface deviation and local refinement | [mesh](./reference/declarations.md#mesh) |
| `material` | Constitutive properties: elastic, plastic, thermal, electrical, magnetic, acoustic and fluid behavior, and damping | [material](./reference/declarations.md#material) |
| `model` | Material assignments, sections, constraints, loads, signals, excitations, interfaces, connectors and initial conditions | [model](./reference/declarations.md#model) |
| `study` | One geometry, mesh and model; the ordered analysis steps; inputs, outputs and assessments | [study](./reference/declarations.md#study) |
| `sweep` | Named cases of one study's inputs, combined as a product or element by element | [sweep](./reference/declarations.md#sweep) |
| `execution` | Placement, resources, fan-out, partitioning and resilience for a study's work | [execution](./reference/declarations.md#execution) |
| `table` | Tabulated or modeled data over a domain that boundaries refer to, such as an acoustic impedance over frequency | [table](./reference/declarations.md#table) |

The declarations are peers. A study refers to a model by name; the model is not nested inside the study, so several studies can share one model, mesh or material without copying it. Nesting is reserved for what a declaration owns: regions inside a geometry, loads inside a model, analysis steps inside a study.

## A complete study

This study fixes one end of a steel cantilever, applies a 100 kPa downward traction to the other end, and solves a linear static step on quadratic tetrahedra (Tet10). It publishes the displacement and von Mises stress fields and the peak tip deflection, and it checks that deflection against a 0.5 mm design limit (L/2,000).

```fea source=cantilever-static.fea
geometry beam {
    file = "./geometry/cantilever.step"
    units = mm
    region body : solid { select name "Beam"; expect exactly_one }
    region fixed_end : face { select name "Fixed end"; expect exactly_one }
    region loaded_end : face { select name "Loaded end"; expect exactly_one }
}

mesh beam on geometry.beam {
    tetrahedra {
        order = quadratic
        target_size = 0.15 m
        maximum_deviation = 0.001 m
    }
}

material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}

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

Read it top to bottom:

- **`geometry`** imports `cantilever.step`, whose length unit is the millimeter, and names three regions. `select name` resolves a name authored in the CAD file, and `expect exactly_one` makes the check fail if the name matches no entity or several.
- **`mesh`** asks for quadratic tetrahedra with a 0.15 m target edge length and at most 0.001 m deviation from the CAD surface.
- **`material`** gives steel its density and isotropic elastic constants. The unit is part of each value: `200 GPa` and `200e9 Pa` are the same quantity.
- **`model`** assigns the material to the body, fixes one end face, applies the traction to the other, and groups the load in a load case.
- **`study`** binds the geometry, mesh and model, runs one `structural static` step named `static`, and publishes results by path, `analysis.<step>.field.<location>.<quantity>`. The `assessment` turns the tip deflection into a pass or fail verdict.

The file is the canonical example [`cantilever-static.fea`](../examples/cantilever-static.fea). The [linear static guide](../guides/linear-static.md) walks through modeling, running and reading it. A two-body version, with two material assignments and two supports, is [`structural-static.fea`](../examples/structural-static.fea).

## Check, plan and run

Three commands take a study from source to result. From the repository root:

```sh
runmat check docs/simulation/examples/cantilever-static.fea
runmat plan docs/simulation/examples/cantilever-static.fea
runmat run docs/simulation/examples/cantilever-static.fea
```

| Command | What it does | What it leaves out |
| --- | --- | --- |
| `runmat check` | Parses the file and its imports, resolves every reference, checks types and units, resolves regions against the names in the CAD file, and checks that each physics family runs each requested procedure. Reports every error with its file, line and column. | Meshing and solving. |
| `runmat plan` | Selects one study or sweep and prints its logical phases and analysis definition: geometry, mesh, each analysis step, state transfers, outputs and placement. | Meshing and solving. |
| `runmat run` | Imports the CAD, meshes, solves each analysis step, evaluates outputs and assessments, and publishes the result with its provenance. | |

When a file declares more than one study or sweep, choose one with `--study <name>` or `--sweep <name>`. `--json` prints structured output for tools and scripts. The same `check` and `run` commands take `.m` scripts, so a project that mixes `.m` code and `.fea` studies uses one workflow. The [diagnostics page](./diagnostics-and-tooling.md) lists the messages and the editor support: completion, hover, go to definition and rename.

## Rules that hold everywhere

- **Every physical value carries a unit.** The checker converts compatible units and rejects incompatible ones, so `target_size = 2 kg` is an error. Dimensionless values, such as Poisson's ratio, are bare numbers. The [units table](./types-units-and-expressions.md) lists every accepted spelling.
- **References are typed paths.** `geometry.beam.region.loaded_end` is a face region, `material.steel` a material, `analysis.static.field.node.displacement` a nodal vector field. A reference to something of the wrong kind is a check error.
- **Region names come from the CAD file.** `select name "Loaded end"` matches a name authored in the STEP or IGES file, `select tag` matches a presentation layer, and `select persistent_id` matches a canonical entity id (`persistent-entity:sha256:…`). Inside the solver, every region is identified by its canonical id. See [geometry, models and assignments](./geometry-models-and-assignments.md).
- **Nothing runs until everything checks.** CAD import, meshing and solving start only after the study and every module it imports check clean.
- **Reuse is explicit.** A file imports exactly the declarations it uses (`import model mounting from "./models/mounting.fea"`), and a model can include another model's declarations. See [files, modules and imports](./files-modules-and-imports.md).

## The same objects from `.m` code

The `fea.*`, `geometry.*` and `meshing.*` functions build studies from `.m` code: `fea.study` binds a geometry, a model and execution settings, and `fea.run` solves the study. After a run, `fea.results` loads the saved result by its run id and `fea.field` selects a field from it for plotting or further computation. See the [function reference](/docs/reference/builtins).

## Read next

Language pages:

- [Types, units and expressions](./types-units-and-expressions.md)
- [Files, modules and imports](./files-modules-and-imports.md)
- [Geometry, models and assignments](./geometry-models-and-assignments.md)
- [Studies, analyses and state](./studies-analyses-and-state.md)
- [Inputs, sweeps and cases](./inputs-sweeps-and-cases.md)
- [Outputs, fields and observations](./outputs-fields-and-observations.md)
- [Execution, parallel work and distributed work](./execution-parallel-and-distributed.md)
- [RunMat function integration](./runmat-function-integration.md)
- [Diagnostics and tooling](./diagnostics-and-tooling.md)

Reference:

- [Declarations](./reference/declarations.md): every declaration and member, with types, units and defaults.
- [Procedures](./reference/procedures.md): every analysis step, by physics family.
- [Grammar](./reference/grammar.md): the syntax in EBNF.
- [Glossary](../glossary.md) and [release status](../validation/status.md).
