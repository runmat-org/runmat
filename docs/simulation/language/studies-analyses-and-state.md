---
description: How a .fea study binds geometry, mesh and model, orders analysis steps with after, and passes typed state such as a modal basis or a thermal preload between steps.
keywords: [.fea study, analysis steps, procedure graph, state transfer, modal basis, prestress, thermal preload, induction heating, load cases]
doc_type: language-reference
examples: [cantilever-static, tuning-fork-modal]
---
# Studies, analyses and state

A study is the smallest runnable unit: one geometry, its mesh, one model, and an ordered graph of analysis steps. Each step solves one procedure of one physics family and publishes typed state that later steps can consume, such as a modal basis, a preload's geometric stiffness or a temperature field.

## Bind geometry, mesh and model

```fea fragment
study bracket_static {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.loaded_bracket
    analysis static { structural static {} }
}
```

`geometry`, `mesh` and `model` are required, and the mesh must be a mesh of that geometry. When a geometry or model declares inputs or parameters, the study binds them by name:

```fea fragment
geometry = geometry.bracket(width = input.width)
model = model.loaded_bracket(service_load = input.load, body = geometry.bracket.region.body)
```

Unknown, duplicate, missing and wrongly typed arguments are check errors. The full member list is in the [declaration reference](./reference/declarations.md#study).

## Declare analysis steps

An analysis step has a study-local name and exactly one procedure block, `<family> <procedure> { <fields> }`:

```fea fragment
analysis modes {
    structural modal { modes = 30 }
    retain basis
}
```

The family and procedure names are a closed set, and their fields are typed: a transient step needs a time range and step, a harmonic step an increasing list of frequencies, a modal step a positive mode count. `runmat check` also verifies that the family runs the procedure, using the same registry the solver uses ([family and procedure matrix](../validation/status.md#families-and-procedures)). Every procedure and its fields are in the [procedure reference](./reference/procedures.md).

## Order steps and pass state

`after analysis.<step>` orders a step after another. A state binding names the exact result the step consumes:

```fea fragment
analysis modes {
    structural modal { modes = 30 }
    retain basis
}

analysis response after analysis.modes {
    structural modal_harmonic {
        basis = analysis.modes.state.basis
        frequencies = [20 Hz, 50 Hz, 100 Hz]
    }
}
```

`after` sets the order; `basis = analysis.modes.state.basis` sets the typed transfer. The checker verifies the producer, the state kind, the mesh and solution space, and that the state is complete and persisted. Nothing is inferred from the order of blocks in the file.

Each step publishes its state under a default name: `primary` for a displacement, temperature or field solution, `modal_basis` for modal steps, `geometric_stiffness` for preload steps. `retain <name>` publishes a step's single state under a chosen name, as `retain basis` does above.

Common step graphs:

| Graph | Steps |
| --- | --- |
| Prestressed modes | `structural preload` → `structural prestressed_modal { prestress = … }` |
| Linear buckling | `structural preload` → `structural buckling { prestress = …; modes = … }` |
| Modal superposition | `structural modal` → `structural modal_harmonic`, `modal_transient`, `modal_response_spectrum` or `random_response` with `basis = …` |
| Thermal prestress | `thermal steady` → `thermomechanical preload { temperature = … }` → `structural prestressed_modal` |
| Induction heating | `electromagnetic harmonic` → `thermal steady { heat_source = … }` |
| Preload, then nonlinear, then modes | `structural preload` → `structural nonlinear` → `structural prestressed_modal` [not yet in the current release](pending:structural.nonlinear.sequence) |

The thermal-prestress graph is the canonical example [`thermal-prestress.fea`](../examples/thermal-prestress.fea):

```fea source=thermal-prestress.fea
    analysis heat { thermal steady {} retain heated }
    analysis preload after analysis.heat {
        thermomechanical preload { temperature = analysis.heat.state.heated }
    }
    analysis modes after analysis.preload {
        structural prestressed_modal {
            prestress = analysis.preload.state.geometric_stiffness
            modes = 1
        }
    }
```

## Select model objects from a step

A step refers to objects the model owns through the model namespace; it does not copy them:

```fea fragment
analysis heat_up {
    thermal transient { start_time = 0 s; end_time = 60 s; time_step = 0.5 s }
    load_cases = [load_case.heater_on]
    initial_conditions = [model.initial.ambient]
}

analysis spectrum after analysis.modes {
    structural modal_response_spectrum {
        basis = analysis.modes.state.basis
        spectra = [model.response_spectrum.qualification_x]
    }
}
```

`load_cases` picks the load cases a step applies (`[]` applies none, and omitting it applies the model's load cases). Supports and prescribed values always apply. Use an empty list on a preload or heating step when a load is meant only for a later response step.

## What a study publishes

A run publishes every `output` the study declares, every assessment verdict, and the provenance of the run: source digests, mesh, solver settings, execution provider and diagnostics. Outputs name fields as `analysis.<step>.field.<location>.<quantity>` and reductions of them; see [outputs, fields and observations](./outputs-fields-and-observations.md). A sweep runs the study once per case; see [inputs, sweeps and cases](./inputs-sweeps-and-cases.md).
