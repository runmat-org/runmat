# Studies, analyses and state

A study binds one geometry instance, its mesh, one model instance, and an ordered analysis graph.
It is the smallest independently runnable declaration.

## Study bindings

```fea
study bracket_static {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.loaded_bracket

    analysis static { structural static {} }
}
```

`geometry`, `mesh`, and `model` are required. The mesh must target the selected geometry. If the
geometry or model declares inputs, bind them with named arguments:

```fea
geometry = geometry.bracket(width = input.width)
model = model.loaded_bracket(service_load = input.load)
```

Unknown, duplicate, missing, or incorrectly typed arguments are errors.

## Analysis declarations

An analysis has a study-local name and exactly one typed procedure block:

```fea
analysis modes {
    structural modal {
        modes = 30
    }
    retain basis
}
```

The first word in the procedure block is the physics family; the second is the procedure. Closed
family and procedure names are checked before execution. Procedure fields are also typed: a
transient procedure requires time quantities, a harmonic procedure requires an ordered frequency
domain, and a modal procedure requires a positive mode count.

## Dependencies and retained state

Use `after` when an analysis depends on another analysis:

```fea
analysis modes {
    structural modal { modes = 30 }
    retain basis
}

analysis response after analysis.modes {
    structural modal_harmonic {
        basis = analysis.modes.state.basis
        frequencies = [20, 50, 100] Hz
    }
}
```

`after` establishes ordering. The `basis` reference establishes the typed state transfer. RunMat
validates the producer, state kind, topology, solution space, persistence, and consuming procedure;
it does not infer a modal basis from proximity in the file.

An analysis with one state output may rename it with `retain <name>`. Procedures with several state
outputs require an unambiguous state declaration rather than an inferred rename.

## Model objects used by an analysis

Procedure fields refer to model-owned objects through the bound model namespace, for example:

```fea
structural modal_response_spectrum {
    basis = analysis.modes.state.basis
    spectra = [model.response_spectrum.qualification_x]
}
```

Constraints, excitation cases, initial conditions, interfaces, and spectra are not copied into the
study. The study selects them; the model remains their owner.

## Current executable boundary

The language and canonical contracts cover the planned physics families. The native `.fea`
execution entrypoint currently executes structural procedure graphs. Other families may have
lower-level typed operations or numerical kernels without yet being runnable through the public
`.fea` command. Check [current status](../status) before treating language acceptance as execution
support or validation.
