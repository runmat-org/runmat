# Diagnostics and tooling

The parser, semantic compiler, CLI, and language server use the same `.fea` frontend. An editor
diagnostic and `runmat fea check` therefore refer to the same source rule.

## Check before execution

```sh
runmat fea check bracket.fea
runmat fea check bracket.fea --study static
runmat fea check bracket.fea --sweep load_envelope --json
```

`check` parses the module graph, resolves imports, binds references, checks types and units, and
validates the authored geometry, model, procedure, output, sweep, and execution contracts. It does
not translate CAD, generate a mesh, validate element-level property coverage, or run a solver.

The command reports one of three states:

- **valid**: source and available project facts are sufficient for the selected root;
- **invalid**: a syntax, resolution, type, unit, topology, or semantic rule failed;
- **incomplete**: checking needs a project fact that is not available in the current host.

Machine output includes the root module, selectable studies and sweeps, status, and structured
diagnostics. Planning is a separate command.

## Inspect the logical plan

```sh
runmat fea plan bracket.fea --study static
runmat fea plan bracket.fea --study static --json
```

`plan` selects one root and reports its bound geometry, mesh, model, execution policy, case count,
and logical phases. It does not perform those phases. Realized geometry, mesh quality, exact
property projection, solver admission, and numerical resource evidence appear only when their work
has actually run.

## Language-server features

For `.fea` files the shared language server provides:

- syntax highlighting and semantic tokens;
- source and semantic diagnostics;
- declaration and member completion;
- hover types, quantities, units, and declaration ownership;
- operation signature help;
- document and workspace symbols;
- go to definition and references across `.fea` modules and imported `.m` functions;
- safe rename;
- canonical formatting.

The formatter is idempotent and preserves comments. Editor overlays may be checked for immediate
feedback; execution uses a saved, frozen project revision.

## Read diagnostics literally

Diagnostics identify a phase and source span. Correct the declaration that owns the problem rather
than suppressing it downstream. Typical failures include:

- a region with the wrong topology role or cardinality;
- a quantity with the wrong physical dimension;
- a model input that was not bound;
- incomplete or overlapping property ownership;
- an analysis consuming a missing or incompatible state;
- a field operation invalid for its location or quantity;
- a sweep case outside an input bound;
- a placement request unavailable to the selected procedure.

For example, this assignment fails because a bulk material requires a solid, shell, beam, or cell
domain; a face is a boundary target:

```fea
model incorrect {
    assign material.steel to geometry.bracket.region.load_face
}
```

Assign the material to the solid region and keep the face for the load:

```fea
model corrected {
    assign material.steel to geometry.bracket.region.body
    load service_force {
        force on geometry.bracket.region.load_face = [0, -1, 0] kN
    }
    load_case service { include load.service_force }
}
```

Warnings do not convert unsupported execution into support. The [current status](../status) page
separates language coverage, executable procedure coverage, and production qualification.
