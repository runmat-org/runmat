# Diagnostics and tooling

The parser, semantic compiler, CLI, and language server use the same `.fea` frontend. An editor
diagnostic and `runmat check` therefore refer to the same source rule.

## Check before execution

```sh
runmat check bracket.fea
runmat check bracket.fea --study static
runmat check bracket.fea --sweep load_envelope --json
```

`check` parses the module graph, resolves imports, binds references, checks types and units, and
validates the authored geometry, model, procedure, output, sweep, and execution contracts. It also
applies every rule `runmat run` applies before meshing, through the same code, so a study that
checks clean is not refused later for a static reason, and a refused study carries the same
diagnostic code from both commands:

- the family and procedure must be one the release runs
  ([family and procedure matrix](../generated/family-procedure-matrix.md)); a planned combination
  names its `pending:<id>` key;
- the procedure graph, constraint kinds, initial conditions and output observations must be ones the
  selected family solves;
- `select name` must resolve against names authored in the CAD file, with the cardinality the
  region expects;
- `accelerators = required` must name families with a device solve path;
- `--study` or `--sweep` must name a root of the document; with several roots, `check` lists them.

It does not generate a mesh, validate element-level property coverage, or run a solver.

The command reports one of three states:

- **valid**: source and available project facts are sufficient for the selected root;
- **invalid**: a syntax, resolution, type, unit, topology, or semantic rule failed;
- **incomplete**: checking needs a project fact that is not available in the current host.

Machine output includes the root module, selectable studies and sweeps, status, and structured
diagnostics. Planning is a separate command.

`runmat check` accepts `.fea` files as well as `.m` files and prints located diagnostics in the
same form for both: the code and message, the `file:line:column` location, the source line with its
leading indentation removed, a caret underline with a label, and a summary line. A unit mismatch
names what the value must measure, what was found, and the expected SI unit:

```text
$ runmat check cantilever.fea
error[fea.semantic.model]: traction must be force per area, found N
 --> cantilever.fea:28:48
   |
28 | traction on geometry.beam.region.tip = [0, 0, -100000] N
   |                                        ^^^^^^^^^^^^^^^^^ expected Pa
checked cantilever.fea: 1 error(s), 0 warning(s)
```

A failure is reported once, where it occurs. A study that depends on a model which failed to check
is not reported again, and a value whose unit or reference already failed is not re-reported by
the declaration that uses it. `runmat check`, `plan`, and `run` print the same located
diagnostics to standard error before any geometry, meshing, or solve work starts; `--json` keeps the
structured form, including each diagnostic's span and label.

## Inspect the logical plan

```sh
runmat plan bracket.fea --study static
runmat plan bracket.fea --study static --json
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

## Rename a study symbol without breaking its users

Use the editor's **Rename Symbol** action on the declaration or any bound reference. Rename is
semantic: it follows the declaration identity established by the checked module graph, rather than
replacing matching text. A material named `steel` can therefore be renamed without changing an
unrelated model, region, or comment that happens to contain the same word.

For a `.fea` declaration, rename updates:

- its declaration;
- references in the declaring module;
- explicit imports and references in dependent `.fea` modules throughout the workspace;
- references reached through transitive imports.

The operation is rejected if the new name is a keyword, is not a `.fea` identifier, or would
collide with another declaration of the same kind in an affected scope. The language server checks
unopened `.fea` files under the workspace roots as well as unsaved editor overlays before it builds
the edit.

## Rename an imported RunMat function and its file

An imported function is one symbol across the `.m` and `.fea` languages. Rename the function from
its `.m` declaration, a RunMat call, the `.fea` import, or a bound `.fea` use:

```fea
import function impactPulse from "../signals/impactPulse.m"

signal impact {
    generated by function.impactPulse {
        arguments {
            scale = 4.0
        }
        domain = time
        quantity = acceleration
        sample at 0 s .. 20 ms step 0.1 ms
    }
}
```

If `impactPulse` becomes `dropPulse`, the workspace edit changes the function declaration and
RunMat calls, the `.fea` import and bound uses, and—when the source file is named
`impactPulse.m`—renames the file to `dropPulse.m` and updates import paths atomically. Package
qualifiers are retained; renaming `signals.impactPulse(...)` changes only its final identifier.

RunMat parameter names also cross the language boundary. Renaming `scale` in the function below
updates checked `.fea` named arguments, including arguments in `generated by function` blocks:

```matlab
function value = impactPulse(time, scale)
    arguments
        time (1,:) double
        scale (1,1) double
    end
    value = scale .* sin(2 .* pi .* 100 .* time);
end
```

Rename fails closed when the function definition is ambiguous, the replacement is invalid in
either language, or the new name would collide with an existing function, local binding, import,
declaration, or named argument. Save or resolve invalid source first: the server does not guess
through an unchecked study graph.

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

For example, this assignment fails in a study: a material on a face declares a shell domain, and
the model gives that shell no section or layup, so the solid body would be left without material:

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

Warnings do not convert unsupported execution into support. The [current status](../validation/status.md) page
separates language coverage, executable procedure coverage, and validation evidence.
