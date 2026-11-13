# Files, modules and imports

Each `.fea` file is a module. A module may declare several geometries, meshes, materials, models,
studies, sweeps, and execution policies. There is no one-study-per-file rule and no file-wide
include.

## Declarations and names

A top-level declaration has a kind, a name, and a body:

```fea
material aluminum_6061 {
    mass_density = 2700 kg/m^3
    elastic {
        youngs_modulus = 68.9 GPa
        poisson_ratio = 0.33
    }
}
```

Namespaces are explicit in references. `material.aluminum_6061` cannot be confused with a model or
geometry that happens to use the same local name. Nested declarations use their owning namespace,
such as `geometry.bracket.region.mounting_face` or `model.bracket.constraint.fixed_support`.

Names are identities, not display labels. Rename a declaration through the language server so its
references change with it.

## Import one symbol

An import states the expected declaration kind, the local name, and the source file:

```fea
import geometry bracket from "./geometry/bracket.fea"
import mesh bracket from "./geometry/bracket.fea"
import material aluminum from "./materials/aluminum.fea"
import model mounting from "./models/mounting.fea"
```

Importing `material aluminum` does not import the other declarations in the file. Imports precede
local declarations. Wildcard imports do not exist.

Paths are resolved relative to the importing file and must remain inside the admitted project or
dependency source graph. RunMat rejects missing symbols, a symbol of the wrong kind, import cycles,
paths outside the project, and changed source bytes after the project revision has been frozen.

## Import a RunMat function

Use the same form for a named function defined by one `.m` file:

```fea
import function impactPulse from "./impactPulse.m"
```

The file must provide the named function entrypoint with a compatible typed signature. A script,
an ambiguous definition, unapproved effects, or a changed source digest fails before numerical
execution. See [RunMat function integration](./runmat-function-integration) for generated signals
and geometry.

## Composition is explicit

A model may include another model:

```fea
import model mounting from "./mounting.fea"

model bracket {
    include model.mounting
}
```

Inputs on the included model are bound by name. Composition retains the defining module for every
object and rejects cycles, duplicate physical ownership, and conflicting assignments. Source order
does not define precedence.

## Source identity

The compiler records each logical module path, source digest, declaration span, imported symbol,
and resolved dependency. Workers receive the resulting typed contracts and content-addressed
artifacts; they do not receive source text and do not resolve imports again.
