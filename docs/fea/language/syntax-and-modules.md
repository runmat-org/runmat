# Syntax, Units, and Modules

Declarations use braces; assignments use `=`; quantities carry units at the point of use. Commas are used inside lists and calls. Comments use the language's ordinary line-comment syntax supported by the formatter and editor.

```fea
material aluminum {
    mass_density = 2700 kg/m^3
    elastic {
        youngs_modulus = 69 GPa
        poisson_ratio = 0.33
    }
}
```

The checker normalizes compatible quantities to SI while retaining authored provenance. It rejects a value with the wrong physical dimension: a pressure cannot be supplied where a force is required, and a bare scalar cannot stand in for a length.

## Names And References

Declarations have stable ids such as `bracket`, `steel`, or `static`. References name both the declaration kind and id:

```fea
geometry = geometry.bracket
mesh = mesh.bracket
model = model.loaded_bracket
assign material.steel to geometry.bracket.region.body
```

Closed behavior—physics families, procedures, topology roles, element order, interpolation, and placement preferences—is parsed as typed language syntax rather than dispatched by arbitrary strings.

## Imports

Import one declaration from another `.fea` module:

```fea
import geometry bracket from "./geometry.fea"
import mesh bracket from "./geometry.fea"
import material steel from "./materials.fea"
import model mounting from "./mounting.fea"
```

Import one exact RunMat function from an `.m` file:

```fea
import function impactPulse from "./impactPulse.m"
```

Paths are relative to the importing module and confined to the frozen project source root. The compiler records the logical module path, source revision, content digest, declaration span, signature, effects, and capabilities. An identically named function in a different file cannot satisfy the import.

Imports must precede declarations. Cycles, missing symbols, wrong declaration kinds, paths outside the project, signature drift, and changed source bytes fail closed.

## Formatting And Editor Support

The canonical formatter is idempotent. RunMat's language server supplies diagnostics, formatting, symbols, completion, hover, semantic tokens, definitions, references, and signature help for `.fea`; imported `.m` definitions participate in navigation and rechecking. Editor overlays are checked for feedback, while execution uses the saved frozen project revision.
