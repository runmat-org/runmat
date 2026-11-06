---
description: In RunMat, a folder is a module. Designs in one folder name each other with no import; `use` brings in designs from other folders and +package folders. Resolution rules, aliases, pins and errors.
keywords: [modules, folders, use, packages, +package, imports, name resolution, design reuse, libraries, monorepo CAD]
---
# Folders, modules and `use`

A folder is the unit of scope. Every design in a folder (`.part`, `.assembly`, `.drawing`, `.circuit`, `.fea`) and every `.m` function in it can be named from any other file in that folder with no import. Designs in other folders come in with `use`, which names a folder or a package, never a file. The file that holds a design can be split, merged or renamed without editing anything that uses it.

## One folder

This folder holds `plate.part`, a package folder `+hardware/` with `standoff.part`, and `base.assembly`. The assembly places the plate by name with no `from`, and the standoff from the `hardware` package.

```part file=plate.part
runmat 0.7
std 1

part plate {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])
        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])
        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s, distance = 6 mm)
    output plate = { body = feature.body.body, part_number = "PL-080" }
}
```

```part file=+hardware/standoff.part
runmat 0.7
std 1

part standoff {
    sketch s on origin.xy {
        c = circle(center = [0 mm, 0 mm], radius = 3 mm)
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
    output standoff = { body = feature.body.body, part_number = "SO-M3-10" }
}
```

```assembly file=base.assembly
runmat 0.7
std 1

assembly base {
    use part plate as plate
    use part standoff from hardware as post {
        placement = { placement = offset, offset = { translation = [10 mm, 10 mm, 6 mm] } }
    }
}
```


## Other folders and packages

A design in another folder comes in with `from` and a folder path, or through a folder alias. Here `fixtures/clamp.part` is placed twice: once by path and once through the alias `fx`.

```part file=fixtures/clamp.part
runmat 0.7
std 1

part clamp {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])
        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 12 mm])
        l3 = line(start = [20 mm, 12 mm], end = [0 mm, 12 mm])
        l4 = line(start = [0 mm, 12 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s, distance = 8 mm)
    output clamp = { body = feature.body.body, part_number = "CL-20" }
}
```

```assembly file=fixture.assembly
runmat 0.7
std 1

use "./fixtures" as fx

assembly fixture {
    use part plate as plate
    use part clamp from "./fixtures" as clamp_a {
        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 6 mm] } }
    }
    use fx.part.clamp as clamp_b {
        placement = { placement = offset, offset = { translation = [60 mm, 0 mm, 6 mm] } }
    }
}
```

## The forms of `use`

| Form | Brings in |
|---|---|
| `use part plate as plate` | the `part plate` of this folder, in any file of it |
| `use part clamp from "./fixtures" as clamp_a` | a design in another folder; `..`, `../wing` and `./a/b` work |
| `use part skeleton from "//wing/skeleton" as sk` | a folder by its path from the project root |
| `use part skeleton from "//wing/skeleton" @ "R3" as sk` | the same, pinned to the released revision `R3` |
| `use part standoff from hardware as post` | the `+hardware/` package (`a.b` is `+a/+b/`) |
| `use "./fixtures" as fx` (at file level), then `use fx.part.clamp as clamp_b` | a folder alias and a design through it |
| `use material aluminum from "./materials"` | a material from another folder |

- **In an assembly**, `use part <name> as <occurrence> { … }` is also the occurrence: it places the design and its block holds the placement, configuration and other occurrence fields. See [Occurrences and mates](/docs/mcad/assembly/mates).
- **Kinds:** `part`, `assembly` and `material`. The kind keyword is part of the name: `use part x` never finds an assembly called `x`.
- **Drawings and interface frames** name designs the same way: `model = { part = hinge_mount }`, `model = { assembly = drone, from = ".." }`, `to = { frame = interface, part = skeleton, name = wing_ref, from = "./interfaces" }`.

## How a name resolves

1. **The same folder.** Every design file in the folder, and every `.m` function in it (including `private/`).
2. **A `from` path or an alias.** The folder it names, relative to the referring file, or from the project root with `//`.
3. **A package.** `from hardware` looks for `+hardware/` in the referring file's folder, then in each folder above it up to the project root, and takes the first it finds. A `+hardware` in `cabin/` therefore shadows the project's `+hardware` for designs in `cabin/`, while other folders still reach the project's.

Subfolders are not in scope until a `use` names them. A `.m` function outside the folder comes from the MATLAB path, as in MATLAB: the project's source roots, its dependencies, then `+pkg` folders as `pkg.fn`.

## Errors

| Problem | Diagnostic |
|---|---|
| The design is not in the folder | *no `part platter` in this folder (did you mean part.plate?)*, at the `use` line |
| Two designs of one kind and name in one folder | an error naming both files |
| Folders that `use` each other in a cycle | an error at the `use` that closes the cycle. Shared and skeleton folders must flow one way. |

## Printing and edits

Structured edits from the editor's commands and the agent reprint an assembly or drawing in the forms above.

## Boards and their circuits

A board file belongs to one circuit and names it by symbol, as other designs do: `use circuit flight_controller` at the top of `flight_controller.circuit.layout` names the circuit that a `.circuit` file in the same folder declares. A circuit reuses another circuit, sheet or interface the same way: `use circuit power`, or `use sheet buck from "./power" as reg` for a sheet of a circuit in the folder `power`. Naming a design by its file (`from "./power.circuit"`) is an error that gives the symbol to write. See [Board layout](/docs/ecad/layout).
