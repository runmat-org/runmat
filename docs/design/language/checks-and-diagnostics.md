---
description: How RunMat checks design files and reports problems. Root causes and cascades, diagnostic codes, typed repairs, and the circuit checks that gate continuous integration.
keywords: [diagnostics, design check, root cause, regeneration errors, ERC, DRC, CI, continuous integration, error codes, repairs]
---
# Checks and diagnostics

The editor checks a design file on every change, and the agent and the command line run the same check. A check regenerates the design: a part rebuilds its feature history, an assembly resolves its occurrences and solves its mates, a drawing compiles against the models its views show, and a circuit elaborates and runs its electrical rules. The result lists every problem with where it is written and what it affects.

## What a diagnostic holds

| Field | Meaning |
|---|---|
| `severity` | `error`, `warning` or `info`. An error stops the design from regenerating cleanly. |
| `code` | A stable code, such as `cad.ref.not_generated` or `cad.asm.mate_conflict`. |
| `message` | What is wrong, naming the item: *`body` ran but no longer makes `body.side(l9)`*. |
| `cause` | `root` (fix this one) or `cascade` (failed because of a root). |
| `root` | For a cascade: the root diagnostic it follows from. |
| `subject` | The item it is about, by canonical ID: `feature.round`, `mate.hinge`, `occurrence.wing/rib@ribs[2]`, `U3`. |
| `failed_features` | For a root: the features that failed or were blocked because of it. |
| `range` | Where it is written in the file. |
| `repairs` | Typed fixes the compiler offers; the editor applies one with a click, and the agent can apply it too. |

A broken reference fails its own feature and blocks the features after it that use what it made. The check reports one root and lists the blocked features under it, so a single broken fillet never reads as a dozen errors.

## Example: a reference that no longer resolves

The fillet below names the side face of a sketch line `l9` that the sketch does not have. The check fails at the fillet, and the rest of the part still regenerates.

```part fails
runmat 0.7
std 1

part bracket {
    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])
        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])
        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = 10 mm)
    feature round = fillet(
        edges = [edges_between(feature.body.face.side(s1.l9), feature.body.face.end)],
        radius = 1 mm
    )
    output bracket = { body = feature.body.body, part_number = "BR-100" }
}
```

The check reports *`body` ran but no longer makes `body.side(l9)`* as a root cause on `feature.round`, with the range of the reference.

## Code families

| Prefix | Covers |
|---|---|
| `cad.ref.*` | References: an entity the maker no longer makes (`not_generated`), a reference to a later feature (`forward_reference`), ambiguous or empty selections, kind mismatches |
| `cad.xref.*` | Reads from other files: an unknown file, a name it does not publish, a missing pinned revision |
| `cad.sketch.*` | Sketches: over- or under-constrained, redundant constraints, open profiles |
| `cad.regen.*`, `cad.kernel.*` | Regeneration: cascades, operations the geometry kernel could not build |
| `cad.region.expectation` | A region that found more or fewer entities than its `expect` |
| `cad.asm.*` | Assemblies: unknown designs and occurrences, cycles, mates that conflict, are redundant or leave parts under-defined, configuration problems, interference and clearance |
| `cad.fastener.*` | Fastener sets: paths, exceptions, fits and tables |
| `cad.pmi.*`, `cad.drawing.*` | PMI references and semantics, drawing references, hole tables |
| `cad.property.*` | Missing or invalid properties |
| `design.value.*` | Values from `.m` functions: unit mismatches, nondeterminism, failures |

A circuit check reports ERC findings (unconnected pins, outputs driven together, missing power flags), and a board check adds DRC findings, unrouted connections and schematic changes the board has not taken yet. See [Rules and checks](/docs/ecad/circuit/rules-and-checks).

## Checking from the command line

`runmat-design check` checks circuits and boards, prints each finding, and exits with status 1 when any file has an error, so a CI job or a coding agent can gate on it:

```sh
runmat-design check board/flight_controller.circuit.layout
runmat-design check board/*.circuit --json
```

| Option | Effect |
|---|---|
| `--json` | Print the reports as JSON. |
| `--no-simulate` | Skip the simulation sections' measures. |
| `--interference` | On an assembly, also check for overlapping parts. |
| `--root <dir>` | The workspace root, when it is not the file's folder. |

`runmat check <file>.circuit` and `runmat test` run the same circuit checks beside `.m` and `.fea` files, with the same `--no-simulate` option.

`runmat-design check` also checks `.part`, `.assembly` and `.drawing` files with the editor's check: it regenerates the part, resolves the assembly's occurrences, mates and patterns, or compiles the drawing's views, prints each diagnostic with its line, and exits with status 1 on any error. `--interference` also checks an assembly for parts that share material, the Evaluate tab's interference check, and fails on any overlap; parts that only touch don't count. Every body of a multi-body part is checked, and each overlap names the first pair of bodies found. Bodies of compressible fill (batt and loose-fill insulation) are left out and counted, because a cable, pipe or box in an insulated cavity displaces the fill:

```sh
runmat-design check house/house.assembly --interference --root house
```

`runmat check` and `runmat test` take those files too [not yet in the current release](pending:design.tools.runmat-check-design).
