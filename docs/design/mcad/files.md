---
description: Reference for RunMat MCAD files. Every statement a .part, .assembly and .drawing file can hold, what it does, and where its full reference lives, with examples that check.
keywords: [.part file, .assembly file, .drawing file, MCAD file format, text CAD format, parametric part file, assembly file format, drawing file format]
---
# MCAD file reference

RunMat MCAD keeps each part, assembly and drawing in its own text file. This page lists every statement each file kind takes; the linked pages hold the full reference with examples. The [design language](/docs/design-language) pages cover what all design files share: the header, values and units, names, folders and `use`.

## `.part`

A part file declares one part: `part <name> { … }`. The name matches the file name (`bracket.part` holds `part bracket`).

| Statement | Declares | Reference |
|---|---|---|
| `input <name> : <kind> = <default>` | a parameter | [Values and units](/docs/design-language/values-and-units#inputs) |
| `let <name> = <value>` | a named value, which may call a `.m` function | [Values and units](/docs/design-language/values-and-units#let-values), [MATLAB functions](/docs/design-language/matlab-functions) |
| `sketch <name> on <plane> { … }` | a constrained 2D sketch | [Sketches and constraints](/docs/mcad/part/sketches) |
| `feature <name> = <operation>(…)` | a step of the feature history | [Features](/docs/mcad/part/features) |
| `output <name> = { body, part_number, material, … }` | a part the file delivers | [Outputs, regions and published values](/docs/mcad/part/regions-and-publish#outputs) |
| `material = { body, material }` | a body's material | [Outputs](/docs/mcad/part/regions-and-publish#outputs) |
| `region <name> : <kind> = { select, expect, label }` | a named, checked selection | [Regions](/docs/mcad/part/regions-and-publish#regions) |
| `publish <name> = …` | a region, value or frame other files read | [Published values](/docs/mcad/part/regions-and-publish#published-values) |
| `datum`, `tolerance`, `size`, `location`, `finish` | semantic PMI | [PMI](/docs/mcad/part/regions-and-publish#pmi) |
| `configuration <name> = { … }` | a member of the product family | [Configurations](/docs/mcad/part/configurations) |
| `link <kind> <item> to "<target>"` | a traceable link | [Links](/docs/mcad/part/regions-and-publish#links) |

## `.assembly`

An assembly file declares one assembly: `assembly <name> { … }`.

| Statement | Declares | Reference |
|---|---|---|
| `input`, `let` | parameters and values, which occurrences can pass to their designs | [Values and units](/docs/design-language/values-and-units) |
| `use part <design> as <occurrence> { … }`, `use assembly …` | an occurrence: a placed part or sub-assembly | [Occurrences and mates](/docs/mcad/assembly/mates#occurrences) |
| `connector <name> = { occurrence, anchor, offset, flip }` | a frame that mates join | [Connectors](/docs/mcad/assembly/mates#connectors) |
| `mate <name> = { kind, a, b }` | a joint between two connectors | [Mates](/docs/mcad/assembly/mates#mates) |
| `pattern <name> = <layout>(seeds = […], …)` | keyed copies of occurrences | [Patterns](/docs/mcad/assembly/patterns-and-mirrors#pattern-layouts) |
| `mirror <name> = { seeds, plane, hand }` | reflected copies | [Mirrors](/docs/mcad/assembly/patterns-and-mirrors#mirrors) |
| `fasteners <name> = { layout, fastener, stack, exceptions }` | a fastener set along a path | [Fasteners](/docs/mcad/assembly/patterns-and-mirrors#fasteners) |
| `publish <name> = …` | frames and values for the assembly's own users | [Published values](/docs/mcad/part/regions-and-publish#published-values) |
| `part_number = "…"` | the assembly's part number | |
| `options = { axes, rules, groups, configurations }` | a configurable product: option axes, rules between them, effectivity groups and named configurations | [Options](#options) |
| `exploded <name> = { … }` | an exploded state for drawings and instructions | |
| `view <name> = { from, up, default }` | a saved camera | |

### Options

A configurable assembly declares its option axes, the rules between them and its named configurations; occurrences take `when` conditions on the axes. The validator checks that alternatives sharing an occurrence name are exclusive, that every legal configuration has each `required` occurrence, and that each named configuration obeys the rules.

```part file=engine_a.part
runmat 0.7
std 1

part engine_a {
    sketch s on origin.xy {
        c = circle(center = [0 mm, 0 mm], radius = 40 mm)
    }
    feature body = extrude(profile = sketch.s, distance = 120 mm)
    output engine_a = { body = feature.body.body, part_number = "EN-A" }
}
```

```part file=engine_b.part
runmat 0.7
std 1

part engine_b {
    sketch s on origin.xy {
        c = circle(center = [0 mm, 0 mm], radius = 45 mm)
    }
    feature body = extrude(profile = sketch.s, distance = 140 mm)
    output engine_b = { body = feature.body.body, part_number = "EN-B" }
}
```

```assembly fragment
use part engine_a as engine {
    when = { when = is, axis = engine, value = a }
    required = true
}
use part engine_b as engine {
    when = { when = is, axis = engine, value = b }
    required = true
}

options = {
    axes = [
        { name = engine, domain = { domain = enum, values = [a, b] }, default = a },
        { name = freighter, domain = { domain = bool }, default = false }
    ],
    rules = [
        {
            name = freighter_needs_engine_b,
            condition = { when = any, of = [{ when = not, of = { when = is, axis = freighter, value = true } }, { when = is, axis = engine, value = b }] }
        }
    ],
    configurations = [{ name = freighter, values = { engine = b, freighter = true } }]
}
```

## `.drawing`

A drawing file declares one drawing: `drawing <name> { … }`.

| Statement | Declares | Reference |
|---|---|---|
| `format <name> = { size, title_block }` | a sheet format and its title block | [Formats and sheets](/docs/mcad/drawing/views-and-annotations#formats-and-sheets) |
| `sheet "<name>" = { format, scale, projection }` | a sheet | [Formats and sheets](/docs/mcad/drawing/views-and-annotations#formats-and-sheets) |
| `view <name> = <kind>(…)` | a view of a part or an assembly | [Views](/docs/mcad/drawing/views-and-annotations#views) |
| `annotation <name> = <kind>(…)` | a dimension, callout, GD&T frame, symbol, note or balloon | [Annotations](/docs/mcad/drawing/views-and-annotations#annotations) |
| `table <name> = <kind>(…)` | a BOM, hole, revision, bend, weld or cut-list table | [Tables](/docs/mcad/drawing/views-and-annotations#tables) |

## How the files connect

- A part is self-contained. It can read published values from other parts (a skeleton) and `.m` functions in its folder.
- An assembly places parts and assemblies by name, picks their configurations or sets their inputs, and mates them.
- A drawing shows parts and assemblies by name and reads their PMI, properties and configurations.
- A study (`.fea`) reads the exported geometry with the region names the part gives it; see [Import and export](/docs/mcad/import-export).
