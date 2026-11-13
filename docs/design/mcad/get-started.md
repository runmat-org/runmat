---
description: Start designing mechanical parts in RunMat. Create a part file, set its inputs, check it, place it in an assembly, and export STEP, in the Desktop, with the agent or from the command line.
keywords: [get started, RunMat MCAD, first part, CAD tutorial, parametric part, text CAD, export STEP]
---
# Get started with MCAD

RunMat MCAD designs live in a project folder as text files: a `.part` per part, a `.assembly` per assembly and a `.drawing` per drawing. The Desktop edits them in a 3D canvas and a source view side by side; the agent and the command line read and write the same files.

## 1. A part

Create `plate.part` in the project folder. The Desktop's New Part command writes the header and an empty `part` block; the sketch, the extrude and the output can then be drawn on the canvas or typed.

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

- `sketch s on origin.xy` draws a closed rectangle on the XY plane.
- `feature body = extrude(…)` makes it 6 mm thick.
- `output plate` declares the part: its body and its part number.

The canvas regenerates on every change, and problems show in the source view at the line that causes them. See [Checks and diagnostics](/docs/design-language/checks-and-diagnostics).

## 2. Inputs

Inputs make a part a parametric family. Replace the literal sizes with inputs, and every view, configuration, run and sweep can set them:

```part file=tab.part
runmat 0.7
std 1

part tab {
    input width : length = 40 mm
    input height : length = 20 mm
    input thickness : length = 3 mm

    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])
        l2 = line(start = [input.width, 0 mm], end = [input.width, input.height])
        l3 = line(start = [input.width, input.height], end = [0 mm, input.height])
        l4 = line(start = [0 mm, input.height], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s, distance = input.thickness)
    output tab = { body = feature.body.body, part_number = "TB-40" }
}
```

The Properties panel shows each input with its unit; editing one writes the new default into the file.

## 3. An assembly

Parts in the same folder name each other with no import. A package folder `+hardware/` holds shared parts, found from any folder below it:

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
    use part tab as tab {
        configuration = { select = inputs, values = { width = 30 mm } }
        placement = { placement = offset, offset = { translation = [40 mm, 20 mm, 6 mm] } }
    }
}
```

The tab occurrence sets its own width, so the same part file serves two sizes. Mates join parts that move; see [Occurrences and mates](/docs/mcad/assembly/mates).

## 4. Check and export

```sh
runmat-design export base.assembly --format step --out exports
runmat-design export plate.part --format dxf --out exports
```

The agent runs the same tools: ask it to check the assembly, change an input, or export STEP for a study, and it proposes each change as a diff you keep or undo. See [Agent tools and the command line](/docs/design-language/agent-and-cli).

## Next

- Guides: [Model a part](/docs/mcad/guides/model-a-part), [Build an assembly](/docs/mcad/guides/build-an-assembly), [Make a drawing](/docs/mcad/guides/make-a-drawing), [Sheet metal](/docs/mcad/guides/sheet-metal) and [Product families and configurations](/docs/mcad/guides/configurations).
- [MCAD file reference](/docs/mcad/files): every statement in `.part`, `.assembly` and `.drawing` files.
- [Features](/docs/mcad/part/features): every modeling operation.
- [Drawing views and annotations](/docs/mcad/drawing/views-and-annotations): drawings that follow the model.
