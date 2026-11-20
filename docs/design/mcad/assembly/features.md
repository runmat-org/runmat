---
description: Reference for assembly features in RunMat. Cut through several components, drill matching holes across parts, or add a weld bead, from a sketch in the assembly, without changing the part files.
keywords: [assembly feature, assembly cut, hole across parts, match drilling, weld bead, assembly-level machining]
---
# Assembly features

An assembly feature removes or adds material at the assembly level. Use one for work done after the parts are put together: a pocket machined through two plates bolted together, holes drilled through a stack in one pass, or a weld bead along a joint. The part files don't change. Each component the feature affects draws its part with the feature applied, where it sits in the assembly, and a drawing or another assembly that uses the part file alone sees it without the feature.

Every assembly feature reads a sketch in the assembly ([Sketches in an assembly](/docs/mcad/assembly/sketches)):

| Feature | Statement | What the sketch gives |
|---|---|---|
| Cut | `operation <name> = cut(profile = sketch.<s>, extent = …)` | the closed profile, extruded from the sketch's plane and removed |
| Hole across parts | `operation <name> = hole(profile = sketch.<s>, diameter = …, extent = …)` | one hole at each point, and at each circle's or arc's center |
| Weld bead | `operation <name> = weld(path = sketch.<s>, size = …)` | the path: the sketch's lines and arcs, joined end to end |

`components = [<occurrence>, …]` limits a feature to those components. Without it, a feature affects every part component it meets. A weld bead is drawn with the first component it names, or with the first component of the assembly.

The features apply after placement, mates and sketches, in the order the file lists them. A sketch on a component's face moves with the component, so a feature from that sketch moves with it too.

## Example

```part file=plate.part
runmat 0.7
std 1

part plate {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
    output plate = { body = feature.body.body }
}
```

```assembly file=stack.assembly
runmat 0.7
std 1

assembly stack {
    use part plate as lower
    use part plate as upper {
        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 20 mm] } }
    }
    sketch pocket on origin.xy {
        l1 = line(start = [30 mm, 15 mm], end = [50 mm, 15 mm])
        l2 = line(start = l1.end, end = [50 mm, 35 mm])
        l3 = line(start = l2.end, end = [30 mm, 35 mm])
        l4 = line(start = l3.end, end = l1.start)
        p1 = point(at = [65 mm, 25 mm])
    }
    sketch seam on origin.xy {
        w1 = line(start = [0 mm, 50 mm], end = [80 mm, 50 mm])
    }
    operation slot = cut(profile = sketch.pocket, extent = through_all, both)
    operation bolt = hole(profile = sketch.pocket, diameter = 6.6 mm, extent = through_all, size = "M6", counterbore = { diameter = 11 mm, depth = 6.5 mm }, components = [lower, upper])
    operation back = weld(path = sketch.seam, size = 5 mm, components = [lower])
}
```

`slot` cuts the 20 mm square through both plates. `bolt` drills an M6 counterbored hole at `p1`, up through both plates. `back` lays a 5 mm fillet weld along the lower plate's back edge.

## Cut

| Argument | Meaning |
|---|---|
| `profile = sketch.<s>` | the sketch whose closed profile is cut. Its loops are read as a part extrude reads them: an inner loop leaves an island. |
| `extent` | `through_all` (along the sketch's normal), `blind(distance = …)` (that far from the plane) or `symmetric(distance = …)` (half each way) |
| `both` | with `through_all`: through everything on both sides of the plane |
| `reverse` | against the sketch's normal |

## Hole across parts

Each hole is drilled from the sketch's plane along its normal; `reverse` drills against it. The holes are the same size in every component they pass through, so the parts line up for the fastener.

| Argument | Meaning |
|---|---|
| `profile = sketch.<s>` | the sketch whose points and circle centers place the holes |
| `diameter` | the drilled diameter |
| `extent` | `through_all` or `blind(distance = …)` |
| `size = "M6"` | the fastener the hole is for. In the panel, a size fills the diameter with the fastener's normal, close or loose clearance, and fills a counterbore or countersink from the standard's row. ISO metric sizes (`M6`) and unified inch sizes (`1/4-20`, `#10-32`) are read. |
| `counterbore = { diameter, depth }` | a counterbore at the start of each hole |
| `countersink = { diameter, angle }` | a countersink at the start of each hole (included angle) |

## Weld bead

A weld bead is a fillet weld's triangle, with both legs `size` long, swept along the path. The triangle stands on the sketch's plane, on the side of its normal, and lies on the left of the path, seen from the normal side and walking from the path's first point. `flip` puts it on the right.

The path is the sketch's lines and arcs, construction geometry left out. They must join end to end in one chain.

## In the editor

**Layout → Assembly feature** opens the panel. Choose **Cut**, **Hole across parts** or **Weld bead**, pick the sketch (its row in the Outline, or one of its curves), set the end condition or sizes, and optionally pick the components it affects. The Outline lists the features under **Assembly features**, each with its sketch and components. A feature whose sketch is missing or whose profile isn't closed shows why. Right-click a feature to change it (**Assembly feature**), suppress it or delete it.
