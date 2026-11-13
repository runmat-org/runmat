---
description: Reference for exploded views in a RunMat assembly. Ordered translate, rotate and radial steps, steps that move a whole pattern or mirror, spacing, base views, sub-assembly views and explode lines.
keywords: [exploded view, explode steps, assembly instructions, radial explode, explode lines, trail lines, pattern explode, mirror explode, service view]
---
# Exploded views

An exploded view pulls an assembly apart in named steps, for assembly instructions, a service manual or a drawing. The Desktop's Explode tab shows each saved view, and a drawing view shows one with `exploded = <name>`. The assembly itself stays assembled: an exploded view only moves occurrences while it is shown.

This tray holds a row of four pegs and their mirror image on the far side. The view lifts the lid, raises the near row, and slides the far row out, each further peg 5 mm more than the one before.

```part file=tray.part
runmat 0.7
std 1

part tray {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, -40 mm], end = [120 mm, -40 mm])
        l2 = line(start = [120 mm, -40 mm], end = [120 mm, 40 mm])
        l3 = line(start = [120 mm, 40 mm], end = [0 mm, 40 mm])
        l4 = line(start = [0 mm, 40 mm], end = [0 mm, -40 mm])
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
    output tray = { body = feature.body.body }
}
```

```part file=peg.part
runmat 0.7
std 1

part peg {
    sketch s on origin.xy {
        l1 = line(start = [-4 mm, -4 mm], end = [4 mm, -4 mm])
        l2 = line(start = [4 mm, -4 mm], end = [4 mm, 4 mm])
        l3 = line(start = [4 mm, 4 mm], end = [-4 mm, 4 mm])
        l4 = line(start = [-4 mm, 4 mm], end = [-4 mm, -4 mm])
    }
    feature body = extrude(profile = sketch.s, distance = 20 mm)
    output peg = { body = feature.body.body }
}
```

```assembly file=pegboard.assembly
runmat 0.7
std 1

assembly pegboard {
    use part tray as tray
    use part tray as lid {
        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 30 mm] } }
    }
    use part peg as peg {
        placement = { placement = offset, offset = { translation = [15 mm, 20 mm, 10 mm] } }
    }

    pattern pegs = linear(seeds = [peg], direction = [1, 0, 0], spacing = 30 mm, count = 4)
    mirror pegs_far = {
        seeds = [pegs],
        plane = { plane = explicit, point = [0 mm, 0 mm, 0 mm], normal = [0, 1, 0] },
        hand = symmetric(point = [0 mm, 0 mm, 0 mm], normal = [0, 1, 0])
    }

    exploded service = {
        steps = [
            { id = lid_off, occurrences = [[{ name = lid }]], motion = { motion = "translate", direction = { direction = "vector", vector = [0, 0, 1] }, distance = 60 mm } },
            { id = near_pegs_up, patterns = [[{ name = pegs }]], motion = { motion = "translate", direction = { direction = "vector", vector = [0, 0, 1] }, distance = 30 mm } },
            { id = far_pegs_out, mirrors = [[{ name = pegs_far }]], motion = { motion = "translate", direction = { direction = "vector", vector = [0, -1, 0] }, distance = 40 mm, spacing = 5 mm } }
        ],
        lines = [{ id = lid_trail, occurrence = [{ name = lid }] }]
    }
}
```

## Steps

`exploded <name> = { steps = […], lines = […], base = <view>, include = […] }`. Steps run in order, and each starts from the positions the steps before it left. Moving an occurrence moves everything inside it.

Each step has an `id`, what it moves, and a `motion`. A step names what it moves in up to three lists, and needs at least one entry:

| Field | Moves |
|---|---|
| `occurrences` | Each occurrence by its path: `[{ name = fuselage }, { name = forward }, { name = panel }]`. A pattern's seed brings every copy the pattern lays out of it, so `[{ name = stack_screw }]` moves all four screws of a grid. A copy named with its instance moves alone: `{ name = rib, instance = { by = ribs, key = { instance = linear, step = 3 } } }`. |
| `patterns` | Each pattern whole: the path of the assembly that declares it, then the pattern's name. The step moves the pattern's seeds and every copy it lays out in the shown configuration. `[[{ name = wing }, { name = front_upper_cap }]]` moves one rivet row of the wing. |
| `mirrors` | Each mirror whole, named the same way. The step moves every image the mirror makes, and leaves the originals where they are. A mirror of a pattern makes one image per seed and per copy, so `[[{ name = wing }, { name = rivets_port }]]` moves every port-side rivet. |

A pattern or mirror keeps up with its layout: when the count changes, the step moves the new copies too. A step that names a pattern it can't find, or names a single copy in `patterns` or `mirrors`, is an error. A pattern that lays out nothing in the shown configuration is skipped with a warning.

## Motions

| Motion | Fields | Effect |
|---|---|---|
| `translate` | `direction`, `distance`, `spacing` (optional) | Moves along `direction = { direction = "vector", vector = [x, y, z] }` (assembly coordinates), or along an axis of an occurrence's frame: `{ direction = "occurrence_axis", occurrence = […], axis = z, reversed = false }`, so a bolt backs out along its own axis. |
| `rotate` | `axis`, `angle` | Turns about `{ about = "explicit", point = […], direction = […] }`, or about an occurrence's axis: `{ about = "occurrence_axis", occurrence = […], axis = z }`. |
| `radial` | `center`, `distance` | Moves each occurrence away from an axis (`{ center = "axis", point = […], direction = […] }`) or a point (`{ center = "point", point = […] }`), along the line from the centre to the occurrence's centre. Skins come off a barrel this way. |

With `spacing`, the step moves its i-th occurrence `distance + i × spacing`, counting in the order the step lists them: first `occurrences` (each seed followed by its copies), then each pattern's seeds and copies in layout order, then each mirror's images. Stacked washers fan out this way.

## Lines, base views and sub-assembly views

- `lines = [{ id, occurrence, anchor, jogs }]` draws a trail from an occurrence's assembled position to its exploded one, through its position after every step that moved it. `anchor` is the trailed point in the occurrence's frame (its centre by default), and each jog (`{ segment, at, offset }`) routes the trail around something in its way.
- `base = <view>` runs another view's steps first, so a service view can extend the assembly-instructions view.
- `include = [{ id, occurrence, view, after }]` runs a sub-assembly's own exploded view inside this one, after the step `after` (before the first step when it is left out).

## What the check reports

The check reports a view or step declared twice, a base view that is missing or that builds on itself, a zero direction or axis, a step that moves nothing or names something twice, and a path that names nothing in the assembly. A step that names an occurrence and its own sub-assembly moves the occurrence twice, and the check notes it. A renamed occurrence still resolves through the lock, with a fix that rewrites the path.

See [Patterns, mirrors and fasteners](/docs/mcad/assembly/patterns-and-mirrors) for how copies are keyed, and [Drawing views and annotations](/docs/mcad/drawing/views-and-annotations) for showing an exploded view on a sheet.
