---
description: Reference for repeating and mirroring occurrences in a RunMat assembly, and for fastener sets laid out along paths. Linear, grid, circular, table and follow-pattern layouts, mirrors of patterns, keyed copies, and fastener rules.
keywords: [assembly pattern, circular pattern, linear pattern, mirror, keyed instances, fasteners, rivets, bolt patterns, follow pattern, skip instances]
---
# Patterns, mirrors and fasteners

A pattern repeats occurrences by a layout, and a mirror reflects them. Each copy is keyed, so a mate, a drawing balloon or a diagnostic can name one copy: `rib@ribs[3]` is the copy of `rib` at key 3 of the pattern `ribs`. Copies are counted with their seed: a linear pattern of `count = 6` places the seed and five more.

This frame repeats a rib every 50 mm, spokes around a hub, and mirrors the row of ribs to the far side.

```part file=rib.part
runmat 0.7
std 1

part rib {
    sketch s on origin.yz {
        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])
        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])
        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])
        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s, distance = 2 mm)
    output rib = { body = feature.body.body, part_number = "RB-40" }
}
```

```part file=spoke.part
runmat 0.7
std 1

part spoke {
    sketch s on origin.xy {
        l1 = line(start = [10 mm, -2 mm], end = [50 mm, -2 mm])
        l2 = line(start = [50 mm, -2 mm], end = [50 mm, 2 mm])
        l3 = line(start = [50 mm, 2 mm], end = [10 mm, 2 mm])
        l4 = line(start = [10 mm, 2 mm], end = [10 mm, -2 mm])
    }
    feature body = extrude(profile = sketch.s, distance = 3 mm)
    output spoke = { body = feature.body.body }
}
```

```assembly file=frame.assembly
runmat 0.7
std 1

assembly frame {
    input pitch : length = 50 mm
    input rib_count : count = 6

    use part rib as rib
    use part spoke as spoke {
        placement = { placement = offset, offset = { translation = [0 mm, 100 mm, 0 mm] } }
    }

    pattern ribs = linear(seeds = [rib], direction = [1, 0, 0], spacing = input.pitch, count = input.rib_count)
    pattern spokes = circular(seeds = [spoke], axis_point = [0 mm, 100 mm, 0 mm], axis = [0, 0, 1], count = 8, angle = 360 deg)
    mirror ribs_far = {
        seeds = [ribs],
        plane = { plane = explicit, point = [0 mm, 50 mm, 0 mm], normal = [0, 1, 0] },
        hand = symmetric(point = [0 mm, 50 mm, 0 mm], normal = [0, 1, 0])
    }

    part_number = "FR-300"
}
```

## Pattern layouts

`pattern <name> = <layout>(seeds = […], …)`:

| Layout | Parameters | Copies |
|---|---|---|
| `linear` | `direction` (a unit vector), `spacing`, `count` | `count` in all, `spacing` apart |
| `grid` | `first`, `first_spacing`, `first_count`, `second`, `second_spacing`, `second_count` | a two-way array |
| `circular` | `axis_point`, `axis`, `count`, `angle` | `count` equally spaced over `angle` |
| `table` | `rows = [{ key, offset }]` | one copy per row, keyed by the row |
| `follow_pattern` | `occurrence`, `publish` | one copy per key of a hole pattern a part publishes, the seed on the first key |

- `skip` leaves keyed copies out, and `when` makes the pattern conditional on the assembly's options.
- A follow pattern keeps fasteners on their holes: when the part's hole pattern gains a hole, the assembly gains a fastener at the new key.

## Mirrors

`mirror <name> = { seeds = […], plane = …, hand = … }` reflects occurrences about a plane:

- `plane = { plane = explicit, point = […], normal = […] }` gives the plane.
- `hand = symmetric(point, normal)` places the seed's own design reflected, for a part that is its own mirror image about that plane (in the part's own coordinates).
- `hand = opposite(source, configuration, point, normal)` places the opposite-hand design: `source = { definition = { kind = part, name = <design> } }` names it, `configuration` selects the member that is the seed's mirror image about the plane `point`, `normal` (in the part's own coordinates), and the BOM lists both part numbers. A part with a left- and a right-hand version usually carries the other hand as a configuration with its own part number:

```assembly fragment
mirror ribs_port = {
    seeds = [rib],
    plane = { plane = explicit, point = [0 mm, 0 mm, 0 mm], normal = [0, 1, 0] },
    hand = opposite(source = { definition = { kind = part, name = rib } }, configuration = { select = named, name = port }, point = [0 mm, 0 mm, 0 mm], normal = [0, 1, 0])
}
```

See [Product families and configurations](/docs/mcad/guides/configurations) for a handed part built this way.
- A seed may be a pattern of the same assembly. The mirror then reflects the pattern's seeds and every copy, keeping their keys and skips: `rib@ribs_far[3]` is the image of `rib@ribs[3]`. A copy reached twice, such as a seed listed directly and through its pattern, is an error that names both routes.

## Fasteners

`fasteners <name> = { … }` lays out a fastener set along a path through a stack of occurrences, such as a riveted lap joint. The fastener's family is a part, found by name like any other design:

```part file=hi_lok.part
runmat 0.7
std 1

part hi_lok {
    input diameter : length = 4.8 mm
    input grip : length = 6 mm
    sketch s on origin.xy {
        c = circle(center = [0 mm, 0 mm], radius = input.diameter / 2)
    }
    feature shank = extrude(profile = sketch.s, distance = input.grip)
    output hi_lok = { body = feature.shank.body, part_number = "HL-4.8" }
}
```

```assembly fragment
fasteners lap0 = {
    layout = along(
        occurrence = [{ name = skin }],
        path = laps,
        chains = [lap0],
        spacing = { spacing = pitch, pitch = 21.6 mm },
        edge_distance = 2 * 4.8 mm,
        rows = [{ key = r0, offset = 0 mm }, { key = r1, offset = 25 mm }]
    ),
    fastener = {
        family = { kind = part, name = hi_lok },
        diameter = 4.8 mm,
        grip = { grip = auto }
    },
    stack = [[{ name = skin }], [{ name = stringer }]],
    exceptions = [
        {
            at = { at = station, station = 74.4 mm, chain = lap0, row = r0 },
            action = { action = oversize, delta = 0.4 mm }
        }
    ]
}
```

| Field | Meaning |
|---|---|
| `layout` | Where the fasteners go: `along(…)` a published path, with a spacing, an edge distance and rows. |
| `fastener` | The family: a fastener part by name, with `from = "<folder>"` when it lives in another folder such as a shared library. The diameter, and the grip (`auto` reads it from the stack). |
| `stack` | The occurrences each fastener passes through, top to bottom. |
| `exceptions` | Changes at given stations: an oversize fastener, a skipped one. |

The check reports a path that is missing or too short, an exception that matches no fastener or several, two fasteners with one key, and a stack no fastener in the family fits.
