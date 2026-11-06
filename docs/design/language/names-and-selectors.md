---
description: How RunMat design files name faces, edges, bodies, occurrences, pins and nets so references survive edits. Generation names, relational selectors, regions, published names and canonical IDs.
keywords: [topological naming, face selection, edge selection, canonical names, regions, published names, occurrence paths, designators, pin references, net names]
---
# Names and selectors

A reference names what it points at by how the design made it, never by a number the kernel happened to assign. `feature.body.face.end` is the face the `body` extrude made at the end of its extent; `feature.body.face.side(outline.l1)` is the side face the sketch line `l1` swept. When the sketch gains a line or an earlier feature changes, these names still point at the same faces, and a reference that can no longer resolve is an error at the reference, with the features that depend on it.

## Kinds and heads

Every reference starts with the kind of item it names:

| Head | Names | Example |
|---|---|---|
| `input` | an input | `input.width` |
| `let` | a let value | `let.half` |
| `sketch` | a sketch, its loops or one of its points | `sketch.outline`, `sketch.outline.l1.end` |
| `feature` | a feature, and the bodies, faces, edges and vertices it made | `feature.body.face.end` |
| `origin` | the part's origin planes, axes and point | `origin.xy`, `origin.z`, `origin.point` |
| `plane`, `axis`, `point` | a datum feature of that kind | `plane.mid`, `axis.hinge` |
| `region` | a named selection | `region.top` |
| `material` | a material | `material.aluminum` |
| `part`, `assembly` | another design | `part.bracket.mass` |
| `net`, `port`, `bus`, `instance` | circuit connections | `net.GND`, `port.IN`, `bus.D.D0` |

The kind keyword keeps a reference's meaning fixed: a `material steel` and a `feature steel` can share a name, and adding a file to the folder never changes what an existing reference means.

## Faces, edges and bodies

`feature.<name>.<topology>.<role>` names an entity by the feature that made it and the role it plays:

| Form | Selects |
|---|---|
| `feature.body.body` | the solid body the feature made |
| `feature.body.face.start`, `feature.body.face.end` | the faces at the start and end of an extrude, revolve or sweep |
| `feature.body.face.side(outline.l1)` | the face the sketch curve `outline.l1` swept |
| `feature.body.face.side(outline.l1, ordinal = 2)` | one piece of a side face that a later feature split |
| `feature.bolt.face.wall(a)` | the wall of the hole at the key `a` |
| `feature.halve.face.split_face(side = positive)` | the face a split made, on one side |
| `feature.base.face.sheet(base, ordinal = 0)` | a sheet-metal face from the sketch line `base` |
| `feature.step.face("Top")` | a face named in an imported CAD file |

The roles a feature gives its faces: `side`, `start`, `end`, `blend` (fillet), `chamfer`, `wall`, `bottom`, `counterbore`, `counterbore_wall`, `countersink` (holes), `inner`, `outer`, `rim` (shell), `thin_wall`, `draft_step`, `split_face`, `boundary`, `patch`, `offset`, `extension`, `thicken`, `thicken_rim`, `mid_surface`, `ruled` (surfaces), `sheet`, `sheet_edge`, `bend`, `relief`, `form` (sheet metal), `member_side`, `member_end`, `cap_plate`, `gusset`, `weld_bead` (weldments), `parting` (mold), `thread`, `emboss_top` and `rib`.

## Relational selectors

Where an entity has no single maker, name it by the entities around it:

| Form | Selects |
|---|---|
| `edges_between(a, b)` | the edges where faces `a` and `b` meet |
| `vertex_of(a, b, c)` | the vertex where three faces meet |
| `piece(a, bounded_by = [b])` | the piece of face `a` bounded by `b` |
| `point(10 mm, 0 mm, 5 mm)` | a point in part coordinates |

This part rounds the edge between its first side face and its top face, drills a hole on its top face, and names the top face as a region:

```part
runmat 0.7
std 1

part block {
    sketch outline on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [60 mm, 0 mm])
        l2 = line(start = [60 mm, 0 mm], end = [60 mm, 40 mm])
        l3 = line(start = [60 mm, 40 mm], end = [0 mm, 40 mm])
        l4 = line(start = [0 mm, 40 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.outline, distance = 20 mm)
    feature bolt = hole(
        face = feature.body.face.end,
        at = { a = point(30 mm, 20 mm, 20 mm) },
        diameter = 6.6 mm,
        through_all
    )
    feature round = fillet(
        edges = [edges_between(feature.body.face.side(outline.l1), feature.body.face.end)],
        radius = 2 mm
    )

    region top : face = { select = [feature.body.face.end], expect = exactly_one }
    region bore : face = { select = [feature.bolt.face.wall(a)], expect = at_least_one, label = "Bolt bore" }

    publish top = region(region = top)

    output block = { body = feature.body.body, part_number = "BK-60" }
}
```

## Regions

A region is a named, checked selection. `region <name> : <kind> = { select = […], expect = … }` selects faces, edges, vertices or bodies, and `expect` states how many it must find:

| `expect` | Meaning |
|---|---|
| `exactly_one` | one entity; zero or several is an error |
| `at_least_one` | one or more |
| `any` | zero is allowed |

`label` gives the region the name its faces carry in a STEP export, so a study selects them with `select name "Bolt bore"`. `body = feature.<name>.body` limits a region to one body's copy of a shared face, as when a fluid body fills a part.

## Published names

`publish` makes an item readable from other files under a stable name. An assembly attaches to a published frame, a drawing dimensions a published face, and another part reads a published value:

| Form | Publishes |
|---|---|
| `publish top = region(region = top)` | a region |
| `publish pitch = value(input = pitch)` | an input's value |
| `publish half = let(name = half)` | a let's value |

Another file reads a published value as `part.<design>.publish.<name>`. See [Outputs, regions and published values](/docs/mcad/part/regions-and-publish).

## Occurrences in an assembly

An assembly names each placed part or sub-assembly by its occurrence name. Paths go down through sub-assemblies with `/`, and copies made by a pattern or a mirror carry the pattern and the copy's key:

| Form | Names |
|---|---|
| `wing_left` | the occurrence `wing_left` |
| `wing/rib` | `rib` inside the sub-assembly occurrence `wing` |
| `rib@ribs[3]` | the copy of `rib` at key 3 of the pattern `ribs` |
| `rivet0@rivets_port[3]` | the mirror `rivets_port`'s image of the copy at key 3 |

## Pins, nets and designators

A circuit names connections by component and pin:

| Form | Names |
|---|---|
| `U1.VIN` | pin `VIN` of `U1` |
| `U.A.OUT` | pin `OUT` of unit `A` of `U` |
| `"C1.1"` | pin `1` of `C1` (quoted, since `1` is not a word) |
| `net.GND`, `"net.+3V3"` | a net |
| `port.IN` | a sheet port |
| `bus.D.D0` | a bus member |
| `instance.pwr.VIN` | a port of a sheet instance |
| `"instance.ch.VIN#2"` | that port on one channel of a repeated instance |

A net inside a repeated sheet takes its instance path: the `FBK` net of the first copy of `ch` is `/ch[0]/FBK` on the board.

## Canonical IDs

The editor, the agent and the 3D view select items by one canonical ID each, such as `feature.boss`, `mate.m1`, `occurrence.wing/rib@ribs[2]`, `net:GND` or `component:U3`. A selection made in one view is the same selection in every other: picking `U3` on the schematic lights its pads on the board and its body in 3D. Diagnostics name their subject with the same IDs.
