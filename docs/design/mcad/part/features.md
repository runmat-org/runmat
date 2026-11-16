---
description: Reference for every feature a RunMat part can hold, with its parameters and examples. Extrude, revolve, sweep, loft, holes and threads, fillets and chamfers, shell, draft, patterns, mirrors, booleans, surfaces, sheet metal, weldments, mold tools, direct edits, imports and datums.
keywords: [CAD features, extrude, revolve, sweep, loft, hole, thread, fillet, chamfer, shell, draft, pattern, mirror, boolean, surface modeling, sheet metal, weldment, mold design, direct editing, datum planes]
---
# Features

A part's geometry is its feature history: `feature <name> = <operation>(…)` statements, regenerated in the order they are written. Each feature names its inputs (sketches, faces, bodies) by reference, and later features name what it made: `feature.body.face.end`, `feature.bolt.face.wall(a)`. Every feature takes `suppressed` to leave it out of the history.

## Solids from sketches

<!-- arguments:operations extrude revolve sweep loft rib emboss -->
| Operation | Does | Arguments |
|---|---|---|
| `extrude` | A prism from a closed sketch profile. | `profile`, `distance`, `through_all`, `up_to_next`, `up_to`, `symmetric`, `extent`, `reverse`, `second`, `from`, `draft`, `thin`, `mode` |
| `revolve` | A solid of revolution: a profile turned about an axis. | `profile`, `axis`, `angle`, `mode` |
| `sweep` | A profile moved along a path. | `profile`, `path`, `follow_path`, `mode` |
| `loft` | A solid through profiles in order. | `profiles`, `guides`, `mode` |
| `rib` | A rib from an open sketch. | `profile`, `thickness`, `depth`, `reverse` |
| `emboss` | Raised or recessed text or shapes on a face. | `profile`, `face`, `height`, `wrap` |
<!-- /arguments -->

`mode` is `new` (the default for the first solid), `add`, `cut` or `intersect`. On the part's first body feature, `add` has no body to join, so it makes the first body as `new` does; `cut` and `intersect` there fail with nothing to act on. A part with a plate and a cut pocket:

```part
runmat 0.7
std 1

part pocket_plate {
    sketch outline on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])
        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])
        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.outline, distance = 12 mm)
    sketch pocket_s on feature.body.face.end {
        p1 = line(start = [20 mm, 15 mm], end = [60 mm, 15 mm])
        p2 = line(start = [60 mm, 15 mm], end = [60 mm, 35 mm])
        p3 = line(start = [60 mm, 35 mm], end = [20 mm, 35 mm])
        p4 = line(start = [20 mm, 35 mm], end = [20 mm, 15 mm])
    }
    feature pocket = extrude(profile = sketch.pocket_s, distance = 5 mm, reverse, mode = cut)
    output pocket_plate = { body = feature.body.body }
}
```

The revolved forms:

```part fragment
feature spin = revolve(profile = sketch.s_rev, axis = origin.z, angle = 360 deg)
feature groove = revolve(profile = sketch.s_gr, axis = axis.gaxis, angle = 360 deg, mode = cut)
feature sw = sweep(profile = sketch.s_swp, path = sketch.s_path)
feature lf = loft(profiles = [sketch.s_l0, sketch.s_l1, sketch.s_l2])
feature box = extrude(profile = sketch.s, distance = 10 mm, thin = { thickness = 2 mm })
feature post = extrude(profile = sketch.s2, extent = offset_from_face(face = feature.body.face.end, offset = 2 mm))
feature logo = emboss(profile = sketch.s_logo, face = feature.body.face.end, height = 1 mm)
```

In the Desktop, Revolve takes the sketch as its profile, and the axis can be a straight edge picked on the model. The preview turns the profile through the angle, here a full turn that makes two hinge knuckles on a leaf's edge:

<!-- design:clip:mcad/revolve -->

A sweep keeps the profile parallel to itself by default: the profile is translated along the path, so every path segment must leave the profile's plane. With `follow_path`, the profile keeps its angle to the path instead. It turns through each arc of the path and is mitred where two straight segments meet, so the path can bend into the profile's own plane. Pipe, duct and cable runs use it:

```part fragment
feature run = sweep(profile = sketch.pipe_s, path = sketch.run_s, follow_path)
```

A `follow_path` sweep needs a closed profile and the exact geometry engine; the grid test kernel builds it only along a straight path.

## Holes and threads

`hole` drills one hole at each key of `at`, on a planar face. Each hole's faces are keyed: `feature.mount.face.wall(a)`.

<!-- arguments:table hole -->
| Argument | Kind | Default | Meaning |
|---|---|---|---|
| `face` | reference | required | The planar face the holes start on. |
| `at` | record | required | Keyed positions: `{ a = point(15 mm, 25 mm, 10 mm), … }`. |
| `diameter` | length (mm) |  | The drilled diameter (default: from `spec`). |
| `depth` | length (mm) |  | A blind hole's full-diameter depth. |
| `through_all` | flag |  | Drill through every body in the way. |
| `up_to_next` | flag |  | Drill to the next face the hole meets. |
| `end` | `blind` \| `through_all` \| `up_to_next` \| `up_to_face` |  | How deep the hole goes, in full form. |
| `form` | `simple` \| `counterbore` \| `countersink` \| `tapped` \| `pipe_tap` \| `slot` | `simple` | The hole's form. |
| `spec` | record |  | A standard size and fit: `{ standard = iso, size = M6 }`; the size sets the diameters. |
| `form.counterbore.diameter` | length (mm) |  | A slot's counterbore diameter. |
| `form.counterbore.depth` | length (mm) |  | A slot's counterbore depth. |
| `drill_point` | angle (deg) |  | A blind hole's drill-point angle (default: a flat bottom). |
| `series` | record |  | A hole through several bodies: `{ bodies, end, middle_diameter }`. |
| `series.middle_diameter` | length (mm) |  | The middle bodies' diameter (default: the start clearance). |
<!-- /arguments -->

```part fragment
feature cb = hole(face = feature.body.face.end, at = { c = point(40 mm, 25 mm, 10 mm) }, through_all, form = counterbore, spec = { standard = iso, size = M6 })
feature cs = hole(face = feature.body.face.end, at = { a = point(60 mm, 25 mm, 10 mm) }, diameter = 6 mm, depth = 8 mm, form = countersink(diameter = 12 mm, angle = 90 deg))
feature slot = hole(face = feature.body.face.end, at = { s = point(40 mm, 25 mm, 10 mm) }, diameter = 6 mm, through_all, form = slot(length = 20 mm))
feature m6_thread = thread(face = feature.tap.face.wall(m6), spec = { designation = M6x1, pitch = 1 mm, depth = 0.54 mm, class = "6H" })
```

## Edges and faces

<!-- arguments:operations fillet chamfer face_fillet full_round shell draft -->
| Operation | Does | Arguments |
|---|---|---|
| `fillet` | Rounds edges. | `edges`, `radius`, `variable`, `profile`, `setbacks`, `propagate` |
| `chamfer` | Bevels edges or vertices. | `edges`, `distance`, `distance2`, `angle`, `distances`, `form`, `flip` |
| `face_fillet` | A blend tangent to two face sets. | `first`, `second`, `radius`, `profile` |
| `full_round` | Replaces center faces by a blend tangent to two side face sets. | `first`, `center`, `second` |
| `shell` | Hollows a body. | `body`, `thickness`, `open_faces`, `faces`, `outward` |
| `draft` | Tilts faces for molding. | `faces`, `neutral`, `angle`, `kind`, `flip` |
<!-- /arguments -->

In the Desktop, Fillet takes the edges you pick on the model and previews the rounds. Type the radius in the box on the model or in the form, then confirm:

<!-- design:clip:mcad/fillet -->

Chamfer works the same way: pick the edges, then set the distance on the model. A fillet or chamfer carries on along the edges tangent to the picked one, so the bevel below follows the lug around its rounded end:

<!-- design:clip:mcad/chamfer -->

This tray drafts its walls 2° and shells itself open at the top, then adds a post and mirrors it about a datum plane:

```part
runmat 0.7
std 1

part tray {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])
        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 60 mm])
        l3 = line(start = [100 mm, 60 mm], end = [0 mm, 60 mm])
        l4 = line(start = [0 mm, 60 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s, distance = 25 mm)
    feature walls = draft(
        faces = [feature.body.face.side(s.l1), feature.body.face.side(s.l2), feature.body.face.side(s.l3), feature.body.face.side(s.l4)],
        neutral = origin.xy,
        angle = 2 deg
    )
    feature hollow = shell(body = feature.body.body, open_faces = [feature.body.face.end], thickness = 2 mm)
    feature mid = plane.offset(base = origin.yz, distance = 50 mm)
    sketch boss on origin.xy {
        c = circle(center = [20 mm, 30 mm], radius = 4 mm)
    }
    feature post = extrude(profile = sketch.boss, distance = 20 mm, mode = add)
    feature twin = mirror(features = [post], plane = plane.mid)
    output tray = { body = feature.body.body }
}
```

## Patterns and mirrors

`pattern` repeats features (`features = […]`) or bodies (`bodies = […]`); `mirror` reflects features about a plane. Each copy is keyed, so a later feature can name one copy's faces.

In the Desktop, select the feature to repeat, start Pattern and pick an edge for the direction. The copies preview along it; set the spacing and confirm:

<!-- design:clip:mcad/pattern -->

<!-- arguments:choices pattern layout -->
| `layout` | Arguments | Gives |
|---|---|---|
| `linear` | `direction`, `spacing`, `count` | `count` copies in all, `spacing` apart along `direction`. |
| `grid` | `first`, `first_spacing`, `first_count`, `second`, `second_spacing`, `second_count` | A two-way array. |
| `circular` | `axis`, `count`, `angle`, `equal_spacing` | `count` copies about `axis`, equally spaced over `angle`. |
| `table` | `rows` | One copy per row of offsets. |
| `curve` | `curve`, `count`, `spacing`, `orientation` | Copies along an edge or curve from its start. |
| `sketch` | `sketch`, `points`, `reference` | One copy at each point of a sketch. |
| `fill` | `boundary`, `spacing`, `margin`, `direction`, `seed`, `lattice` | Copies filling a sketch region on a lattice. |
| `variable` | `instances` | Copies each with their own offset and input values. |
<!-- /arguments -->

`skip = [{ instance = linear, step = 2 }]` leaves copies out.

```part fragment
feature row = pattern(features = [h], layout = linear(direction = origin.x, spacing = 15 mm, count = 4), skip = [{ instance = linear, step = 2 }])
feature ring = pattern(features = [pin], layout = circular(axis = axis.cax, count = 6, angle = 360 deg))
feature pat = pattern(features = [pocket], layout = sketch(sketch = pts))
feature copies = pattern(bodies = [feature.body.body], layout = table(rows = [{ key = aft, dx = 0 mm, dy = 100 mm, dz = 0 mm }]))
feature mir = mirror(features = [pk2], plane = plane.mp)
```

## Bodies

<!-- arguments:operations boolean.union boolean.subtract boolean.intersect split body_op.delete body_op.translate body_op.rotate body_op.copy body_op.indent import library -->
| Operation | Does | Arguments |
|---|---|---|
| `boolean.union` | Joins tool bodies to a target body. | `target`, `tools`, `keep_tools` |
| `boolean.subtract` | Cuts tool bodies from a target body. | `target`, `tools`, `keep_tools` |
| `boolean.intersect` | Keeps what a target body shares with the tool bodies. | `target`, `tools`, `keep_tools` |
| `split` | Splits a body with a plane or surface into `body(side = positive)` and `body(side = negative)`. | `body`, `tool` |
| `body_op.delete` | Removes bodies. | `bodies` |
| `body_op.translate` | Moves bodies by an offset. | `bodies`, `dx`, `dy`, `dz` |
| `body_op.rotate` | Turns bodies about an axis. | `bodies`, `axis`, `angle` |
| `body_op.copy` | Copies bodies, the originals kept: copy `k` moves `k` times by the offset, then turns `k` times by `angle` about `axis`. | `bodies`, `dx`, `dy`, `dz`, `axis`, `angle`, `count` |
| `body_op.indent` | Presses a tool body into a target; the tool, grown by `clearance`, is cut from the target and kept. | `target`, `tool`, `clearance` |
| `import` | A body from a STEP file, held by the file's digest; its named faces read as `feature.<name>.face("Top")`. | `source`, `digest` |
| `library` | An instance of a reusable feature definition (`feature_def`). | `def`, `args`, `bindings`, `mode` |
<!-- /arguments -->

```part fragment
feature join = boolean.union(target = feature.body.body, tools = [feature.ear.body])
feature halve = split(body = feature.body.body, tool = plane.cutp)
feature drop = body_op.delete(bodies = [feature.halve.body.body(side = negative)])
feature step = import(source = "supplier/housing.step", digest = d2)
feature boss1 = library(def = "feature_def.boss", args = { h = 8 mm }, mode = add)
```

## Datums and curves

<!-- arguments:operations plane.offset plane.angle plane.three_point plane.mid plane.normal_to_curve plane.tangent_to_face axis.through axis.of_face axis.of_edge axis.two_planes axis.normal point.at point.coords point.offset point.center_of point.on_curve point.intersection curve.helix curve.spiral curve.projected curve.composite curve.split_line curve.intersection curve.through_points curve.equation mate_connector -->
| Operation | Does | Arguments |
|---|---|---|
| `plane.offset` | A datum plane parallel to a base plane, `distance` along its normal; read as `plane.<name>`. | `base`, `distance` |
| `plane.angle` | A datum plane turned about an axis lying in a base plane. | `base`, `axis`, `angle` |
| `plane.three_point` | A datum plane through three points. | `a`, `b`, `c` |
| `plane.mid` | A datum plane midway between two planes. | `a`, `b` |
| `plane.normal_to_curve` | A datum plane normal to a curve at a point on it. | `curve`, `parameter` |
| `plane.tangent_to_face` | A datum plane tangent to a face at a point. | `face`, `at` |
| `axis.through` | A datum axis through two points; read as `axis.<name>`. | `a`, `b` |
| `axis.of_face` | The axis of a cylindrical or conical face. | `face` |
| `axis.of_edge` | The axis along a straight edge, or a circular edge's axis. | `edge` |
| `axis.two_planes` | The axis where two planes meet. | `a`, `b` |
| `axis.normal` | The axis normal to a plane through a point. | `plane`, `through` |
| `point.at` | A datum point at another point. | `at` |
| `point.coords` | A datum point at coordinates. | `x`, `y`, `z` |
| `point.offset` | A datum point offset from another. | `base`, `dx`, `dy`, `dz` |
| `point.center_of` | A datum point at the center of a circular edge or face. | `of` |
| `point.on_curve` | A datum point at a fraction along a curve. | `curve`, `parameter` |
| `point.intersection` | A datum point where an axis meets a plane. | `axis`, `plane` |
| `curve.helix` | A helix about an axis. | `axis`, `start`, `pitch`, `turns` |
| `curve.spiral` | A flat spiral. |  |
| `curve.projected` | A sketch curve projected onto faces. |  |
| `curve.composite` | One curve joined from edges. |  |
| `curve.split_line` | A curve that splits faces. |  |
| `curve.intersection` | The curve where a face meets a plane. |  |
| `curve.through_points` | A curve through points in order. |  |
| `curve.equation` | A curve from parametric equations. |  |
| `mate_connector` | A frame an assembly mates to. | `placement`, `offset`, `rotation`, `flip` |
<!-- /arguments -->

```part fragment
feature mid = plane.offset(base = origin.yz, distance = 40 mm)
feature mp = plane.angle(base = origin.yz, axis = origin.z, angle = -45 deg)
feature tilted = plane.three_point(a = point(0 mm, 0 mm, 0 mm), b = point(10 mm, 0 mm, 0 mm), c = point(0 mm, 10 mm, 10 mm))
feature cax = axis.through(a = point(40 mm, 25 mm, 0 mm), b = point(40 mm, 25 mm, 10 mm))
feature coil = curve.helix(axis = origin.z, start = point(10 mm, 0 mm, 0 mm), pitch = 2 mm, turns = 3)
feature mc = mate_connector(placement = { on = frame, frame = { frame = on_face, face = feature.body.face.end } }, offset = [0 mm, 0 mm, 2 mm])
```

## Surfaces

`surface.extrude`, `surface.revolve`, `surface.sweep`, `surface.loft`, `surface.boundary`, `surface.fill`, `surface.planar`, `surface.ruled`, `surface.offset`, `surface.extend`, `surface.trim`, `surface.untrim`, `surface.knit`, `surface.thicken`, `surface.delete_hole`, `surface.cut_with_surface` and `surface.mid_surface` build and edit sheet bodies; `surface.thicken` and `surface.knit` turn them into solids.

<!-- design:clip:mcad/surfaces -->

```part fragment
feature patch = surface.planar(profile = sketch.s)
feature grow = surface.extend(edges = [feature.patch.edge.boundary(s.b)], distance = 10 mm)
feature solid = surface.thicken(bodies = [feature.patch.body], thickness = 3 mm)
feature trim = surface.cut_with_surface(body = feature.body.body, surface = feature.sheet.body, keep = negative)
```

## Sheet metal, weldments and molds

| Family | Operations |
|---|---|
| Sheet metal | `sheet_metal.base_flange`, `edge_flange`, `miter_flange`, `hem`, `jog`, `sketched_bend`, `lofted_bend`, `convert_to_sheet_metal`, `closed_corner`, `break_corner`, `corner_relief`, `tab_and_slot`, `form_tool`, `unfold`, `fold`, `flat_pattern` |
| Weldments | `weldment.structural_member`, `trim_extend`, `end_cap`, `gusset`, `weld_bead` |
| Molds | `mold.shrink`, `parting_line`, `shut_off`, `parting_surface`, `core_cavity`, `undercuts` |

A sheet-metal part carries its rules (thickness, bend radius, K-factor) on its base flange, and `flat_pattern` unfolds it from a fixed face. A weldment's structural members follow the segments of a layout sketch with a profile and a material, and its cut list is a drawing table.

<!-- design:clip:mcad/weldment -->

<!-- design:clip:mcad/mold -->

```part fragment
feature base = sheet_metal.base_flange(profile = sketch.s_chain, depth = 40 mm, rules = { thickness = 2 mm, bend_radius = 3 mm, rule = k_factor(k = 0.44) })
feature flat = sheet_metal.flat_pattern(fixed_face = feature.base.face.sheet(base, ordinal = 0))
feature members = weldment.structural_member(groups = [{ sketch = layout, segments = [bottom, right, top, left], profile = rect_tube(width = 0.02, height = 0.02, wall = 0.002), material = S235 }])
feature shrink = mold.shrink(bodies = [feature.body.body], x = 0.005, y = 0.005, z = 0.005)
feature draft_check = mold.parting_line(body = feature.body.body, pull = origin.z, min_draft = 1 deg)
```

## Direct edits

`direct.move_faces`, `direct.replace_faces`, `direct.delete_faces` and `direct.move_bodies` edit imported or finished geometry as features in the history. `keep` lists relations the edit must keep, such as two faces staying parallel; an edit that would break one is an error (`cad.direct.relation_broken`). In the Desktop, Move face's Keep field writes these relations.

`direct.move_faces` takes one of three motions:

| Motion | What moves | Desktop |
|---|---|---|
| `offset(distance)` | Each face moves along its own outward normal by `distance`. A hole's radius changes, and an angled face moves square to itself. | Offset face, Pull, and Move face → Offset |
| `along(direction, distance, reverse)` | The faces move together as one rigid set, `distance` along `direction`. An edge or axis gives its own direction; a plane or planar face gives its normal, written `{ along = normal, plane = origin.xy }` for the Top plane. `reverse = true` flips the direction. | Move face → Translate along a direction |
| `rotate(axis, angle)` | The faces turn together about `axis` by `angle`. | Move face → Rotate about an axis |

In every motion the neighboring faces extend or trim to meet the moved faces. Move face → Offset and Offset face write the same feature.

**Instant 3D** (`direct.instant-3d`) drags a face in the view by its arrow, or takes a typed distance. When a feature dimension drives the face, Instant 3D changes the dimension: the end cap of a blind extrude (`feature.<f>.face.end`), or the extrude picked itself, changes the extrude's `distance` by the drag (twice the drag for a mid-plane extrude, which grows both ways). A literal depth is rewritten (`10 mm` dragged 3 mm becomes `13 mm`); a depth written as an expression gains `+ 3 mm`. A drag that would leave the depth at zero or less is refused. Any other face moves as a direct edit, `direct.move_faces(faces = [<face>], motion = offset(distance = <drag>))`, the same feature Pull writes.

```part fragment
feature lift = direct.move_faces(faces = [feature.body.face.side(s1.l2)], motion = along(direction = { along = normal, plane = origin.xy }, distance = 16.59 mm))
```

<!-- design:clip:mcad/direct -->

```part fragment
feature pull = direct.move_faces(faces = [feature.body.face.end], motion = offset(distance = 5 mm), keep = [{ keep = parallel, a = feature.body.face.end, b = feature.body.face.start }])
```
