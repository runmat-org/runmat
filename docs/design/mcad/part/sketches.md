---
description: Reference for RunMat sketches. Sketch planes, lines, arcs, circles, splines, ellipses, text, points and construction geometry, shared points, geometric and dimensional constraints, limits and the solved cache.
keywords: [sketch, 2D sketch, constraints, parametric sketch, coincident, tangent, horizontal, vertical, midpoint, symmetric, dimension, construction geometry, spline, ellipse, elliptical arc, sketch text, engraving, embossing, sketch solver]
---
# Sketches and constraints

A sketch is 2D geometry on a plane or a planar face, solved from its constraints. Features use a sketch's closed loops as profiles (`profile = sketch.outline`), its open chains as paths, and its points as positions.

```part
runmat 0.7
std 1

part link {
    input length : length = 60 mm
    input radius : length = 8 mm

    sketch s on origin.xy {
        bottom = line(start = [0 mm, -8 mm], end = [60 mm, -8 mm])
        right = arc(center = [60 mm, 0 mm], start = bottom.end, end = [60 mm, 8 mm])
        top = line(start = right.end, end = [0 mm, 8 mm])
        left = arc(center = [0 mm, 0 mm], start = top.end, end = bottom.start)
        hole = circle(center = left.center, radius = 3 mm)
        axis = line(start = left.center, end = right.center, construction)
        constrain fix(axis.start)
        constrain horizontal(axis)
        constrain length(axis) = input.length
        constrain tangent(bottom, right)
        constrain tangent(top, left)
        constrain parallel(bottom, top)
        constrain equal(right, left)
        constrain radius(right) = input.radius
        constrain diameter(hole) = 6 mm
    }
    feature body = extrude(profile = sketch.s, distance = 5 mm)
    output link = { body = feature.body.body }
}
```

## Planes

`sketch <name> on <plane> { … }` places the sketch:

| Plane | Example |
|---|---|
| An origin plane | `on origin.xy`, `on origin.yz`, `on origin.zx` |
| A datum plane | `on plane.top_plane` |
| A planar face | `on feature.body.face.end` |

A sketch on a face moves with the face when an earlier feature changes.

To move a sketch to another plane, select it in the Outline and click **Edit plane** (`part.edit-sketch-plane`), then pick an origin plane, a datum plane or a planar face. The entities keep their sketch coordinates, so the drawing moves with the plane, and features built on the sketch follow on the next rebuild. A face made by the sketch's own feature or a later one is refused, because the sketch regenerates before that feature.

## Converted edges (Use edges)

**Use edges** (`sketch.use-edges`) brings the part's own edges and faces into the open sketch. Each new entity records where it comes from, and every rebuild reads it again from the model as it stands just before the sketch, carries it onto the sketch plane along the sketch normal and holds it fixed. The rest of the sketch can be dimensioned and related to it, and it follows the model when an earlier feature changes.

| Type | Picks | What comes into the sketch | Written |
|---|---|---|---|
| Project | Edges; a face (each edge round it); a body (every edge) | Each edge, projected | `from = project(<edge>)` |
| Intersection | Faces; a body (every face that crosses the plane) | Where each face crosses the sketch plane, one entity per connected piece | `from = intersection(<face>, chain = 0)` |
| Silhouette | Curved faces; a body (every curved face) | Where each face turns edge-on, seen square to the sketch, one entity per piece | `from = silhouette(<face>, chain = 0)` |
| Offset on surface | A planar face (each edge round it); or edges, with the planar face to offset them across | Each edge moved `distance` across the face, into it for a positive distance, then projected | `from = offset_on_surface(<edge>, face = <face>, distance = 3 mm)` |

For example, a pocket inset 4 mm from the top face's outline:

```text
sketch inset on feature.body.face.end {
    l1 = line(start = [-16 mm, -6 mm], end = [16 mm, -6 mm], from = offset_on_surface(edges_between(feature.body.face.end, feature.body.face.side(s1.l1)), face = feature.body.face.end, distance = 4 mm))
    ...
}
```

- The geometry written in the file is where the source was at the last edit. Each rebuild replaces it.
- The source is read as a line when it runs straight, as a circle when it closes on one radius, as an arc on one radius, and otherwise as a spline through it (to 0.01 mm).
- A cut through a cylinder reads as its exact circle, and a cylinder seen from the side outlines as its two exact generator lines. Other curved faces read as splines through the cut or the outline.
- `chain` picks one piece when a face gives several, numbered by where each piece's centre sits on the sketch plane (left to right, then bottom to top). A face that gives fewer pieces after a change is reported on the sketch.
- Offset on surface moves a straight edge along the face, square to the edge, and changes a circular edge's radius. Offsets of edges that met at a corner meet again where the two offset curves cross, so the inset of a face's outline is a closed profile. It takes planar faces; a curved face is refused by name.
- A source that projects to a single point (an edge along the sketch normal) and a planar face picked for Silhouette are reported by name.
- In an assembly, Use edges projects component edges instead; see [Assembly sketches](/docs/mcad/assembly/sketches#converted-edges).

## Entities

| Entity | Form |
|---|---|
| Line | `l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])` |
| Arc | `a1 = arc(center = [0 mm, 0 mm], start = [10 mm, 0 mm], end = [0 mm, 10 mm])` |
| Circle | `c1 = circle(center = [40 mm, 25 mm], radius = 5 mm)` |
| Spline | `lobe = spline(points = [[10 mm, 0 mm], [0 mm, 10 mm], [-10 mm, 0 mm]], closed)` |
| Ellipse | `e1 = ellipse(center = [0 mm, 0 mm], major = 20 mm, minor = 8 mm, angle = 30 deg)` |
| Elliptical arc | `e2 = ellipse(center = [0 mm, 0 mm], major = 20 mm, minor = 8 mm, start = [20 mm, 0 mm], end = [-20 mm, 0 mm])` |
| Text | `t1 = text(text = "RUNMAT", at = [5 mm, 5 mm], height = 6 mm, font = single_line, stroke = 1 mm)` |
| Point | `p1 = point(at = [3 mm, 4 mm])` |

- A point is `[x, y]` with a unit on each coordinate, or `[x, y] mm` with one unit after the list.
- **Shared points.** Writing another entity's point (`start = bottom.end`) makes the two the same point, so the loop stays closed as the solver moves it. Entity points are `start`, `end` and `center`; a point entity's position is `p1.at`.
- **Construction geometry.** `construction` after an entity's arguments makes it a reference for constraints only; it never becomes part of a profile.
- **Ellipses.** `major` and `minor` are the semi-axes, and `angle` turns the major axis counter-clockwise from the sketch's x axis (0 when left out). With `start` and `end`, the ellipse is the arc that runs counter-clockwise from `start` to `end`; both points lie on the ellipse. A closed ellipse is a profile on its own, and an elliptical arc closes a loop with other curves. Features cut and extrude it as an exact ellipse.
- Sketch points are read elsewhere as `sketch.<sketch>.<entity>.<point>`, such as a hole at `sketch.marks.p1.at`.

### Text

Text places lettering that features extrude (embossed) or cut (engraved). `at` is the left end of the first line's baseline, `height` is the capital height, and `angle` turns the baseline counter-clockwise (0 when left out). `\n` in the string starts a line below the first.

`font` names the typeface. `single_line` is the default and the only font today: each glyph is drawn as pen strokes, and the profile is the outline of those strokes drawn with a round pen `stroke` wide. When `stroke` is left out, it is 0.15 of the height. The outline is exact lines and arcs, so a letter with a counter, such as `A` or `B`, cuts with its island left standing. Lowercase letters are small capitals, and a character outside the font draws as an empty box.

```part fragment
sketch label on feature.body.face.end {
    t1 = text(text = "RUNMAT", at = [8 mm, 6 mm], height = 6 mm, font = single_line, stroke = 1 mm)
    constrain fix(t1.at)
}
feature engrave = extrude(profile = sketch.label, distance = 0.5 mm, mode = cut)
```

The text's anchor is the point `t1.at`, so constraints place it like any other point. Text along a curve is not in the sketch model yet.

## Constraints

`constrain <relation>` states a relation the solver keeps. A dimensional constraint takes its value after `=`, and the value may be any expression (`= input.length`, `= 2 * input.radius`).

| Constraint | Form |
|---|---|
| Fix | `constrain fix(p1)`, `constrain fix(l1.start)` |
| Coincident | `constrain coincident(l1.end, l2.start)` |
| Horizontal, vertical | `constrain horizontal(l1)`, `constrain vertical(l2)`; on an ellipse, its major axis |
| Parallel, perpendicular | `constrain parallel(l1, l3)`, `constrain perpendicular(l1, l2)` |
| Tangent | `constrain tangent(l1, a1)`; lines, circles, arcs and ellipses in any pair except two lines |
| Equal | `constrain equal(a1, a2)`; two ellipses take equal semi-axes |
| Midpoint | `constrain midpoint(p1.at, l1)`: the point sits halfway along the line |
| Concentric, coradial | `constrain concentric(c1, c2)` (one center, ellipses included), `constrain coradial(c1, a1)` (one center and one radius) |
| Collinear | `constrain collinear(l1, l5)`: both lines lie on one infinite line |
| Symmetric | `constrain symmetric(l1.start, l2.end, axis)`: two points mirror each other in a line |
| Distance | `constrain distance(l1.start, l2.end) = 60 mm` |
| Length | `constrain length(l1) = 80 mm` |
| Radius, diameter | `constrain radius(c1) = 5 mm`, `constrain diameter(c1) = 10 mm` |
| Ellipse semi-axes | `constrain major_radius(e1) = 20 mm`, `constrain minor_radius(e1) = 8 mm` |
| Angle | `constrain angle(l1, l2) = 90 deg` |
| Limit | `constrain limit(length(l3, value = 80 mm), min = 60 mm, max = 90 mm)`: a dimension that may move between bounds |

`id = <name>` names a constraint (`constrain length(l1, id = span) = 80 mm`), so a configuration or the editor can address it.

## Solving

The check reports a sketch that is over-constrained (with the conflicting constraints), under-constrained (with the free directions), or has redundant constraints, a profile that does not close (`cad.sketch.open_profile`), and a profile whose sides cross (`cad.sketch.self_intersecting`). The crossing message names each pair of crossing lines and the point where they cross, in millimeters, for example `` `a` crosses `d` at (-12, 0) mm ``. The editor writes the solver's last result in a `solved { … }` block inside the sketch, in SI units, so a file reopens without re-solving. An ellipse's entries are `e1.major`, `e1.minor` and `e1.angle` (radians):

```part fragment
sketch s1 on origin.xy {
    l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
    c1 = circle(center = [40 mm, 25 mm], radius = 5 mm)
    constrain horizontal(l1)
    solved {
        l1.start = [0, 0]
        c1.radius = 0.005
    }
}
```

The `solved` block is a cache: editing the constraints re-solves the sketch and rewrites it.

## Checking a sketch for a feature

**Check sketch for feature** (`sketch.check`, in the sketch's Sketch tab and on its Outline row) checks a sketch for the feature it will make before you use it. Choose the use; the result lists each problem under the canvas, and selecting a row selects its curves in the sketch.

| Use | What it checks |
|---|---|
| Extrude | the curves close into loops: no open end, no branch (three or more ends at one point), no overlapping curves and no crossing |
| Revolve | the Extrude checks, and a construction line with the whole profile on one side of it, which Revolve can turn the profile about |
| Sweep path | one chain of curves end to end: one piece, with two free ends (an open path) or none (a closed path), and no branch or crossing |
| Loft profile | the Extrude checks, and exactly one closed loop |

Each finding states where it is, in millimeters, and an open end names the nearest other curve end and its distance, so a 0.05 mm gap reads as a gap to close. Construction curves and points are left out of the checks: they shape the sketch but are no part of a profile or path. A sketch in an assembly checks the same way; see [Sketches in an assembly](/docs/mcad/assembly/sketches).
