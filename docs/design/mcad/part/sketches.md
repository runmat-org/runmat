---
description: Reference for RunMat sketches. Sketch planes, lines, arcs, circles, splines, points and construction geometry, shared points, geometric and dimensional constraints, limits and the solved cache.
keywords: [sketch, 2D sketch, constraints, parametric sketch, coincident, tangent, horizontal, vertical, dimension, construction geometry, spline, sketch solver]
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

## Entities

| Entity | Form |
|---|---|
| Line | `l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])` |
| Arc | `a1 = arc(center = [0 mm, 0 mm], start = [10 mm, 0 mm], end = [0 mm, 10 mm])` |
| Circle | `c1 = circle(center = [40 mm, 25 mm], radius = 5 mm)` |
| Spline | `lobe = spline(points = [[10 mm, 0 mm], [0 mm, 10 mm], [-10 mm, 0 mm]], closed)` |
| Point | `p1 = point(at = [3 mm, 4 mm])` |

- A point is `[x, y]` with a unit on each coordinate, or `[x, y] mm` with one unit after the list.
- **Shared points.** Writing another entity's point (`start = bottom.end`) makes the two the same point, so the loop stays closed as the solver moves it. Entity points are `start`, `end` and `center`; a point entity's position is `p1.at`.
- **Construction geometry.** `construction` after an entity's arguments makes it a reference for constraints only; it never becomes part of a profile.
- Sketch points are read elsewhere as `sketch.<sketch>.<entity>.<point>`, such as a hole at `sketch.marks.p1.at`.

## Constraints

`constrain <relation>` states a relation the solver keeps. A dimensional constraint takes its value after `=`, and the value may be any expression (`= input.length`, `= 2 * input.radius`).

| Constraint | Form |
|---|---|
| Fix | `constrain fix(p1)`, `constrain fix(l1.start)` |
| Coincident | `constrain coincident(l1.end, l2.start)` |
| Horizontal, vertical | `constrain horizontal(l1)`, `constrain vertical(l2)` |
| Parallel, perpendicular | `constrain parallel(l1, l3)`, `constrain perpendicular(l1, l2)` |
| Tangent | `constrain tangent(l1, a1)` |
| Equal | `constrain equal(a1, a2)` |
| Distance | `constrain distance(l1.start, l2.end) = 60 mm` |
| Length | `constrain length(l1) = 80 mm` |
| Radius, diameter | `constrain radius(c1) = 5 mm`, `constrain diameter(c1) = 10 mm` |
| Angle | `constrain angle(l1, l2) = 90 deg` |
| Limit | `constrain limit(length(l3, value = 80 mm), min = 60 mm, max = 90 mm)`: a dimension that may move between bounds |

`id = <name>` names a constraint (`constrain length(l1, id = span) = 80 mm`), so a configuration or the editor can address it.

## Solving

The check reports a sketch that is over-constrained (with the conflicting constraints), under-constrained (with the free directions), or has redundant constraints, and a profile that does not close (`cad.sketch.open_profile`). The editor writes the solver's last result in a `solved { … }` block inside the sketch, in SI units, so a file reopens without re-solving:

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
