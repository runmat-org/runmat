---
description: Reference for sketches in a RunMat assembly. Sketch on an assembly origin plane or on a placed component's face, with the same tools, relations and dimensions as a part sketch, and keep the sketch on the face when the component moves.
keywords: [assembly sketch, in-context sketch, layout sketch, sketch on a component face, top-down design, sketch in an assembly]
---
# Sketches in an assembly

An assembly can hold its own sketches. Use one to lay out where components go, or to draw on a component's face where it sits in the assembly. An assembly sketch has the same entities, relations, dimensions and solver as a part sketch; see [Sketches](/docs/mcad/part/sketches).

This assembly has two sketches. `layout` sits on the assembly's XY plane. `slot` sits on the top face of the `upper` plate, through the connector `top`, and it moves with that plate.

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

    connector top = { occurrence = [{ name = upper }], at = [0 mm, 0 mm, 10 mm] }

    sketch layout on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        constrain horizontal(l1)
    }

    sketch slot on plane.top {
        c1 = circle(center = [20 mm, 25 mm], radius = 4 mm)
    }
}
```

## Planes

An assembly sketch is written `sketch <name> on <plane> { … }`, as in a part. The plane is one of these:

| Plane | Meaning |
|---|---|
| `origin.xy`, `origin.yz`, `origin.zx` | An origin plane of the assembly. |
| `plane.<connector>` | The XY plane of one of the assembly's connectors, with its normal along the connector's Z axis. |

A connector is fixed in its component (see [Occurrences and mates](/docs/mcad/assembly/mates)). A sketch on a connector therefore follows that component when its placement changes, when mates move it, or when it is dragged. A connector anchored to a face the part publishes also follows that face when the part changes.

A sketch whose connector is missing, or whose component is suppressed, doesn't draw. The rest of the assembly still builds, and the error `cad.asm.unknown_connector` names the sketch.

An assembly's `feature` entries hold sketches only. Any other feature kind is refused with `cad.ref.kind_mismatch`.

## Layout geometry for mates

A mate can pick a point, line, circle or arc of an assembly sketch as one of its two sides. The mate adds a connector of the assembly itself, anchored to that sketch entity:

| Picked | The connector |
|---|---|
| A sketch point, or a named point of an entity (`l1.end`, `c1.center`) | At the point, with Z along the sketch's normal. |
| A circle or arc | At its center, with Z along the sketch's normal, so a concentric mate puts a shaft on it. |
| A line | At its start, with Z along the line and X along the sketch's normal. |

In this assembly the point `p1` of the sketch `marks` places the cover: a ball mate puts the cover's origin on the point.

```assembly file=marked.assembly
runmat 0.7
std 1

assembly marked {
    use part plate as base
    use part plate as cover {
        fixed = false
        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 30 mm] } }
    }

    sketch marks on origin.xy {
        p1 = point(at = [100 mm, 0 mm])
    }

    connector mark = { occurrence = [], anchor = { anchor = sketch, sketch = marks, entity = p1 } }
    connector corner = { occurrence = [{ name = cover }] }

    mate on_mark = { kind = { mate = ball }, a = mark, b = corner }
}
```

The layout drives the mate. When you edit the sketch, for example by moving `p1`, the component mated to it moves too.

The sketch is solved before the mates that read it, so it can't move with them. It must sit on an origin plane or on a fixed component. A mate that reads a sketch on the component it moves, or on a component that other mates move, is refused: the component would follow its own sketch. In a file, such a connector reports the same error.

## Converted edges

A sketch entity written with `from` takes its shape from edges a component publishes. The assembly reads those edges where the component sits, projects them onto the sketch's plane and holds the entity fixed there. When the component moves, or its part changes, the entity moves with the edges, and the curves related or dimensioned to it follow.

```text
sketch outline on origin.xy {
    c1 = circle(center = [60 mm, 30 mm], radius = 5 mm, from = published(occurrence = pin, name = rim))
    c2 = circle(center = [60 mm, 30 mm], radius = 8 mm)
    constrain concentric(c1, c2)
}
```

- `occurrence` names the component, as a dotted path for one inside a sub-assembly. `name` is an edge set its part publishes (`publish rim = region(region = rim)`).
- The geometry written in the file is where the edges were at the last edit. Each rebuild replaces it with the projected edges.
- The edges are read as a line when they run straight, as a circle when they close on one radius, as an arc on one radius, and otherwise as a spline through them.
- The published name must be one connected chain of edges. An edge set that projects to a single point (an edge along the sketch's normal) is reported on the sketch.

**Use edges** (`sketch.use-edges`, type Project) writes these entities for you. Open a sketch, click **Use edges > Project edges**, and pick one or more component edges in the view. Each edge is published from its part (`publish mate_edge1 = entities(kind = edge, ...)`, the same entry a mate writes) and comes into the sketch as a converted line, circle, arc or spline. Intersection curve, Silhouette edges and Offset on surface aren't built yet; picking one of them reports that by name.

**Offset of a component edge:** in **Offset**, a click on a component's edge (outside the sketch's own curves) picks it. Confirm brings the edge in as construction geometry and adds the offset copy, related to it so it follows the edge:

```text
l1 = line(start = [30 mm, 10 mm], end = [50 mm, 10 mm], construction, from = published(occurrence = block, name = mate_edge1))
l2 = line(start = [30 mm, 15 mm], end = [50 mm, 15 mm])
constrain parallel(l1, l2)
constrain distance(l2.start, l1) = 5 mm
```

A circular edge's copy is concentric with it and takes the offset radius as a dimension.

## Profiles for assembly features

An assembly feature takes a closed region of an assembly sketch as its profile, written `profile = sketch.<name>`. The sketch is solved after placement and mates, so the profile is where the sketch is drawn in the assembly.

## In the Desktop

- **Start a sketch:** click **Sketch** on the Assemble tab and pick an origin plane or a planar face of a component.
  - A face pick adds a connector on that component, anchored to the face its part publishes. The sketch is written on that connector.
  - The sketch opens in sketch mode with the view normal to it.
- **Draw:** the sketch ribbon has the same tools, fields and handles as a part sketch: lines, rectangles, circles, arcs, slots, polygons, splines, ellipses, points and text; offset, trim, extend, split, mirror, pattern and transform; fillet and chamfer; relations, dimensions and auto-constrain.
  - Trim and Extend act on the curve you click, at once: Extend runs the end nearer the click to the next curve it meets. With a curve picked in the Outline instead, Extend's Confirm runs the end that meets a curve sooner.
  - Offset's first click on a sketch curve fills its Entities field; a second click sets the distance and side, or Confirm uses the distance in the form.
  - Transform's base point is optional: Move and Copy use the deltas, and Rotate and Scale turn about the sketch origin when no base point is picked.
- **Use edges:** **Use edges > Project edges** brings picked component edges into the sketch (see Converted edges).
- **Mate to a sketch:** in a mate, pick a sketch point, line, circle or arc as either side, in the view or under the sketch's row in the Outline. Outside sketch mode the sketch's curves and points still pick in the view.
- **Check a sketch:** **Check sketch for feature** checks it for an extrude, a revolve, a sweep path or a loft profile, as in a part; see [Checking a sketch for a feature](/docs/mcad/part/sketches#checking-a-sketch-for-a-feature).
- **Delete curves:** in sketch mode, pick curves or points and press Delete. Points other curves shared with them stay where they are, and the relations and dimensions on them go.
- **The Outline:** the **Sketches** folder lists each sketch with its plane and how it solved.
  - Right-click a sketch for **Edit sketch**, **Suppress** and **Delete**.

The agent and the command line run the same commands: `assembly.sketch` with `values.plane`, then each `sketch.*` tool with `values.sketch` naming the sketch.
