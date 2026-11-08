---
description: Reference for RunMat assemblies. Place parts and sub-assemblies as occurrences, set their configurations and placements, and join them with connectors and mates, including revolute, slider, gear, rack-and-pinion, screw, pin-slot and cam mates.
keywords: [assembly, occurrences, mates, mate connectors, revolute, slider, cylindrical, planar, ball, gear mate, rack and pinion, screw mate, pin slot, cam mate, kinematics, degrees of freedom]
---
# Occurrences and mates

An assembly places designs as occurrences and positions them with placements and mates. The mate solver moves floating occurrences until every mate holds, reports the degrees of freedom left, and names mates that conflict or repeat each other.

## A hinge

The base plate is fixed where it is placed. The arm floats, and a revolute mate turns it about the pivot by the `opening` input.

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
    feature body = extrude(profile = sketch.s, distance = 10 mm)
    output plate = { body = feature.body.body }
}
```

```part file=arm.part
runmat 0.7
std 1

part arm {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, -5 mm], end = [100 mm, -5 mm])
        l2 = line(start = [100 mm, -5 mm], end = [100 mm, 5 mm])
        l3 = line(start = [100 mm, 5 mm], end = [0 mm, 5 mm])
        l4 = line(start = [0 mm, 5 mm], end = [0 mm, -5 mm])
    }
    feature body = extrude(profile = sketch.s, distance = 4 mm)
    output arm = { body = feature.body.body }
}
```

```assembly file=hinge.assembly
runmat 0.7
std 1

assembly hinge {
    input opening : angle = 30 deg

    use part plate as base
    use part arm as arm {
        fixed = false
        placement = { placement = offset, offset = { translation = [40 mm, 25 mm, 10 mm] } }
    }

    connector pivot = { occurrence = [{ name = base }], anchor = { anchor = explicit, offset = { translation = [40 mm, 25 mm, 10 mm] } } }
    connector eye = { occurrence = [{ name = arm }], anchor = { anchor = origin } }

    mate hinge = { kind = { mate = revolute, angle = { target = exact, value = input.opening } }, a = pivot, b = eye }
}
```

## Occurrences

`use part <design> as <occurrence> { … }` (or `use assembly …`) places a design. The design is found by name through the folder scope; see [Folders, modules and use](/docs/design-language/modules). The block holds the occurrence's fields:

| Field | Meaning |
|---|---|
| `placement` | Where it goes: `{ placement = offset, offset = { translation = [x, y, z], rotation = … } }`, or `{ placement = attach, own = …, to = …, offset = …, flip = true }` to put one of its frames on a frame of the assembly or of an interface part. With no placement it sits at the assembly origin. |
| `fixed` | `true` (the default): the placement is final. `false`: the placement is a starting pose and mates move it. |
| `configuration` | `{ select = named, name = long }`, or input values: `{ select = inputs, values = { length = input.length } }`, optionally on a `base` configuration. Values may read the assembly's own inputs and lets. |
| `flexible` | A sub-assembly whose own occurrences join this assembly's mate solve, so a linkage inside it can move. |
| `when` | A condition on the assembly's options; alternatives of one occurrence share a name. |
| `required` | Some alternative of this name must be present in every legal configuration. |
| `material` | A material override of the design's own material. |
| `suppressed` | Leaves the occurrence out. |

Rotations are `{ rotation = axis_angle, axis = [1, 0, 0], angle = 90 deg }` or Euler angles. A pattern or mirror makes keyed copies of occurrences; see [Patterns, mirrors and fasteners](/docs/mcad/assembly/patterns-and-mirrors).

## Connectors

A connector is a frame on an occurrence (or on the assembly itself) that mates join. Its Z axis is the joint axis.

| Field | Meaning |
|---|---|
| `occurrence` | The occurrence path: `[{ name = base }]`; empty for the assembly's own frame. |
| `anchor` | `{ anchor = origin }` (the design's origin), `{ anchor = published, name = <frame> }` (a frame the design publishes, such as a `mate_connector` feature), or `{ anchor = explicit, offset = { translation = …, rotation = … } }`. |
| `offset` | A further offset in the anchor's axes. |
| `flip` | Turns the frame half a turn about its X axis, so two faces meet face to face. |

Publishing frames from the parts (`mate_connector` features) keeps an assembly's connectors on the geometry as the parts change.

### Seating a part on a curved body

A part that mounts to a curved surface, such as a clamp foot on a lofted lower shell, needs a connector at a point on that surface with its Z axis along the surface normal. `design.inspect` with the `surface` section finds both: give it a ray, a start point outside the body and a direction, and it returns where the ray meets each body, nearest first, with the face, the point in millimetres and the outward normal there.

```json
{ "path": "/drone.assembly", "sections": ["surface"],
  "ray": { "from": [-40, 30, -60], "direction": [0, 0, 1], "occurrences": ["body_bottom"] } }
```

The first hit with `"entering": true` is the underside under that point. Put the body's connector at `point_mm`, turned so its Z axis runs along `normal` (for a normal of `[0, 0, -1]`, `{ rotation = axis_angle, axis = [1, 0, 0], angle = 180 deg }`), and fasten the foot's connector to it. A flat foot touches a curved surface along a line only, so leave a standoff of about 0.5 mm between the foot and the surface, or follow the surface: trim the foot's top with a `loft` in `mode = cut` through the same section sketches the body's loft uses, so the foot takes the body's shape. Query each mounting station on its own; the normal changes along a loft.

After the change, check interference only for the new and moved occurrences with `changed` (see [Interference and clearance](/docs/mcad/evaluate#interference-and-clearance)).

## Mates

`mate <name> = { kind = { mate = <kind>, … }, a = <connector>, b = <connector> }` joins two connectors. Joint mates take an optional drive (`{ target = exact, value = 30 deg }`) or a range of motion (`{ target = range, min = 0 deg, max = 90 deg }`).

| Kind | Leaves free | Parameters |
|---|---|---|
| `fasten` | nothing | |
| `revolute` | rotation about Z | `angle` |
| `slider` | translation along Z | `distance` |
| `cylindrical` | rotation about and translation along Z | `angle`, `distance` |
| `planar` | motion in the XY plane | |
| `ball` | all rotation | |
| `parallel` | translation; the Z axes stay parallel | |
| `tangent` | a plane on a sphere or cylinder | `kind = { tangent = plane_sphere, radius }` or `plane_cylinder` |
| `distance` | everything but one distance | `measure = point_point` or `plane_point`, `target` |
| `angle` | everything but one angle | `measure = about_z` or `between_z`, `target` |
| `pin_slot` | a pin (`a`, its Z the pin axis) sliding along and turning in a slot (`b`, along its X) | |
| `screw` | `b` turns about `a`'s Z and advances `lead` per turn | `lead` |
| `gear` | `b` turns −1/`ratio` of `a`'s turn about its own axis | `ratio` (`b`'s teeth over `a`'s) |
| `rack_pinion` | a rack (`b`, along its X) moves `travel` per turn of the pinion (`a`) | `travel` (π × the pitch diameter) |
| `cam` | a follower point (`b`) stays on a cam's published profile (`a`, the closed path seen along its Z) as the cam turns; the follower may sit anywhere along the cam's Z | `profile` |

Gear and rack-and-pinion mates couple motion measured from the pose where the solve starts, so the authored placements set the phase.

```assembly fragment
mate drive = { kind = { mate = revolute, angle = { target = exact, value = 90 deg } }, a = base_pin, b = pinion_eye }
mate slide = { kind = { mate = slider, distance = { target = range, min = 0 mm, max = 120 mm } }, a = guide, b = rack_rail }
mate mesh = { kind = { mate = gear, ratio = 2 }, a = gear_a, b = gear_b }
mate rack = { kind = { mate = rack_pinion, travel = 100 mm }, a = pinion_eye, b = rack_teeth }
mate nut = { kind = { mate = screw, lead = 2 mm }, a = shaft_axis, b = nut_eye }
mate follower = { kind = { mate = pin_slot }, a = link_pin, b = base_slot }
mate gap = { kind = { mate = distance, measure = plane_point, target = { target = exact, value = 5 mm } }, a = face_c, b = tip_c }
```

```assembly fragment
mate lift = { kind = { mate = cam, profile = lobe }, a = cam_frame, b = follower_tip }
```

## What the solve reports

| Code | Meaning |
|---|---|
| `cad.asm.under_defined` | an occurrence still has free directions; the occurrence's degrees of freedom are listed |
| `cad.asm.mate_conflict` | a mate contradicts others, which it names |
| `cad.asm.mate_redundant` | a mate is implied by others |
| `cad.asm.limit_active` | a mate rests on its range limit |
| `cad.asm.mate_not_converged` | the solve could not satisfy the mates around one |
| `cad.asm.unknown_connector` | a mate uses a connector the assembly does not declare |
