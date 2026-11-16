---
description: Build an assembly in RunMat MCAD. Place parts by name, set an occurrence's size, join moving parts with mates, repeat parts with patterns, drive the motion from an input, and check fit, interference and the bill of materials.
keywords: [build an assembly, CAD assembly tutorial, mates, revolute joint, place parts, pattern, bill of materials, interference check, clearance check, motion]
---
# Build an assembly

This guide builds a small rotary stage: a base plate, a turntable that turns on a revolute mate, and four standoffs repeated by a pattern. It shows how an assembly finds its parts, places them, joins the moving ones and checks the result.

## 1. The parts

An assembly places parts by their names. Parts in the same folder need no import, and a `+<package>/` folder above the assembly holds parts many assemblies share. This page's three parts sit beside the assembly:

```part file=base.part
runmat 0.7
std 1

part base {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])
        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 100 mm])
        l3 = line(start = [100 mm, 100 mm], end = [0 mm, 100 mm])
        l4 = line(start = [0 mm, 100 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s, distance = 8 mm)
    output base = { body = feature.body.body, part_number = "RS-BASE", material = "material.aluminum_6061" }
}
```

```part file=turntable.part
runmat 0.7
std 1

part turntable {
    input radius : length = 35 mm
    sketch s on origin.xy {
        c = circle(center = [0 mm, 0 mm], radius = input.radius)
    }
    feature body = extrude(profile = sketch.s, distance = 6 mm)
    output turntable = { body = feature.body.body, part_number = "RS-TT", material = "material.aluminum_6061" }
}
```

```part file=standoff.part
runmat 0.7
std 1

part standoff {
    input height : length = 12 mm
    sketch s on origin.xy {
        c = circle(center = [0 mm, 0 mm], radius = 3 mm)
    }
    feature body = extrude(profile = sketch.s, distance = input.height)
    output standoff = { body = feature.body.body, part_number = "SO-M3", material = "material.steel_304" }
}
```

## 2. Place the parts

`use part <design> as <occurrence>` places a design. The occurrence's block sets where it goes and how it is configured. An occurrence is fixed at its placement unless it says `fixed = false`, which makes the placement a starting pose that mates move.

```assembly file=stage.assembly
runmat 0.7
std 1

assembly stage {
    input angle : angle = 0 deg

    use part base as base
    use part turntable as table {
        fixed = false
        configuration = { select = inputs, values = { radius = 40 mm } }
        placement = { placement = offset, offset = { translation = [50 mm, 50 mm, 8 mm] } }
    }
    use part standoff as post {
        placement = { placement = offset, offset = { translation = [10 mm, 10 mm, 8 mm] } }
    }

    connector axis = { occurrence = [{ name = base }], anchor = { anchor = explicit, offset = { translation = [50 mm, 50 mm, 8 mm] } } }
    connector hub = { occurrence = [{ name = table }], anchor = { anchor = origin } }
    mate spin = { kind = { mate = revolute, angle = { target = exact, value = input.angle } }, a = axis, b = hub }

    pattern posts = grid(seeds = [post], first = [1, 0, 0], first_spacing = 80 mm, first_count = 2, second = [0, 1, 0], second_spacing = 80 mm, second_count = 2)

    part_number = "RS-100"
}
```

- `configuration = { select = inputs, values = { radius = 40 mm } }` sets this occurrence's radius. The same `turntable.part` can serve another stage at its default 35 mm.
- `placement = offset` puts the occurrence at a translation and an optional rotation in the assembly's axes.

## 3. Join the moving parts

A connector is a frame on an occurrence; its Z axis is the joint axis. A mate joins two connectors and leaves some motion free. `revolute` leaves rotation about Z, so the turntable spins about the axis at the base's center.

The mate's `angle` drives that rotation from the assembly's `angle` input. Set the input in Properties, or in a run, and the turntable turns. A range in place of an exact value (`{ target = range, min = 0 deg, max = 270 deg }`) limits the motion, and the solve reports `cad.asm.limit_active` when the turntable rests on a limit.

In the Desktop, Mate takes two faces picked on the model and a mate type. The preview shows where the floating occurrence goes, and confirming moves it there. Here a concentric mate puts a hinge pin in its bore:

<!-- design:clip:mcad/mate -->

The solve reports an occurrence that still moves freely (`cad.asm.under_defined`) and mates that contradict each other (`cad.asm.mate_conflict`). See [Occurrences and mates](/docs/mcad/assembly/mates) for every mate kind, including gear, rack-and-pinion, screw and pin-in-slot mates.

## 4. Repeat parts

`pattern posts = grid(…)` copies the standoff into a 2 × 2 grid 80 mm apart. Each copy has a key that later statements name it by, and `skip` leaves copies out. A `follow_pattern` places one copy per hole of a hole pattern a part publishes, so fasteners stay on their holes as the part changes. See [Patterns, mirrors and fasteners](/docs/mcad/assembly/patterns-and-mirrors).

To place one more of a mated component somewhere else, use **Copy with mates** on the Assemble tab. Pick the components, then one new reference per mate that joins them to something outside the copy, in the order the Outline lists those mates. The copy (`block2` for `block`) gets its own connectors and mates, and each outside mate end moves to its new reference. A mate end with no pick left keeps its original reference. Mates between two copied components join the two copies. The copy floats, so its mates place it.

## 5. Check the assembly

Evaluate measures the assembly in its current pose:

- **Interference** lists pairs of occurrences whose solids overlap, with the overlap volume.
- **Clearance** lists pairs closer than a gap you set, such as 0.5 mm between the turntable and the standoffs.
- **Mass properties** adds every part at its placement: total mass, center of gravity and inertia.
- **Assembly visualization** sorts the part occurrences by mass, volume, cost or count, largest first, and colors the model in five bands from the smallest value (blue) to the largest (red). Cost reads each component's `cost` property, else the `cost` property of its part. Count is how many occurrences share the part, such as 4 for each standoff. Selecting a row selects that component; closing the results clears the colors.

The bill of materials lists each part number with its quantity: one base, one turntable and four standoffs. A drawing places the same list as a BOM table with balloons. See [Evaluate](/docs/mcad/evaluate).

## 6. Show a component as its envelope

A component's representation sets what the assembly draws and checks for it. Right-click a component and choose **Representation**, or set it in the component's Properties:

- **Full** draws and checks the part's exact solid. It is the default and writes nothing.
- **Envelope** draws the component as its bounding box, shaded translucent, and interference and clearance checks use that box. A sub-assembly at envelope draws one box around all its parts.
- **Defeatured**, **Mid-surface** and **Beam** draw the part's own analysis representation of that kind, which the part declares (see [Analysis representations](/docs/mcad/part/regions-and-publish#analysis-representations)). A part that declares none of that kind is refused with its file named. When the part declares several of one kind, the component draws the first.

```assembly fragment
use part base as base {
    representation = envelope
}
```

Use an envelope to keep space for a component whose detail doesn't matter here, such as a purchased motor or a neighboring module. Here the stage's base draws as its box. Mass properties still add the parts an envelope stands for at their own mass and center of gravity.

## 7. Reserve space

An envelope is a box of space the assembly keeps for something, such as room to service a part or the path a cable runs along. **Envelope** on the Layout tab fits the box round the components you pick and adds a margin on every side. It writes one line:

```assembly fragment
envelope service = { owner = base, min = [-5 mm, -5 mm, -5 mm], max = [105 mm, 85 mm, 15 mm], keep_out = true }
```

The view draws the box's edges, and the Outline lists it under **Envelopes**. With **Keep out** on, Interference lists every component inside the box except its owner, the one component allowed in it. Run Envelope from the row to change it: with no components picked, the box stays and only the other settings change. Delete removes it.

## 8. Drive the layout from a skeleton

A skeleton is a part that holds the assembly's layout: sketches, planes and frames that the other parts are placed against. **Skeleton part** on the Layout tab creates the part beside the assembly and places it at the origin as a reference component. Name a part already beside the assembly to use that one instead.

```part file=layout.part
runmat 0.7
std 1

part layout {
}
```

```assembly fragment
use part layout as layout {
    reference = true
}
```

Open the skeleton to draw the layout and publish the frames other parts attach to. A reference component is drawn, picked and mated to like any other, and is left out of the parts list, mass properties and interference checks. Set `reference = true` on any component to treat it the same way.

## 9. Name the contact regions

A study needs to know which faces of different components touch. **Contact regions** on the Simulate tab scans every pair of components and names each pair of touching faces as a contact region. Two flat faces touch when they face each other, the gap between them is within the gap tolerance (0.05 mm by default), and part of one lies on the other. Two cylindrical faces touch when they share an axis and a radius within the tolerance and overlap along the axis, such as a pin in its hole.

```assembly fragment
contact contact1 = { a = { occurrence = [{ name = plate }], face = top }, b = { occurrence = [{ name = block }], face = bottom }, bonded = true }
```

Each face is published from its part, so the region follows the part's edits. A bonded region is glued together in a study. Clear **Treat as bonded** to let the faces slide or separate. Running the command again adds only pairs that have no region yet. To change whether a region is bonded, run Contact regions from its row under Contact regions in the Outline. Reference and suppressed components are left out.

## 10. Offer product options

One assembly file can describe every option a product is sold with. **Options and effectivity** on the Manage tab adds an option and its values, the first one standard. The optional effectivity limits the other values to a serial range: `MSN 1-99` declares the `msn` axis and writes the rule `engine_effectivity`, so engine `b` is legal only on MSN 1 to 99. Run the command again with the same option name to change its values or range.

```assembly fragment
options = { axes = [{ name = engine, domain = { domain = enum, values = [a, b] }, default = a }, { name = msn, domain = { domain = unit, min = 1, max = 99 }, default = 1 }], rules = [{ name = engine_effectivity, condition = { when = any, of = [{ when = not, of = { when = is, axis = engine, value = a } }, { when = range, axis = msn, min = 1, max = 99 }] } }] }
```

A component written with `when` appears only in the configurations its condition allows. **Options** in the document bar picks the values the assembly shows, such as `engine=b` and `MSN 42`, and checks each against its option. The Outline lists each option under Options with its values, the value shown and its effectivity. Delete removes an option once no component or rule reads it.

## 11. Release parts and pin components

A component follows its part file live: every edit to the part shows in the assembly. To freeze a component at a released version, release the part and pin the component to it.

**Release** on the Manage tab of a part saves the part's text as its next released revision, `R1`, then `R2`. In an assembly, **Release** releases the files of the components you select, or the assembly itself when nothing is selected. The project keeps each file's revisions beside it in `.runmat/releases/`, for example `.runmat/releases/turntable.part.json`. Releasing a file that has not changed since its latest release is refused.

**References and pins** on the assembly's Manage tab updates the components you pick, or every component when you pick none:

- **Latest released** pins each one to its file's latest released revision.
- **Latest** sets each one to follow its file live again.

```assembly fragment
use part turntable @ "R1" as turntable
```

A pinned component regenerates from the released text, so it keeps its shape, mass and connectors while the part file changes. A pin to a revision the project does not hold is reported on the component.

## 12. Choose the units

**Document settings** on the Manage tab sets the units the assembly reads in: millimeter, gram, second (`mmgs`); meter, kilogram, second (`mks`); or inch, pound, second (`ips`).

```assembly fragment
settings = { units = ips }
```

Mass properties, Measure and the BOM read in these units. With units set, the BOM adds a **Mass each** column: the mass of one of each line, such as one standoff. Without settings, the assembly reads in millimeters, with masses in grams and kilograms. The units change only how values read: the model is the same.

## Next

- [Make a drawing](/docs/mcad/guides/make-a-drawing) of the assembly with its BOM.
- [Motion studies](/docs/mcad/assembly/motion) turn the table through its range over time and check the motion for collisions.
- [Scripting and sweeps](/docs/mcad/scripting) runs the stage at many angles from a `.m` script.
