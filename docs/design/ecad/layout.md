---
description: Reference for RunMat board files (.circuit.layout). Stackups and rigid-flex regions, outline and cutouts, limits, net classes and rules, rooms, footprints and packages, tracks, arcs, vias, pours, keepouts, text and dimensions, the sync section, and the design-rule checks.
keywords: [PCB layout, board file, stackup, rigid flex, footprints, tracks, vias, copper pour, keepout, design rules, DRC, clearance, differential pairs, length matching]
---
# Board layout

A `.circuit.layout` file is the board for one circuit. Every item is one statement, and every statement but the stackup and package blocks is one line, so a board with ten thousand tracks diffs line by line in review. Lengths are millimeters, exact to the nanometer; any length may carry a unit (`50 mil`). Angles are degrees.

## A board file

The board names its circuit, which sits beside it, then declares its outline and limits. The editor and the agent add the rest as the board is laid out.

```circuit file=sensor.circuit
runmat 0.7
std 1

// A 3.3 V rail from a 5 V input, a power LED, and an RC filter ahead of an ADC input.
circuit sensor {
    root = main
    net_class power = { track_width = "0.40 mm", clearance = "0.20 mm" }
    variant lite = { overrides = { D1 = not_fitted } }

    sheet main {
        component U1 = { part = "LDO-3V3-SOT23" }
        component C1 = { part = "C0805-10U" }
        component C2 = { part = "C0402-1U" }
        component D1 = { part = "LED0603-G" }
        component R1 = { part = "R0402-1K" }
        component R2 = { part = "R0402-10K" }
        component C3 = { part = "C0402-100N" }
        component P1 = { part = "PAD-WIRE-20AWG" }
        component P2 = { part = "PAD-WIRE-20AWG" }
        component P3 = { part = "PAD-WIRE-20AWG" }
        net GND = { class = power, global = true, power_flag = true }
        net VIN = { class = power, power_flag = true }
        net "+3V3" = { class = power, global = true }
        connect = [net.VIN, "P1.1", U1.VIN, U1.EN, "C1.1"]
        connect = [net.GND, "P2.1", U1.GND, "C1.2", "C2.2", D1.K, "C3.2"]
        connect = ["net.+3V3", U1.VOUT, "C2.1", "R1.1"]
        connect = [net.LED, "R1.2", D1.A]
        connect = [net.SENSE, "P3.1", "R2.1"]
        connect = [net.ADC, "R2.2", "C3.1"]
        no_erc = [U1.NC]
    }

    // The 10 k / 100 nF filter ahead of the ADC: a 159 Hz corner.
    simulation filter {
        sheet = main
        source VS = { p = net.SENSE, ac = 1 V, pulse = { from = 0 V, to = 1 V, delay = 1 ms, rise = 1 us, fall = 1 us, width = 20 ms, period = 40 ms } }
        tolerance resistors = { match = "R*", value = 1 % }
        tolerance capacitors = { match = "C*", value = 10 % }
        analysis step = tran { stop = 10 ms, step = 10 us }
        analysis response = ac { from = 1 Hz, to = 100 kHz, points = 50 }
        probe = [net.SENSE, net.ADC]
        measure corner = { of = response, bandwidth = net.ADC, spec = { min = 140 Hz, max = 180 Hz } }
        measure rise = { of = step, rise_time = net.ADC, from = 1 ms, to = 10 ms, spec = { max = 3 ms } }
        measure settled = { of = step, final = net.ADC, spec = { min = 0.99 V, max = 1.01 V } }
    }
}
```

```circuit.layout file=sensor.circuit.layout
runmat 0.7
std 1

use circuit sensor

layout sensor {
    outline = [[0, 0], [40, 0], [40, 25], [0, 25]]
    limits = { clearance = 0.15, track = 0.15, via_diameter = 0.45, drill = 0.2, annular = 0.1, hole_to_hole = 0.25, copper_to_edge = 0.3, silk_to_pad = 0.1, microvia_drill = 0.1 }
    net_class power = { track_width = 0.4 mm, clearance = 0.2 mm }
}
```

## Statements

```circuit.layout fragment
stackup main {
    finish = ENIG
    layer "F.Cu" = { kind = copper, thickness = 0.035, weight = 1, material = copper }
    layer core = { kind = core, thickness = 1.51, material = fr4 }
    layer "B.Cu" = { kind = copper, thickness = 0.035, weight = 1, material = copper }
}
outline = [[0, 0], [100, 0], [100, 80], [0, 80]]
cutout = [[40, 30], [40, 50], [60, 50], [60, 30]]
net_class power = { track_width = 0.5 mm, clearance = 0.2 mm }
rule hv = { select = ":net-class(hv)", clearance = 0.8 mm, priority = 2 }
rule hv_lv = { between = [":net-class(hv)", ":net-class(lv)"], clearance = 1.5 mm }
net GND : power
net SW
room ch0 = { outline = [[0, 0], [20, 0], [20, 20], [0, 20]], members = [R1, C1] }
footprint R1 = { package = R0402, at = [10.5, 20], rotation = 90, side = bottom, locked = true, pads = { "1" = GND, "2" = SW } }
fiducial FID1 = { at = [3, 3], diameter = 1, opening = 2 }
track GND = { layer = F.Cu, width = 0.25, from = [10, 20], to = [30, 20] }
arc SW = { layer = F.Cu, width = 0.25, from = [0, 0], mid = [1, 1], to = [2, 0] }
via GND = { at = [5, 5], size = 0.6, drill = 0.3 }
pour GND = { name = gnd, layers = [F.Cu, B.Cu], outline = [[0, 0], [100, 0], [100, 80]], priority = 1 }
text "REV A" = { layer = F.SilkS, at = [90, 5], size = 1, thickness = 0.15 }
package R0402 {
    smd = true
    pad "1" = { shape = roundrect, size = [0.5, 0.6], ratio = 0.25, at = [-0.5, 0] }
    pad "2" = { shape = roundrect, size = [0.5, 0.6], ratio = 0.25, at = [0.5, 0] }
    courtyard = [[-1, -0.5], [1, -0.5], [1, 0.5], [-1, 0.5]]
}
```

| Statement | Declares |
|---|---|
| `use circuit <name>` | the circuit the board belongs to (at file level), declared by a `.circuit` file in the same folder; add `from "./<folder>"` for another folder |
| `stackup <name> { finish, layer … }` | the layer stack: copper, prepreg, core and mask layers with thickness, copper weight and material (`fr4`, or a record with `dk`, `df`, thermal conductivity, density and heat capacity) |
| `region <name> = { stackup, outline, bends }` | a rigid-flex region with its own stackup and bend lines |
| `outline`, `cutout` | the board edge and holes in it, as point rings |
| `limits = { clearance, track, via_diameter, drill, annular, hole_to_hole, copper_to_edge, silk_to_pad, microvia_drill }` | the fabricator's minimums |
| `fabrication = { … }` | fabrication settings |
| `net_class <name> = { track_width, clearance, … }` | class rules; the circuit's classes come across |
| `rule <name> = { select or between, clearance, …, priority }` | a scoped rule; selectors such as `":net-class(hv)"` pick nets, and `between` sets a rule for a pair of sets |
| `net <name> [: <class>]` | a net of the board |
| `diff_pair`, `component_class` | pairs and classes from the circuit, with board rules |
| `room <name> = { outline, members }` | a placement area for a group, such as one channel |
| `footprint <designator> = { package, at, rotation, side, locked, pads, value, properties }` | a placed component; `pads` maps each pad to its net |
| `package <name> { pad …, courtyard, silk, fab, model, body }` | a footprint's package, defined once in the file so the board is self-contained |
| `track`, `arc` | copper segments on a layer, named by net |
| `via <net> = { at, size, drill, layers }` | a via; `layers` makes it blind or buried |
| `pour <net> = { name, layers, outline, priority }` | a copper zone, filled around everything else |
| `keepout <name> = { outline, layers, tracks, vias, pours }` | an area that forbids what it lists |
| `fiducial`, `text`, `graphic`, `dimension` | fiducials, silkscreen and fabrication-layer text, lines and dimensions |
| `sync { … }` | the link from footprints to schematic components and the schematic changes the board has taken |

## Board-house capability sets

`limits = { capability = <set> }` loads a named set of fabricator minimums. A value written beside `capability` overrides the set's value for that key, and every other key takes the set's value:

```circuit.layout fragment
limits = { capability = advanced, copper_to_edge = 0.4 }
```

| Set | Track and space | Via diameter | Drill | Annular ring | Hole to hole | Copper to edge | Silk to pad | Microvia drill |
|---|---|---|---|---|---|---|---|---|
| `standard` | 0.15 mm | 0.6 mm | 0.3 mm | 0.15 mm | 0.5 mm | 0.3 mm | 0.15 mm | 0.1 mm |
| `advanced` | 0.1 mm | 0.45 mm | 0.2 mm | 0.1 mm | 0.25 mm | 0.25 mm | 0.1 mm | 0.1 mm |
| `hdi` | 0.075 mm | 0.3 mm | 0.15 mm | 0.075 mm | 0.2 mm | 0.2 mm | 0.075 mm | 0.075 mm |

Compare the set with your fabricator's published capabilities before you order, and write any value that differs beside `capability`. The Board tab's **Board house** command (Rules group) sets `capability` and replaces any values written beside an earlier set. In the rules table, a value that comes from the set reads, for example, `0.30 mm (hdi capability)`. A DRC finding against such a value points at the `capability` line.

## Sync with the schematic

The `sync` section links each footprint to its component and records the schematic version the board last took. When the circuit changes (a component added, a net renamed, a part swapped), the check lists the changes the board has not taken, and taking them updates footprints and nets in one edit. `renamed_from` in the circuit keeps a renamed component's placement.

## Design rules

DRC runs incrementally as the board is edited, and in full in a check:

| Code | Finds |
|---|---|
| `clearance` | copper of different nets closer than the rule allows, zone fills included |
| `track-width` | a track outside its width rule |
| `annular-ring`, `hole-size`, `hole-to-hole` | drill and ring sizes and spacing below the limits |
| `via-type` | a via kind the rules do not allow; a microvia spanning more than one layer pair |
| `silk-to-pad`, `courtyard-overlap` | silkscreen on pads; courtyards that overlap |
| `edge-clearance` | copper too close to the edge or a cutout |
| `keepout` | an item inside a keepout that forbids it |
| `layer-not-in-region`, `via-in-bend` | copper on a layer a flex region lacks; a via in a bend |
| `unrouted` | a net whose copper is in more than one piece |
| `length`, `skew` | a net outside its length range; a match group or pair with too much skew |
| `diff-pair-gap`, `diff-pair-uncoupled` | a pair's gap off target; too much uncoupled length |
| `min-spokes` | a thermal relief with too few spokes |

## In the assembly

A board is a component of the product: an assembly places it as a part with its 3D model, and each component on it is its own face, selected by `component:<designator>`, so a pick in 3D selects the part on the schematic and the board. A board placed in an enclosure checks its fit and clearance against the enclosure's parts [not yet in the current release](pending:design.ecad.enclosure), and board studies (thermal and vibration) read the board's stackup and component power through a study's `pcb` source [not yet in the current release](pending:design.ecad.board-studies).
