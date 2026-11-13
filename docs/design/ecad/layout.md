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
package R0402 = {
    pads = [
        { number = "1", shape = roundrect, size = [0.5, 0.6], at = [-0.5, 0], layers = smd },
        { number = "2", shape = roundrect, size = [0.5, 0.6], at = [0.5, 0], layers = smd }
    ],
    courtyard = rect([-1, -0.5], [1, 0.5])
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
| `package <name> = { pads, courtyard, silk, fab, body }` | a footprint's package, defined once in the file so the board is self-contained; the same record as a library's `footprint <name> = { … }` |
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
| `length`, `skew` | a net outside its length range; a match group or pair with too much skew, in length or in ps |
| `length-match` | a match-group member outside its group's target by more than the tolerance, in mm or ps |
| `impedance`, `diff-impedance` | a track, or a coupled pair run, more than 10 % off its class's target impedance on its layer |
| `diff-pair-gap`, `diff-pair-uncoupled` | a pair's gap off target (a class's solved gap on that layer); too much uncoupled length |
| `return-path` | a controlled track crossing a split in its reference plane, or running over a gap in it |
| `via-stub` | a via whose stub past its net's connected layers is longer than `stub_max`, after any back-drill |
| `min-spokes` | a thermal relief with too few spokes |

## High-speed design

### Impedance-driven widths

Give a net class a target impedance in ohms, and RunMat solves its track width, and for a pair its gap, on each copper layer from the stackup:

```circuit.layout fragment
net_class rf = { impedance = 50 }
net_class usb = { diff_impedance = 90 }
net_class ddr_clk = { impedance = 40, diff_impedance = 80 }
net_class pcie = { diff_impedance = 85, diff_gap = 0.15 mm }
```

| Written | Solved per layer |
|---|---|
| `impedance` | the width that meets it |
| `diff_impedance` | the width and gap that meet it, with the gap equal to the width and at least the board's minimum clearance |
| `diff_impedance` and `impedance` | the width for the single-ended target, then the gap for the differential one |
| `diff_impedance` and `diff_gap` | the width that meets the target at the written gap |

Outer layers are surface microstrip over the nearest plane (Hammerstad–Jensen); inner layers are stripline between the planes above and below (IPC-2141A), with the dielectric's thickness-weighted Dk. Widths and gaps round to 1 µm. A layer whose solved width or gap is below the board's limits keeps the solved value and says so, and DRC flags tracks there against the minimum-track and clearance rules.

The solved values become the class's rules on each layer (`:net-class(usb):layer(In1.Cu)`), so they take precedence over the class's own `track_width`. They reach:

- **Route**, which draws at the net's width for the layer when you leave Width empty;
- **Diff pair**, which routes at the pair's solved width and gap for the layer when you leave them empty;
- **Autoroute**, which draws each layer's run at that layer's width;
- **DRC**, which checks each track against `impedance` and each coupled pair run against `diff_impedance` (both ±10 %, the tolerance board houses hold controlled impedance to), and the pair's gap against the layer's solved gap.

The outline lists each impedance class's width, gap and impedance per layer, and the **Layer stack** command reports them for the stackup you set.

### Matched-length groups and skew

Nets with the same `match_group` form a group. Set the group's target and tolerance on its rules; a delay unit (`ps`, `ns`) makes it a delay group, compared in picoseconds:

```circuit.layout fragment
rule byte0 = { select = ":net(DQ0, DQ1, DQ2, DQ3, DQ4, DQ5, DQ6, DQ7, DM0)", match_group = byte0, match_target = 420 ps, match_tolerance = 5 ps }
rule addr = { select = ":net(A*)", match_group = addr, match_tolerance = 0.5 mm }
rule clk = { select = ":net(CLK_P)", skew_max = 1 ps }
```

- With no `match_target`, the group matches its longest member (in a delay group, its slowest).
- With no `match_tolerance`, the tolerance is 0.1 mm, or 1 ps in a delay group.
- `skew_max` on a pair limits the difference between its two nets, as a length or a delay.

Each track's delay comes from its layer's propagation velocity: `√εeff / c`, where `εeff` is the stripline's Dk or the microstrip's effective permittivity at that width. A via adds the span between the layers its net connects on, at the Dk of the dielectric it crosses, plus half the delay of any stub (see [Via stubs and back-drill](#via-stubs-and-back-drill)). A member routed on an inner layer reaches a delay target with less copper than one on an outer layer.

The outline's **Length groups** folder lists each member's length or delay against the target: matched, short (needs tuning) or long (needs a shorter route). DRC reports each member outside the tolerance (`length-match`).

**Tune length** adds meanders to a net's longest straight track until it reaches a target length or delay, in one of three patterns:

| Pattern | Shape |
|---|---|
| Trombone | rectangular bumps to one side of the run |
| Accordion | rectangular bumps alternating sides, centered on the run |
| Sawtooth | triangular teeth to one side, for a run with little room beside it |

**Tune group** tunes every short member of a group to the group's target with the chosen pattern, and reports any member it could not bring in. Two short members that form a declared pair are tuned together: one meander follows the centreline of the pair's longest coupled run, and each net follows it at half the pair's pitch, so the pair keeps its gap and coupling through the meander. A rectangular bump adds the same length to both nets, so the pair's skew stays as it was; tune that separately with Tune length on the shorter net.

### Lengths through series parts

A signal that runs through a series resistor, a ferrite or another two-pin part is tuned as one signal when its rules set `through_parts`. The signal follows each two-pad part whose far net goes on to exactly one more pad, adds that net's copper, and adds the part's `through_delay`:

```circuit.layout fragment
rule dq = { select = ":net(DQ*)", match_group = byte0, through_parts = true }
rule series = { select = "footprint#R21", through_delay = 8 ps }
```

A part's body adds no length. Groups, DRC and Tune length use the signal's total.

### Via stubs and back-drill

A via is plated through every layer it is drilled through, but its net connects only where its tracks or pads touch the barrel. The barrel past the outermost connected layer is a stub. A through via that takes a signal from F.Cu to In1.Cu on a 1.6 mm board leaves about 1 mm of stub hanging down to the bottom copper. A stub resonates when it is a quarter wave long, so at multi-gigabit rates it notches the channel; well below that it loads the line as a small capacitance.

RunMat measures each via's stub from the stackup: from the centre of the last connected layer to the outer face of the drilled span's end copper. Set limits and back-drill on a net or class:

```circuit.layout fragment
net_class pcie = { diff_impedance = 85, stub_max = 0.25 mm, backdrill = true }
rule fab_tolerance = { select = ":net-class(pcie)", backdrill_stub = 0.15 mm, backdrill_oversize = 0.25 mm }
```

| Property | Meaning |
|---|---|
| `stub_max` | the longest stub allowed; DRC reports a longer one (`via-stub`) at the via |
| `backdrill` | `true` drills out each stub that reaches the board surface, from that surface |
| `backdrill_stub` | the stub the back-drill leaves past the connected layer, the fabricator's depth tolerance (0.2 mm when unset) |
| `backdrill_oversize` | how much wider the back-drill is than the via's drill (0.2 mm when unset) |

A back-drill enters from the stub's face and stops `backdrill_stub` short of the connected layer, its must-not-cut layer. A stub inside the board (a buried via's, or a blind via's inner end) can't be back-drilled; the `via-stub` finding says so, and a shorter via span is the fix.

Stubs count in delay, not in length. An open stub much shorter than a quarter wave acts as a shunt capacitance `C = t_stub / Z`, which delays an edge on a `Z` line by `Z·C/2`, so each stub adds half its own propagation delay to its net. Length groups, delay targets and Tune length use that delay. Back-drilling a 1 mm stub to 0.2 mm takes about 2.8 ps off the net.

The outline's **Via stubs** folder lists each via with a stub: its length as drilled and after back-drill, red when over `stub_max`. Its Properties give the connected span, the delay the stub adds, and each back-drill's face, drill, depth and must-not-cut layer.

**Fabrication outputs** write the back-drills as their own Excellon file per face and must-not-cut layer (`<board>-Backdrill-bottom-3-4.drl`: non-plated, controlled-depth, each tool's depth from its face), and the fabrication drawing adds a note per file, for example:

> Back-drill 4 holes from the bottom face with a 0.500 mm drill, 0.816 mm deep; must not cut layer 2 (In1.Cu); stub left at most 0.200 mm

### Return paths over plane splits

A controlled line's return current flows in its reference plane, under the track. Where the plane has a gap, or the track crosses from one pour to another of a different net, the return current detours around the gap: the line sees a step in impedance and the loop it forms radiates.

DRC checks each track of a net with an `impedance` or `diff_impedance` target against its reference planes: the copper layers next to it in the stack that carry a pour. It walks the track's centreline in 0.1 mm steps and reports `return-path` where:

- the pour under the track changes net (a split: `` `USB_DP` on F.Cu crosses a split in its reference plane In1.Cu (`GND` to `+3V3`) ``), marked where the first pour ends;
- the track runs over no pour for longer than `return_gap_max` (0.5 mm when unset), marked where the gap starts. The default is longer than a via's antipad, so a track's own via doesn't count.

```circuit.layout fragment
net_class usb = { diff_impedance = 90, return_gap_max = 1 mm }
rule debug_uart = { select = ":net(UART_TX)", return_path = true }
rule rf_out = { select = ":net(RF_OUT)", return_path = false }
```

`return_path = true` checks a net with no impedance target; `return_path = false` turns the check off. To fix a finding, route the line over one continuous pour or move the split. Where a stitching capacitor across the split carries the return current, turn the check off for that net.

### Cross-sections for field-solved impedance

The closed forms hold to a few percent inside their stated ranges. For a field-solved value, the board study's `pcb` source carries a cross-section for each controlled line: each net or declared pair in an impedance class, per copper layer and width it is routed at. A cross-section holds the conductor width, the pair gap, the copper thickness, the class's `etch_factor` when written, the reference planes above and below, and every stack row between them (solder mask over an outer layer included) with its thickness, Dk and Df. Its region is `pcb/<board>/line/<net>/<layer>` (a pair: `pcb/<board>/line/<p>+<n>/<layer>`). A 2D electrostatic study solves Z0, Zdiff, delay and the per-length L and C, and the layer-stack view shows the solved value beside the closed form [not yet in the current release](pending:design.ecad.field-solved-impedance).

## In the assembly

A board is a component of the product: an assembly places it as a part with its 3D model, and each component on it is its own face, selected by `component:<designator>`, so a pick in 3D selects the part on the schematic and the board. A board placed in an enclosure checks its fit and clearance against the enclosure's parts [not yet in the current release](pending:design.ecad.enclosure), and board studies (thermal, IR drop and warpage, made with Board » Analysis » New study) read the board's stackup, copper, pads and component power through a study's `pcb` source [not yet in the current release](pending:design.ecad.board-studies).
