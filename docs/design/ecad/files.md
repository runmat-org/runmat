---
description: Reference for RunMat ECAD files. Every statement a .circuit schematic and a .circuit.layout board can hold, sheets and hierarchy, repeated channels, components and library parts, nets, buses and net classes, variants, electrical rules and design-review suites, with examples that check.
keywords: [.circuit file, .circuit.layout file, schematic file format, PCB file format, text schematic, hierarchical schematic, net classes, variants, ERC, design review, code-first PCB]
---
# ECAD file reference

RunMat ECAD keeps a schematic in a `.circuit` file and its board in a `.circuit.layout` file beside it. The schematic holds the design intent: components, connections, net classes, variants and simulations. The board holds where everything goes: outline, stackup, footprints, copper and rules. This page covers every `.circuit` statement and the board's top level; [Board layout](/docs/ecad/layout) covers the board in full.

## A complete circuit

A 3.3 V regulator with its capacitors, a power LED, and an RC filter ahead of an ADC input, with a variant that leaves the LED off and a simulation of the filter:

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

Its board, before placement:

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

| Statement | Declares | Section |
|---|---|---|
| `root = <sheet>` | the top sheet | [Sheets](/docs/ecad/circuit/sheets-and-channels) |
| `param <name> = <value>` | a design parameter, such as `param supply = 3.3 V ± 5 %` | [Components](/docs/ecad/circuit/components-and-parts) |
| `net_class <name> = { … }` | rules shared by a class of nets | [Nets](/docs/ecad/circuit/nets-and-classes) |
| `interface <name> = { members, diff_pairs }` | a bundle type for harnesses | [Nets](/docs/ecad/circuit/nets-and-classes) |
| `variant <name> = { base, overrides }` | a build variant | [Variants](/docs/ecad/circuit/variants) |
| `part "<mpn>" = { … }` | a library part embedded in the file | [Components](/docs/ecad/circuit/components-and-parts) |
| `sheet <name> { … }` | a schematic sheet | [Sheets](/docs/ecad/circuit/sheets-and-channels) |
| `simulation <name> { … }` | a simulation section | [Simulation sections and measures](/docs/ecad/circuit/simulation) |

Inside a sheet:

| Statement | Declares |
|---|---|
| `port <name> : <direction>` | a sheet port: `in`, `out`, `inout`, `passive` or `power`; `port D : in = bus [D0, D1]` for a bus port |
| `component <designator> = { part, value, classes, properties, variants, power, renamed_from }` | a component |
| `net <name> = { class, global, power_flag, renamed_from }` | a net and its settings; a net first named in a `connect` needs no declaration |
| `bus <name> = [<members>]` | a bus |
| `harness <name> = <interface>` | a harness of an interface type |
| `instance <name> = { sheet, repeat, bind }` | a sheet placed as a block, or repeated as channels |
| `connect = [<endpoints>]` | joins every endpoint listed into one net |
| `no_erc = [<pins>]` | pins the electrical rules leave alone |
| `diff_pair <name> = { p, n }` | a differential pair |
| `text <name> = "<words>"` | a note drawn on the sheet, such as `text note1 = "5 V in from J1"`; it has no electrical meaning |

Endpoints are `U1.VIN`, `"C1.1"`, `net.GND`, `"net.+3V3"`, `port.IN`, `bus.D.D0`, `harness.USB.DP` and `instance.pwr.VIN`; see [Names and selectors](/docs/design-language/names-and-selectors#pins-nets-and-designators).

## Reference pages

| Page | Covers |
|---|---|
| [Sheets, hierarchy and channels](/docs/ecad/circuit/sheets-and-channels) | the root sheet, ports, blocks, channels repeated with `repeat` |
| [Components and library parts](/docs/ecad/circuit/components-and-parts) | components, values and tolerances, values from `.m` functions, designators, parts declared in the file |
| [Nets and net classes](/docs/ecad/circuit/nets-and-classes) | `connect` and endpoints, net settings, buses, harnesses, differential pairs, net classes |
| [Variants](/docs/ecad/circuit/variants) | fitted and not-fitted parts, alternates, variants on variants |
| [Rules and checks](/docs/ecad/circuit/rules-and-checks) | the electrical rules, the design-review suites, the board checks, the command line |
| [Simulation sections and measures](/docs/ecad/circuit/simulation) | sources, models, analyses, measures and limits |
| [Board layout](/docs/ecad/layout) | every `.circuit.layout` statement and the design rules |
| [Fabrication outputs and netlists](/docs/ecad/fabrication) | the fabrication set and netlists |
