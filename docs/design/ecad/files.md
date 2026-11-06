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
| `root = <sheet>` | the top sheet | [Sheets](#sheets-hierarchy-and-channels) |
| `param <name> = <value>` | a design parameter, such as `param supply = 3.3 V ± 5 %` | [Components](#components-and-parts) |
| `net_class <name> = { … }` | rules shared by a class of nets | [Nets](#nets-buses-and-net-classes) |
| `interface <name> = { members, diff_pairs }` | a bundle type for harnesses | [Nets](#nets-buses-and-net-classes) |
| `variant <name> = { base, overrides }` | a build variant | [Variants](#variants) |
| `part "<mpn>" = { … }` | a library part embedded in the file | [Components](#components-and-parts) |
| `sheet <name> { … }` | a schematic sheet | [Sheets](#sheets-hierarchy-and-channels) |
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

Endpoints are `U1.VIN`, `"C1.1"`, `net.GND`, `"net.+3V3"`, `port.IN`, `bus.D.D0`, `harness.USB.DP` and `instance.pwr.VIN`; see [Names and selectors](/docs/design-language/names-and-selectors#pins-nets-and-designators).

## Sheets, hierarchy and channels

A sheet is a page of the schematic, and a sheet can be placed in another as a block with `instance`. `repeat = N` places it N times as channels: each copy gets its own designators and its own nets (`/ch[0]/FBK`, `/ch[1]/FBK`), and `bind` maps the sheet's ports onto the parent's nets or onto consecutive members of a bus.

```circuit fragment
sheet channel {
    port IN : in
    port OUT : out
    component U = { part = OPA4171 }
    component RF = { part = "R0402-10K" }
    connect = [port.IN, net.IN, "U.A.IN+"]
    connect = [net.FBK, "U.A.IN-", "RF.1"]
    connect = [port.OUT, net.OUT, U.A.OUT, "RF.2"]
}

sheet main {
    bus AIN = [AIN0, AIN1, AIN2]
    bus ADC = [ADC0, ADC1, ADC2]
    instance ch = { sheet = channel, repeat = 3, bind = { IN = { bus = AIN, offset = 0 }, OUT = { bus = ADC, offset = 0 } } }
}
```

Channel `k` binds `IN` to `AIN<k>` and `OUT` to `ADC<k>`. The board can place each channel in its own room, so a layout made for one channel can be copied to the rest.

## Components and parts

`component <designator> = { part = "<mpn>" }` places a part. The part comes from the parts library or from a `part "<mpn>" = { … }` declaration in the file; see [The parts library](/docs/ecad/parts-library).

| Field | Meaning |
|---|---|
| `part` | the library part, by MPN |
| `value` | the component's value with its tolerance: `value = 10 kohm ± 1 %` |
| `classes` | component classes, for board rules and rooms |
| `properties` | typed properties |
| `variants` | how the component is fitted in each variant |
| `power` | the dissipation the board's thermal map uses, in W |
| `renamed_from` | the old designator after a rename, so the board keeps the placement |

A designator ending in `?` (`R?`) is numbered automatically.

## Nets, buses and net classes

A net joins pins. `global = true` joins nets of the same name across every sheet, as power rails and ground do; `power_flag = true` marks a net as driven by a source outside the schematic, such as a connector. A net class carries the rules its nets share:

```circuit fragment
net_class power = { track_width = "0.40 mm", clearance = "0.20 mm" }
net_class analog = { clearance = "0.20 mm" }
interface usb2 = { members = [DP, DN], diff_pairs = [[DP, DN]] }
```

The board reads each class's track width and clearance, and DRC checks them.

## Variants

A variant changes how components are fitted: `variant lite = { overrides = { D1 = not_fitted } }`. An override is `fitted`, `not_fitted`, or an alternate part or value. `base = <variant>` builds on another variant, and a component inside a channel is named by its path (`"/ch[0]/CD2"`). ERC, the BOM and the assembly outputs follow the variant, so a not-fitted part neither raises ERC findings nor appears in that variant's pick-and-place file.

## Rules and checks

A circuit check elaborates the design and runs the electrical rules:

| Code | Severity | Finds |
|---|---|---|
| `pin-conflict` | from the pin-type matrix | two pin types on one net that may not meet |
| `multiple-drivers` | error | more than one push-pull driver on a net |
| `power-not-driven` | error | power inputs on a net with no power output and no power flag |
| `input-not-driven` | warning | inputs on a net nothing drives |
| `pin-unconnected` | warning | a pin on no net, not no-connect and not in `no_erc` |
| `single-pin-net` | warning | a net with one pin |
| `no-connect-connected` | error | a no-connect pin on a net with other pins |

Design-review suites check circuits the way a reviewer would, each finding with a typed fix:

| Suite | Checks |
|---|---|
| `decoupling` | every IC power pin has a capacitor from its rail to ground |
| `i2c` | each SCL and SDA net has a pull-up of 1 kΩ to 10 kΩ |
| `crystal` | each crystal pin has a load capacitor, and the load matches the crystal within 10 % |
| `straps` | reset and boot pins are pulled to a defined level |
| `usb-c` | a USB-C sink has 5.1 kΩ pull-downs on CC1 and CC2, and D+ and D− are paired |
| `switcher` | a switching regulator has its inductor, capacitors, and a feedback divider within 3 % of the rail |

A board check adds DRC (see [Board layout](/docs/ecad/layout#design-rules)), unrouted connections and schematic changes the board has not taken. `runmat-design check` runs all of them and every simulation's measures; see [Checks and diagnostics](/docs/design-language/checks-and-diagnostics#checking-from-the-command-line).
