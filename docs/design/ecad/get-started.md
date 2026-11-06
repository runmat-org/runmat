---
description: Start designing electronics in RunMat. Write a schematic as a .circuit file, check it, start its board, simulate a filter and write fabrication outputs, in the Desktop, with the agent or from the command line.
keywords: [get started, RunMat ECAD, first schematic, PCB tutorial, code-first schematic, circuit simulation, fabrication outputs]
---
# Get started with ECAD

A RunMat ECAD design is a schematic in a `.circuit` file and its board in a `.circuit.layout` file beside it. The Desktop shows the schematic, the board and the 3D board as tabs of one design; the agent and the command line read and write the same text.

## 1. A schematic

Create `sensor.circuit`. Components name library parts by MPN, and `connect` lists join pins into nets:

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

- `net_class power` sets the track width and clearance the board uses for power nets.
- `variant lite` builds the board without the LED.
- `simulation filter` checks the RC filter's corner and step response against limits.

## 2. Check it

```sh
runmat-design check sensor.circuit
```

The check runs the electrical rules, the design-review suites and every simulation's measures, prints each finding, and exits with status 1 on an error. See [Rules and checks](/docs/ecad/files#rules-and-checks).

## 3. Start the board

Create `sensor.circuit.layout` with its outline and the fabricator's limits:

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

A check of the new board reports the schematic changes it has not taken yet (`board_out_of_sync`); the board tab's Update board takes them, adding the components as footprints to place. Then routing, pours and vias are added on the canvas or by the agent. DRC runs as the board changes. See [Board layout](/docs/ecad/layout).

## 4. Fabrication outputs

```sh
runmat-design export sensor.circuit.layout --format fabrication --out fab
```

This writes Gerber, drill, the fabrication drawing, IPC-2581, ODB++, the IPC-D-356A netlist, and the BOM and pick-and-place file of each variant. See [Fabrication outputs and netlists](/docs/ecad/fabrication).

## Next

- [ECAD file reference](/docs/ecad/files): every statement in `.circuit` files.
- [Simulation sections and measures](/docs/ecad/circuit/simulation): analyses, measures and limits.
- [The parts library](/docs/ecad/parts-library): parts, declarations and models.
