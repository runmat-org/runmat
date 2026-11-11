---
description: Run a circuit simulation in RunMat ECAD. Add a simulation section to a schematic, drive it with sources, run transient, AC and Monte Carlo analyses, measure bandwidth, rise time and settled value against limits, and run the same checks in CI.
keywords: [circuit simulation tutorial, SPICE simulation, RC filter, transient analysis, AC analysis, Monte Carlo, tolerance analysis, bandwidth, rise time, measures, simulation in CI]
---
# Run a circuit simulation

This guide checks an RC low-pass filter ahead of an ADC input: 10 kΩ and 10 nF. The simulation lives in the schematic as a `simulation` section, so the circuit and the checks on it change together, and every check that runs the circuit runs the simulation too.

The filter's corner frequency and its 10–90 % rise time are

$$
f_c = \frac{1}{2\pi R C} \approx 1.59\ \text{kHz}, \qquad t_r = 2.2\,R C = 0.22\ \text{ms}.
$$

## 1. The circuit and its simulation

```circuit file=adc_filter.circuit
runmat 0.7
std 1

// An RC low-pass filter ahead of an ADC input: 10 kOhm and 10 nF, a 1.59 kHz corner.
circuit adc_filter {
    root = main

    sheet main {
        component R1 = { part = "R0402-10K" }
        component C1 = { part = "C0402-10N" }
        component P1 = { part = "PAD-WIRE-20AWG" }
        component P2 = { part = "PAD-WIRE-20AWG" }
        component P3 = { part = "PAD-WIRE-20AWG" }
        net GND = { global = true, power_flag = true }
        connect = [net.IN, "P1.1", "R1.1"]
        connect = [net.OUT, "R1.2", "C1.1", "P3.1"]
        connect = [net.GND, "C1.2", "P2.1"]
    }

    simulation response {
        sheet = main
        source VS = { p = net.IN, ac = 1 V, pulse = { from = 0 V, to = 1 V, delay = 0.1 ms, rise = 1 us, fall = 1 us, width = 2 ms, period = 4 ms } }
        tolerance resistors = { match = "R*", value = 1 % }
        tolerance capacitors = { match = "C*", value = 10 % }
        analysis step = tran { stop = 1.5 ms, step = 1 us }
        analysis sweep = ac { from = 10 Hz, to = 1 MHz, points = 50 }
        analysis spread = monte_carlo { of = sweep, runs = 100, seed = 7 }
        probe = [net.IN, net.OUT]
        measure corner = { of = sweep, bandwidth = net.OUT, spec = { min = 1.4 kHz, max = 1.8 kHz } }
        measure rise = { of = step, rise_time = net.OUT, from = 0.1 ms, to = 1.5 ms, spec = { max = 0.3 ms } }
        measure settled = { of = step, final = net.OUT, spec = { min = 0.99 V, max = 1.01 V } }
    }
}
```

## 2. Sources

`source VS` drives net `IN` against ground. One source can carry several waveforms, and each analysis uses the one it needs: `ac = 1 V` is the small-signal amplitude for the AC sweep, and `pulse` is the step for the transient. Other waveforms are `dc`, `sin` and `pwl`; `current = true` makes it a current source.

## 3. Analyses

- `tran { stop, step }` runs the time response for 1.5 ms.
- `ac { from, to, points }` sweeps 10 Hz to 1 MHz.
- `monte_carlo { of = sweep, runs = 100, seed = 7 }` repeats the AC sweep 100 times with R and C drawn from their `tolerance` lines (1 % and 10 %). The fixed seed makes the runs repeat exactly.

Operating point, DC sweep, noise and temperature analyses take the same form. See [Simulation sections and measures](/docs/ecad/circuit/simulation#analyses).

## 4. Measures and limits

A measure reads one number from an analysis, and `spec` sets its limits:

- `corner` is the -3 dB bandwidth at `OUT`, within 1.4 kHz to 1.8 kHz.
- `rise` is the 10–90 % rise time after the step, at most 0.3 ms.
- `settled` is the final value, within 1 % of the 1 V step.

## 5. Run it

The Desktop's Simulate runs the section and plots the probed waveforms (`probe = [net.IN, net.OUT]`), with each measure and its verdict beside them. The agent runs it with `design.simulate`. From the command line, the check runs every section:

```sh
runmat-design check adc_filter.circuit
```

```text
adc_filter.circuit: measure response.corner = 1591.474140 Hz (pass)
adc_filter.circuit: measure response.rise = 0.000220 s (pass)
adc_filter.circuit: measure response.settled = 0.999999 V (pass)
adc_filter.circuit: passed (0 errors, 0 warnings; ran compile, erc, review, simulation:response)
```

A measure outside its limits fails the check, and the command exits with status 1, so a CI job that runs it stops a change that moves the corner out of band. `runmat check` and `runmat test` run the same circuit checks beside `.m` and `.fea` files.

## 6. Models

Passive parts simulate from their values. A part with a SPICE model in the library (its `spice` field) simulates with no extra lines. A section's `model "<part>" = { library, subckt, pins }` statement overrides it, so one section can run a switching regulator's full model and another its averaged model. See [The parts library](/docs/ecad/parts-library#part-declarations).

## Next

- [Simulation sections and measures](/docs/ecad/circuit/simulation) lists every source, analysis and measure.
- [Make fabrication outputs](/docs/ecad/guides/make-fabrication-outputs) when the design passes.
