---
description: Reference for simulation sections in a RunMat circuit. Sources, models, tolerances, operating point, DC, AC, transient, noise, temperature and Monte Carlo analyses, probes, and measures with limits that a check enforces.
keywords: [circuit simulation, SPICE, transient analysis, AC analysis, Monte Carlo, tolerance analysis, measures, bandwidth, rise time, settling time, noise analysis, simulation in CI]
---
# Simulation sections and measures

A `simulation <name> { … }` section in a `.circuit` file runs one sheet of the schematic as written: the same components, values and tolerances, with sources and analyses added. Each `measure` reads a number from an analysis and compares it with its `spec`, so a check fails when the circuit stops meeting it. The section is part of the design and is reviewed with it.

This section checks the 10 kΩ / 100 nF filter of the [sensor circuit](/docs/ecad/files#a-complete-circuit): its corner at $f_c = 1/(2\pi R C) \approx 159\ \text{Hz}$, its 10–90 % rise time of $2.2\,RC \approx 2.2\ \text{ms}$, and its settled level. The `tolerance` lines give the spread a `monte_carlo` analysis samples.

```circuit fragment
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
```

## Statements

| Statement | Declares |
|---|---|
| `sheet = <sheet>` | the sheet the section runs |
| `source <name> = { p, n, dc, ac, pulse, sin, pwl, current }` | a voltage source between nets `p` and `n` (ground when `n` is left out), or a current source with `current = true` |
| `model "<part>" = { library, subckt, pins }` | a SPICE model for a part, by designator or MPN; a section's model wins over the part's own |
| `tolerance <name> = { match, value, distribution }` | the spread of matching components (`match = "R*"`), `uniform` or `gauss` |
| `analysis <name> = <kind> { … }` | an analysis (below) |
| `probe = [<nets and pins>]` | the waveforms to keep |
| `measure <name> = { of, <kind> = <net or pin>, from, to, spec }` | a number from an analysis, with limits |
| `option <name> = <value>` | a simulator option |

A net in a source, probe or measure is `net.X` (its voltage); `"R1.2"` is the current into that pin.

## Analyses

| Kind | Parameters |
|---|---|
| `op` | the operating point |
| `dc { … }` | a source swept over values |
| `ac { from, to, points }` | small-signal frequency response |
| `tran { stop, step, max_step }` | time response |
| `noise { output, input, from, to, points }` | output and input-referred noise |
| `temperature { values }` | another analysis repeated over temperatures |
| `monte_carlo { of, runs, seed }` | another analysis repeated with values drawn from the tolerances; a fixed `seed` makes the runs repeat exactly |

## Measures

| Kind | Reads |
|---|---|
| `at`, `max`, `min`, `peak_to_peak`, `average`, `rms`, `final` | a waveform's value at a time, its extremes, mean, RMS or last value |
| `rise_time`, `settling`, `overshoot` | step response: 10–90 % rise, settling into a `band`, overshoot |
| `bandwidth`, `peak_frequency`, `q`, `unity_gain`, `phase_margin`, `gain_at`, `phase_at` | frequency response |
| `output_noise`, `input_noise`, `noise_density` | noise |

`spec = { min, max }` sets the limits. A check reports each measure's value and whether it passes:

```text
sensor.circuit: measure filter.corner = 159.147414 Hz (pass)
sensor.circuit: measure filter.rise = 0.002196 s (pass)
sensor.circuit: measure filter.settled = 0.999877 V (pass)
```

## Running

`design.simulate` runs a section from the agent, and `runmat-design check` runs every section and fails on a measure outside its spec. Part models come from the part (its `spice` field) or from the section's `model` statements; SPICE libraries in other common dialects read in, with their sweep and measure lines reported as warnings, since the section states its own analyses and measures.
