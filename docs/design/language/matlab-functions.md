---
description: A RunMat design value can call a MATLAB function in the same folder. Resolution, arguments and units across the boundary, determinism, volatile values and caching, with a disc cam whose profile comes from its motion law.
keywords: [MATLAB, .m functions, design values, parametric CAD, cam profile, motion law, code-driven design, volatile, cache, determinism]
---
# MATLAB functions in design values

A value in a design file can be the result of a MATLAB function. The function lives in a `.m` file beside the design, and the design calls it by name with named arguments. When an input changes, RunMat calls the function again and regenerates what depends on it. The function is ordinary MATLAB code: it can call other functions in its folder, use `private/` helpers, and be tested on its own.

## A cam from its motion law

This disc cam's profile comes from a dwell-rise-dwell-return motion law. `camBaseRadius.m` sizes the base circle so the pressure angle stays under its limit, and `camRadius.m` gives the pitch radius at each 30° station. The part sketches a closed spline through the stations.

```matlab file=camBaseRadius.m
function rb = camBaseRadius(lift, rise, max_pressure)
% Smallest base-circle radius that keeps a cycloidal rise under the pressure-angle limit.
% A cycloidal rise of `lift` over `rise` radians peaks at ds/dtheta = 2*lift/rise, at
% mid-rise (s = lift/2). Inputs and result are SI (m, rad).
rb = (2 * lift / rise) / tan(max_pressure) - lift / 2;
end
```

```matlab file=camRadius.m
function r = camRadius(theta, base, lift, rise, dwell)
% Pitch radius of a dwell-rise-dwell-return disc cam at cam angle theta (SI: m, rad).
% Cycloidal rise over `rise`, a dwell of `dwell` at full lift, a cycloidal return over
% `rise`, then a dwell at the base circle.
s = 0;
if theta < rise
    s = cycloidal(theta / rise) * lift;
elseif theta < rise + dwell
    s = lift;
elseif theta < 2 * rise + dwell
    s = (1 - cycloidal((theta - rise - dwell) / rise)) * lift;
end
r = base + s;
end
```

```matlab file=cycloidal.m
function y = cycloidal(u)
% Normalized cycloidal motion law: 0 at u = 0, 1 at u = 1, zero velocity and acceleration at both ends.
y = u - sin(2 * pi * u) / (2 * pi);
end
```

```part m file=disc_cam.part
runmat 0.7
std 1

// A disc cam whose profile comes from its motion law, computed in MATLAB code: camBaseRadius.m
// sizes the base circle for the pressure-angle limit, and camRadius.m gives the pitch radius at
// each 30 deg station (cycloidal rise, dwell, cycloidal return). Change an input and the
// profile regenerates from the functions.
part disc_cam {
    input lift : length = 8 mm
    input rise : angle = 150 deg
    input dwell : angle = 30 deg
    input max_pressure : angle = 30 deg
    input thickness : length = 10 mm

    let base = camBaseRadius(lift = input.lift, rise = input.rise, max_pressure = input.max_pressure) as m
    let r0 = camRadius(theta = 0 deg, base = base, lift = input.lift, rise = input.rise, dwell = input.dwell) as m
    let r30 = camRadius(theta = 30 deg, base = base, lift = input.lift, rise = input.rise, dwell = input.dwell) as m
    let r60 = camRadius(theta = 60 deg, base = base, lift = input.lift, rise = input.rise, dwell = input.dwell) as m
    let r90 = camRadius(theta = 90 deg, base = base, lift = input.lift, rise = input.rise, dwell = input.dwell) as m
    let r120 = camRadius(theta = 120 deg, base = base, lift = input.lift, rise = input.rise, dwell = input.dwell) as m
    let r150 = camRadius(theta = 150 deg, base = base, lift = input.lift, rise = input.rise, dwell = input.dwell) as m
    let r180 = camRadius(theta = 180 deg, base = base, lift = input.lift, rise = input.rise, dwell = input.dwell) as m
    let r210 = camRadius(theta = 210 deg, base = base, lift = input.lift, rise = input.rise, dwell = input.dwell) as m
    let r240 = camRadius(theta = 240 deg, base = base, lift = input.lift, rise = input.rise, dwell = input.dwell) as m
    let r270 = camRadius(theta = 270 deg, base = base, lift = input.lift, rise = input.rise, dwell = input.dwell) as m
    let r300 = camRadius(theta = 300 deg, base = base, lift = input.lift, rise = input.rise, dwell = input.dwell) as m
    let r330 = camRadius(theta = 330 deg, base = base, lift = input.lift, rise = input.rise, dwell = input.dwell) as m

    sketch profile on origin.xy {
        // Station points at r(theta) * [cos theta, sin theta].
        lobe = spline(points = [[r0, 0 mm], [r30 * 0.866025, r30 * 0.5], [r60 * 0.5, r60 * 0.866025], [0 mm, r90], [r120 * -0.5, r120 * 0.866025], [r150 * -0.866025, r150 * 0.5], [-r180, 0 mm], [r210 * -0.866025, r210 * -0.5], [r240 * -0.5, r240 * -0.866025], [0 mm, -r270], [r300 * 0.5, r300 * -0.866025], [r330 * 0.866025, r330 * -0.5]], closed)
    }
    feature disc = extrude(profile = sketch.profile, distance = input.thickness)
}
```

Change `lift` from 8 mm to 10 mm and the base circle grows to keep the pressure angle at 30°, and every station radius follows.

## Calling a function

| Form | Meaning |
|---|---|
| `camBaseRadius(lift = input.lift, rise = 150 deg, max_pressure = 30 deg) as m` | a same-folder function with named arguments |
| `camRadius(0 deg, base, input.lift, input.rise, input.dwell) as m` | positional arguments, in the order of the `function` line |
| `cams.lift(2 mm) as m` | `+cams/lift.m` in this folder or a folder above it |
| `lib.offset(input.width) as m` | a function of a folder brought in with `use "./lib" as lib` |
| `volatile liveWind() as m` | a function with side effects, marked at the call |

- **Resolution.** A bare name is a function in the design's folder (its `private/` folder included), then a function on the project's MATLAB path. `pkg.fn` is `+pkg/fn.m`, found in the referring folder or the nearest folder above it, as in MATLAB. A name the folder does not define is an error at the call.
- **Arguments.** Named arguments map to the names on the `function` line; positional arguments fill the remaining parameters in order. An unknown name is an error that lists what the function takes: *takes (lift, rise, max_pressure), not `pitch`*.
- **Units.** Arguments cross into MATLAB as SI numbers: `8 mm` arrives as `0.008`, `150 deg` as `2.618`. The function returns a plain number, so the call states the unit of the result with `as <unit>`. The unit is then written in the design file, where review and search see it. A result with no `as` is dimensionless, and using it as a length is an error.
- **One result.** A value takes one number. A function that returns a vector or matrix is an error at the call (*returned 2 numbers*).

## Determinism

A design must regenerate the same way every time, so a value function must be deterministic.

- **Effects need `volatile`.** A function whose code calls the clock (`tic`, `toc`, `clock`, `now`, `datetime`), reads or writes files (`fopen`, `load`, `save`, `readtable`, `readmatrix`), uses the network (`webread`, `websave`), reads the environment (`getenv`), asks for input, runs a shell command (`system`) or draws unseeded random numbers (`rand`, `randn`, `randi`, `randperm`) is an error at the call unless the call is marked `volatile`. Functions it calls in its folder are checked the same way.
- **The audit.** On first use, RunMat evaluates each deterministic call twice. Results that differ bit for bit fail as `design.value.nondeterministic`.
- **Volatile values** are evaluated when the design is regenerated, and a release records the value it used in `runmat.lock`, so the released design reproduces bit for bit [not yet in the current release](pending:design.language.volatile-lock).

## Caching

Each result is cached under a key made of the function file's digest, the digests of every folder function it reaches, the SI arguments bit for bit and the runtime version. Editing `cycloidal.m` re-runs every call that reaches it and nothing else. A failed function blocks only the values and features that depend on it; the rest of the design still regenerates.

## Quantities across the boundary

A function can also take and return values with units: an `arguments` block that declares `lift (1,1) quantity.Length` receives `8 mm` as a length, and a function that returns `quantity(0.012, "m")` needs no `as` at the call [not yet in the current release](pending:design.language.m-quantities).

## Circuit values

A `.circuit` component value or `param` can call a function the same way, with the same resolution through the circuit's folder, `private/` and `+package` folders, the same SI arguments and `as` unit, and the same refusal of functions with side effects. This filter's resistor comes from its corner frequency and its capacitor:

```matlab file=rcResistor.m
function r = rcResistor(fc, c)
% The resistor that puts an RC low-pass corner at fc (SI: Hz, F, ohm).
r = 1 / (2 * pi * fc * c);
end
```

```circuit file=lowpass.circuit
runmat 0.7
std 1

circuit lowpass {
    root = main
    sheet main {
        component R1 = { part = "R0402-10K", value = rcResistor(fc = 160 Hz, c = 100 nF) as ohm }
        component C1 = { part = "C0402-1U", value = 100 nF }
        component P1 = { part = "PAD-WIRE-20AWG" }
        component P2 = { part = "PAD-WIRE-20AWG" }
        component P3 = { part = "PAD-WIRE-20AWG" }
        net GND = { global = true, power_flag = true }
        connect = [net.IN, "P1.1", "R1.1"]
        connect = [net.OUT, "R1.2", "C1.1", "P3.1"]
        connect = [net.GND, "C1.2", "P2.1"]
    }
}
```

R1 is $1/(2\pi \cdot 160\ \text{Hz} \cdot 100\ \text{nF}) \approx 9.95\ \text{k}\Omega$. The circuit keeps the call, so the schematic, the BOM and the simulation read the computed value, and changing the corner recomputes it. While the runtime hasn't answered a call yet, the check reports the value as pending, with no error on what depends on it.
