---
description: Drive RunMat MCAD designs from MATLAB-language code. Compute design values in .m functions, regenerate a part at chosen input values for one run, and sweep inputs from a script with cad.evaluate, cad.study and cad.check, under runmat test.
keywords: [scripting, sweeps, parameter sweep, design of experiments, MATLAB, .m script, cad.evaluate, cad.study, cad.check, cad.inputs, parfor, runmat test, design automation, what-if]
---
# Scripting and sweeps

MATLAB-language code works with a design in three ways:

- **Inside the design:** a value in a `.part` calls a `.m` function, and the part regenerates when the function's inputs change.
- **One run at chosen inputs:** a tool call regenerates a part or an assembly at input values for that run and leaves the file unchanged.
- **From a script:** a `.m` script sets inputs, regenerates the design, reads its mass and published values, runs a study on it, and checks the results against limits, in a loop or a `parfor` sweep.

In every case the design file stays the one description of the design. A run uses the same regeneration as the editor's Rebuild.

## Values from functions

A `let` or a feature parameter can call a function in the part's folder, `private/`, or a `+package` above it, with named arguments:

```part fragment
let base = camBaseRadius(lift = input.lift, rise = input.rise, max_pressure = input.max_pressure) as m
```

The Desktop and the browser run the function in the RunMat runtime and regenerate the part with its result. The Outline's Values folder lists each value, marks the computed ones, and links each to its function's file. See [MATLAB functions in design values](/docs/design-language/matlab-functions) for resolution, units, determinism and caching.

## One run at chosen inputs

`design.inspect` and `design.export` take `set`, a map from input name to value text. The part or assembly regenerates with those values for that call only:

```json
{ "path": "arm.part", "set": { "thickness": "3.5 mm" }, "sections": ["mass"] }
```

- An unknown name is an `unknown_input` error that lists the inputs the design declares.
- A value outside the input's `minimum` and `maximum` is an error at the input, never a silent clamp.
- `density` gives `design.inspect` a density in kg/m³ for mass properties of bodies with no material.

The agent uses runs to answer what-if questions ("how heavy is the arm at 3.5 mm?") without proposing an edit, and `design.export` with `set` writes geometry at chosen inputs, such as a STEP file for a study.

## Sweeps from a script

The `cad` functions call the same tools from MATLAB-language code [not yet in the current release](pending:design.tools.cad-matlab-api).

| Function | Does |
|---|---|
| `r = cad.evaluate(path, Set=…, Configuration=…, Density=…)` | Regenerates a `.part` or `.assembly` with `Set` applied and returns a result struct. |
| `s = cad.study(path, Set=…)` | Runs a `.fea` study, whose own inputs flow to the part it names, and returns its checks and fields. |
| `cad.check(r, Name=Limit…)` | True when the run regenerated with no error and every limit holds. With no output it throws `cad:limitFailed`, naming each failed limit with its value. |
| `cad.inputs(path)` | A table of the design's inputs: name, kind, unit, default, minimum, maximum and the configurations that override each. |

`Set` takes a struct or name/value pairs. A value is a quantity string (`"3.5 mm"`) or a number in SI units (`0.0035`).

The result of `cad.evaluate` holds the run's values in SI units:

| Field | Holds |
|---|---|
| `r.inputs` | every input's value for this run |
| `r.mass`, `r.volume`, `r.area` | kg, m³, m² |
| `r.cg`, `r.inertia` | the center of gravity (1×3, m) and the inertia about it (3×3, kg·m²) |
| `r.bbox` | the bounding box's min and max corners (2×3, m) |
| `r.published` | the design's published plain values |
| `r.parts` | assemblies: a table of occurrence, design, mass and center of gravity |
| `r.diagnostics`, `r.ok` | the run's diagnostics, and whether it regenerated with no error |

### Mass against thickness

```matlab
t = (2.5:0.25:4) * 1e-3;            % m
m = zeros(size(t));
parfor k = 1:numel(t)
    r = cad.evaluate("mechanical/arm/arm.part", Set={"thickness", t(k)});
    m(k) = r.mass;
end
plot(t * 1e3, m * 1e3, "o-"); xlabel("Thickness (mm)"); ylabel("Mass (g)");
yline(18, "--", "limit");
```

Each `parfor` iteration regenerates on its own worker. A point with the same sources and inputs as an earlier run comes from the regeneration cache.

### A study at each point

`cad.study` runs a study whose geometry block names the part by symbol and binds its inputs, so the part regenerates at each point with no export step [not yet in the current release](pending:design.fea.part-geometry):

```matlab
t = [2.5 3 3.5] * 1e-3;
f1 = zeros(size(t));
parfor k = 1:numel(t)
    s = cad.study("simulation/arm_modal.fea", Set={"arm_thickness", t(k)});
    f1(k) = s.checks{"first_flap", "value"};
end
```

### Limits under `runmat test`

A test function checks a design against its limits. `runmat test` reports a failed `cad.check` as a test failure with the limit and the value:

```matlab
% tests/test_arm.m
function test_arm
    r = cad.evaluate("mechanical/arm/arm.part");
    cad.check(r, Mass=[0 0.018]);      % fails above 18 g
    s = cad.study("simulation/arm_modal.fea");
    cad.check(s);                      % every study check passes
end
```

## Next

- [MATLAB functions in design values](/docs/design-language/matlab-functions) covers values computed in `.m` code.
- [Agent tools and the command line](/docs/design-language/agent-and-cli) lists every design tool and its inputs.
