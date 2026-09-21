# Loads, excitations, and natural boundaries

A load applies a known constant quantity directly. An excitation applies a named signal and keeps
the signal's domain, interpolation, callable identity, and data provenance separate from its target,
direction, distribution, and kinematics. Natural boundaries contribute traction or flux without
prescribing the primary unknown.

| Domain | Guide |
| --- | --- |
| Force, moment, traction, pressure, and body force | [Structural loads](./structural) |
| Heat generation, heat flux, ambient temperature, and convection | [Thermal loads](./thermal) |
| Voltage, current, charge, current density, vector potential, and insulation | [Electromagnetic loads](./electromagnetic) |
| Pressure, normal velocity, rigid/radiating/impedance boundaries | [Acoustic loads](./acoustic) |
| Inlet velocity, mass flow, wall motion, and outlet pressure | [Fluid loads](./fluid) |

## Find a load by modeling task

| I need to model… | Start here |
| --- | --- |
| A force distributed over a bracket, bolt hole, face, edge, or point | [Total force and distribution](./structural#how-do-i-apply-a-total-force-to-a-face-edge-or-point) |
| Torque, thrust, or a combined shaft wrench | [Moment and combined force/torque](./structural#how-do-i-apply-a-torque-or-moment) |
| Pressure vessel, wind, bearing, hydrostatic, or follower pressure | [Structural pressure](./structural#how-do-i-apply-pressure-to-a-structural-surface) |
| Gravity, self-weight, frame acceleration, centrifugal load, or imbalance | [Structural body and inertial loads](./structural#how-do-i-apply-gravity-or-another-body-force) |
| Impact, pulse, cyclic, harmonic, random, or base excitation | [Varying loads](#how-do-i-make-a-load-vary-with-time-or-frequency) |
| Electronics loss, battery heat, heater power, laser flux, or convection | [Thermal loads](./thermal) |
| Voltage source, terminal current, charge, coil drive, or current density | [Electromagnetic loads](./electromagnetic) |
| Loudspeaker, piston, pressure pulse, liner, open end, or rigid acoustic wall | [Acoustic loads](./acoustic) |
| Inlet velocity, mass flow, moving/rotating wall, suction, or outlet pressure | [Fluid loads](./fluid) |
| Loads transferred between thermal, fluid, structural, or electromagnetic solves | [Interfaces](../constraints/interfaces) |

The guides use searchable questions as headings and distinguish a directly supported declaration
from a required coupled procedure or a capability that is not yet first class. An unsupported model
is never represented by silently reusing a quantity with different dimensions.

## Constant and varying forms

```fea
load service_force {
    force on geometry.bracket.region.loaded_face = [0, -1, 0] kN
}
```

```fea
signal impact_history : force over time {
    samples = [
        { at = 0 ms, value = 0 N },
        { at = 2 ms, value = 1 kN },
        { at = 8 ms, value = 0 N },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation impact {
    force {
        target = geometry.bracket.region.loaded_face
        signal = signal.impact_history
        direction = global([0, -1, 0])
        kinematics = fixed_frame
    }
}
```

The checker verifies quantity, domain, shape, target topology, and frame compatibility. The runtime
presentation contract retains the resolved region, SI value and display unit, vector direction,
signal/function identity, source location, and step/case activation so the Desktop can render what
will actually run.

## Which load declaration should I use?

Choose the declaration by the physical quantity you know, not by which numeric value is easiest to
enter:

| What you know | RunMat declaration | What it preserves |
| --- | --- | --- |
| Total force on a structural region | `force` | Resultant force in N |
| Surface force density | `traction` | Local traction in Pa |
| Normal surface loading | `pressure` or `structural.pressure` | Pressure and resolved normal direction |
| Translational/rotational inertial source | `body_force` | Force per volume |
| Total electrical current | `electric_current` | Integrated current in A |
| Distributed electrical current | `current_density` | Current per area |
| Total fluid flow by mass | `mass_flow` | Integrated mass rate in kg/s |
| Fluid speed/profile | `inlet_velocity` or `fluid.velocity` | Boundary velocity in m/s |
| Total thermal power density | `heat_generation` | Power per volume |
| Surface thermal power density | `heat_flux` or `thermal.heat_flux` | Power per area |

Do not convert a resultant to an intensity without using the resolved CAD measure. A `1 kN` force
on a changing face remains 1 kN; a `1 kPa` traction changes its resultant with face area. The result
evidence records both the authored quantity and the resolved projection so this distinction remains
auditable after remeshing.

## How do I make a load vary with time or frequency?

Use a typed `signal` and an `excitation`. The signal declares the independent domain, samples or
RunMat function, interpolation, extrapolation, and physical quantity. The excitation declares where
and how that signal acts. A constant `load` is preferable when there is no varying domain.

```fea
signal shaker_force : force over frequency {
    samples = [
        { at = 10 Hz, value = 5 N },
        { at = 100 Hz, value = 20 N },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation shaker {
    force {
        target = geometry.fixture.region.drive_point
        signal = signal.shaker_force
        direction = global([0, 0, 1])
        kinematics = fixed_frame
    }
}
```

The analysis sampling domain must fall inside the signal domain unless extrapolation explicitly
permits otherwise. Frequency-domain amplitude and phase are part of the physical input; a real
magnitude must not be described as a complete complex spectrum when phase matters.

## How do I drive a load from a RunMat `.m` function?

Import the file by path, bind a symbol, and use that symbol as the signal function. The function is
type-checked against the signal's input domain and output quantity; it does not become a second,
loosely typed expression language inside `.fea`.

```fea
import function impactForce from "./impactForce.m"

signal impact : force over time {
    generated by function.impactForce {
        arguments {
            peak = 1200 N
            duration = 6 ms
        }
        sample from 0 s to 20 ms at 10 kHz
        interpolation = linear
        extrapolation = reject
        tolerance = 1e-6 N
    }
}
```

```matlab
function force = impactForce(time, peak, duration)
    force = peak .* sin(pi .* min(time, duration) ./ duration) .* (time <= duration);
end
```

The `.fea` declaration supplies units and admissible domain; the `.m` function supplies numerical
shape. Its module path and symbol identity remain in provenance and the Desktop annotation.

## How do load cases combine?

Name each load or excitation once, group compatible objects into explicit cases, and activate the
case in the analysis step. Do not duplicate the same physical load merely to include it in several
studies. Case combination is algebra on admitted physical inputs, not a text merge, and retains each
object's source identity for result attribution.

## How do I verify that a load reached the intended mesh entities?

Inspect admission and solve evidence for:

1. region resolution against persistent geometry identity;
2. target topology and covered measure;
3. projected entity count and coverage ratio;
4. integrated resultant or source power/current/flow;
5. the active case and analysis step; and
6. signal/function sampling and extrapolation decisions.

An empty or partial projection is an error unless the declaration explicitly permits that
cardinality. RunMat must not fall back from a named CAD region to ordinal entity indices.

## Why is a pressure, heat flux, or current pointing the wrong way?

Normal quantities inherit orientation from the resolved target. Review the geometry region's face
orientation and the convention documented by the physics family. Changing the sign to make a plot
look right can conceal an inverted region or primary/secondary pairing error; verify the integrated
signed resultant instead.

## Why is my model overconstrained after adding a load?

A natural load normally contributes to the right-hand side or boundary operator and does not fix the
primary unknown. Some pairs still conflict physically—for example, prescribing complete inlet
velocity and an incompatible mass flow, or prescribing terminal voltage and total current at the
same terminal. Treat the checker diagnostic as a modeling conflict, not a solver-tolerance problem.

## How do I apply the same load in several analyses?

Declare it once under the model, include it in named load cases, and activate the applicable case in
each analysis step. This keeps one stable source identity and lets results state exactly which case
ran. Use typed inputs or sweeps for amplitudes that change between cases; avoid copied declarations
whose values can drift.

## How do I parameterize a load for a sweep?

Bind its supported numeric field to a typed study input, then sweep that input. Units and admissible
ranges belong to the input declaration. The compiler expands cases without turning the physical
quantity into a string substitution, and the result key retains the input tuple.

## How do I apply loads to an assembly with several materials?

Loads target geometry regions; materials are assigned to solid/cell regions separately. A force on a
face does not select a material, and a material assignment does not create a load. Body forces and
coupled source laws that depend on density, conductivity, or another property must resolve the
property from each assigned material region.

## How do I tell whether a quantity is a resultant or an intensity?

Check its dimension and projection invariant. Newtons, amperes, coulombs, kilograms per second, and
newton-metres are resultants. Pascals, watts per square metre, watts per cubic metre, amperes per
square metre, and force per volume are intensities. Evidence should report both authored value and
integrated realized resultant where an integral is meaningful.

## What happens when CAD or mesh resolution changes?

Resultant-preserving loads keep their total while their nodal distribution changes. Intensity loads
keep their local magnitude, so their integrated total changes if selected area or volume changes.
Persistent region identity and projection evidence make that behavior reviewable; ordinal face or
element indices are not accepted as a durable substitute.
