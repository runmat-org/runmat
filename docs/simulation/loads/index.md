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

## Complete load and excitation declaration catalog

The following names are the closed load/excitation action set accepted by the `.fea` checker. The
quantity column is part of the type contract: a signal with a different quantity cannot be attached
even when both happen to be represented by one number.

| Action | Signal quantity | Valid geometry target | Default projection | Typical use |
| --- | --- | --- | --- | --- |
| `force` | force | vertex, edge, or face | preserve resultant | Applied mechanical force |
| `moment` | moment | vertex, edge, or face | preserve resultant and moment | Torque or couple |
| `traction` | traction | face | intensity per area | Vector surface traction |
| `pressure` | pressure | face | intensity per area | Normal structural pressure |
| `body_force` | body-force density | solid or whole model | intensity per volume | Gravity-equivalent or inertial source |
| `heat_generation` | heat-generation density | solid or whole model | intensity per volume | Internal thermal power density |
| `heat_flux` | heat flux | face | intensity per area | Surface heating or cooling |
| `temperature_field` | temperature | solid | uniform kinematic or topology field | Prescribed volume temperature input |
| `ambient_temperature` | temperature | face | uniform kinematic or topology field | Varying convection ambient |
| `terminal_voltage` | electric potential | face | uniform kinematic or topology field | Voltage-driven terminal |
| `electric_current` | electric current | face | preserve resultant | Total terminal current |
| `electric_charge` | electric charge | face | preserve resultant | Total boundary charge |
| `current_density` | electric-current density | face | intensity per area or topology field | Distributed current injection |
| `magnetic_vector_potential` | magnetic vector potential | face | uniform kinematic or topology field | Magnetic boundary drive |
| `acoustic_pressure` | pressure | face | uniform kinematic or topology field | Acoustic pressure source |
| `incident_pressure` | pressure | planar exterior face | uniform kinematic | Normally incoming pressure; matching outgoing operator included |
| `normal_velocity` | velocity | face | uniform kinematic or topology field | Acoustic piston or normal wall velocity |
| `inlet_velocity` | velocity | face | uniform kinematic or topology field | Fluid inlet profile |
| `mass_flow` | mass-flow rate | face | preserve resultant | Total inlet or outlet mass flow |
| `wall_motion` | velocity | face | uniform kinematic or topology field | Prescribed fluid-wall motion |

`force`, `moment`, `electric_current`, `electric_charge`, and `mass_flow` preserve an integrated
resultant when the mesh changes. Intensities preserve their local value. `force` and `moment` may
also use concentrated projection. `force`, `traction`, and `pressure` may use follower kinematics;
the other actions are fixed-frame. These rules are checked rather than inferred by a solver.

Natural boundary operators use a separate closed set because they assemble an operator rather than
a sampled excitation: `structural.pressure`, `thermal.heat_flux`, `thermal.convection`,
`magnetic.insulation`, `acoustic.rigid_wall`, `acoustic.radiation`, `acoustic.impedance`, and
`fluid.outlet_pressure`. The family guides show their exact fields and sign conventions.

## Which load families can I execute today?

The source checker understands every declaration in the catalog above, but a checked declaration is
not by itself an executable end-to-end study. The native whole-study executor currently admits
structural procedure graphs. Thermal, electromagnetic, acoustic, and fluid load contracts can be
authored and checked, but their complete geometry-to-publication `.fea` execution paths are not yet
admitted. `runmat check` therefore answers “is this a valid typed study definition?”; it does
not admit a physics family for execution.

The family pages mark that boundary where it matters. See [FEA capability status](../status) for the
current execution envelope and [verification and validation](../validation) for the evidence needed
before a family is promoted. Do not replace a rejected nonstructural procedure with a structural
load that happens to share the same unit.

## How do I choose between a constant load and an excitation?

Use `load` when the action has one value for the analysis case. Use `signal` plus `excitation` when
the value varies over time, frequency, mode, or another admitted independent domain. Do not wrap a
constant in a two-point signal merely to make it reusable; name the load and include it in more than
one load case instead.

```fea
load dead_weight {
    body_force on whole_model = [0, 0, -77.0] kN/m^3
}

signal operating_force : force over time {
    samples = [
        { at = 0 s, value = 0 N },
        { at = 1 s, value = 3 kN },
        { at = 5 s, value = 3 kN },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation operating_cycle {
    force {
        target = geometry.machine.region.load_pad
        signal = signal.operating_force
        direction = global([0, -1, 0])
        kinematics = fixed_frame
    }
}
```

The direct load and the varying excitation remain separate named objects. A load case can include
both without losing which contribution supplied each resultant.

## How do I apply a spatially varying field instead of a uniform value?

Use a topology-field distribution only for actions whose typed contract admits it. The sampled
field must resolve onto the target topology with the declared quantity and frame. This is used for
measured pressure maps, imported heat flux, CFD-to-structure traction, inlet profiles, and similar
data. Preserve the source artifact, coordinate transform, interpolation rule, uncovered-region
policy, and projection error in evidence. A list of values with no topology identity is not a field.

## How do I apply a follower load in a geometrically nonlinear analysis?

Follower kinematics are supported only for `force`, `traction`, and `pressure`. Their direction is
updated from the deformed configuration according to the admitted formulation. Moments, body
forces, thermal sources, electrical sources, and flow inputs cannot be labeled follower to bypass
their own physical contracts. Compare external work and load direction through the increment
history; the final arrow alone does not establish correct work conjugacy.

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
newton-meters are resultants. Pascals, watts per square meter, watts per cubic meter, amperes per
square meter, and force per volume are intensities. Evidence should report both authored value and
integrated realized resultant where an integral is meaningful.

## What happens when CAD or mesh resolution changes?

Resultant-preserving loads keep their total while their nodal distribution changes. Intensity loads
keep their local magnitude, so their integrated total changes if selected area or volume changes.
Persistent region identity and projection evidence make that behavior reviewable; ordinal face or
element indices are not accepted as a durable substitute.

## How do I read a load declaration from left to right?

For a direct load, read the kind, target, value, and unit as one physical sentence:

```fea
load service_pressure {
    pressure on geometry.vessel.region.inner_wall = 2.5 MPa
}
```

This says that `service_pressure` is a normal surface-pressure action, that its target is the
persistent `inner_wall` face region, and that its constant intensity is 2.5 MPa. The declaration
does not choose a material, mesh entity, solver step, or load combination. Those belong to material
assignment, mesh realization, analysis activation, and `load_case`, respectively.

For a varying excitation, read the signal separately from the physical application:

```fea
signal pressure_cycle : pressure over time {
    samples = [
        { at = 0 s, value = 0 Pa },
        { at = 2 s, value = 2.5 MPa },
        { at = 10 s, value = 2.5 MPa },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation cyclic_pressure {
    pressure {
        target = geometry.vessel.region.inner_wall
        signal = signal.pressure_cycle
        kinematics = fixed_frame
    }
}
```

The signal owns the independent variable and sampling policy. The excitation owns the target and
application semantics. This separation lets several procedures reuse one measured history without
copying its samples or losing its source identity.

## How do I know whether RunMat supports the load I need?

Use the family pages as the closed source-language reference. A section with `.fea` syntax names a
supported declaration. A section that says a first-class contract is unavailable describes the
required physics and the safe alternative; it is not permission to place the same number into a
different kind. `runmat check study.fea` rejects unknown kinds, wrong quantities, invalid target
roles, missing signal domains, and incompatible combinations before execution.

## Which result proves that a load was applied correctly?

No single contour proves it. Use the invariant associated with the declaration:

| Load form | Evidence to compare with the authored input |
| --- | --- |
| Resultant force or moment | Integrated projected force and moment about a stated origin |
| Pressure, traction, flux, or current density | Resolved measure, orientation, and integrated resultant |
| Body or volume source | Resolved volume and integrated source |
| Terminal current, charge, or mass flow | Conserved terminal resultant and global balance |
| Signal-driven excitation | Sample-domain coverage, interpolation decisions, and active step/case |
| Coupled source | Producer result identity, transfer coverage, and conservation residual |

Then compare global equilibrium, energy, charge, or mass balance for the physics family. A plausible
field picture with the wrong integrated input is not an accepted load realization.
