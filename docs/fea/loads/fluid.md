# Fluid loads and boundary inputs

Mass flow is the conserved boundary resultant
$\dot m=\int_\Gamma\rho\mathbf u\cdot\mathbf n\,d\Gamma$. Inlet velocity and wall motion prescribe
kinematics; outlet pressure contributes the natural traction associated with the momentum equation.

| Kind | Quantity/fields | Use |
| --- | --- | --- |
| `inlet_velocity` | m/s | Apply an inlet-velocity signal |
| `mass_flow` | kg/s | Preserve total mass-flow rate |
| `wall_motion` | m/s | Drive wall velocity |
| `fluid.outlet_pressure` | `target`, `value` in Pa | Natural outlet pressure |

```fea
load flow_rate {
    mass_flow on geometry.channel.region.inlet = 0.2 kg/s
}

load outlet {
    fluid.outlet_pressure {
        target = geometry.channel.region.outlet
        value = 0 Pa
    }
}
```

## How do I apply a time-varying inlet velocity?

```fea
signal startup_profile : velocity over time {
    samples = [
        { at = 0 s, value = 0 m/s },
        { at = 0.5 s, value = 1.5 m/s },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation inlet_ramp {
    inlet_velocity {
        target = geometry.channel.region.inlet
        signal = signal.startup_profile
        direction = global([1, 0, 0])
        kinematics = fixed_frame
    }
}
```

This separates scalar speed history from direction and target. Use `fluid.velocity` constraint for a
constant prescribed vector. A flat inlet profile and a developed profile can yield different
pressure drop and entrance behavior; the current contract must not imply a profile it did not author.

## How do I prescribe a steady uniform inlet velocity?

Use the `fluid.velocity` essential constraint when the complete vector is known and constant:

```fea
constraint steady_inlet {
    fluid.velocity {
        target = geometry.channel.region.inlet
        value = [2 m/s, 0 m/s, 0 m/s]
    }
}
```

Use `inlet_velocity` for a signal-driven magnitude/direction. Do not apply both declarations to the
same target.

## How do I apply a parabolic or fully developed inlet profile?

A scalar `inlet_velocity` excitation does not silently invent a parabolic profile. Define the
spatial profile through a topology field or a RunMat function only when the admitted inlet contract
accepts that field shape, or extend the formulation with a typed developed-profile declaration.
For a circular laminar pipe, the analytic profile is
$u(r)=2\bar u[1-(r/R)^2]$; its mean, maximum, radius, axis, and coordinate frame must all remain
explicit.

## How do I specify turbulence intensity and length scale at an inlet?

These are transported-model boundary data, not velocity units. The current closed load set does not
expose a turbulence-inlet object. A turbulence formulation must add typed intensity/kinetic-energy,
length-scale or dissipation data with admissible ranges. Do not hide turbulence quantities in a
velocity signal or rely on undocumented solver defaults.

## How do I specify total mass flow?

```fea
load commanded_flow {
    mass_flow on geometry.channel.region.inlet = 0.2 kg/s
}
```

Mass flow preserves $\dot m$, not velocity. A realization must use density and resolved area to
construct a compatible boundary field and report the achieved resultant. Check sign/orientation and
do not separately prescribe an incompatible complete velocity field on the same inlet.

## How do I split a known mass flow across several inlets?

Create one `mass_flow` load per inlet with an explicit signed share, then include all of them in the
same load case. The shares should sum to the intended total. Do not apply the full total to every
inlet and expect RunMat to divide it by area automatically.

```fea
load branch_a_flow {
    mass_flow on geometry.manifold.region.inlet_a = 0.12 kg/s
}

load branch_b_flow {
    mass_flow on geometry.manifold.region.inlet_b = 0.08 kg/s
}
```

For a solved flow split, prescribe the physically known upstream/downstream conditions instead and
let the network resistance determine each branch rate.

## How do I drive a moving fluid wall?

```fea
signal belt_speed : velocity over time {
    samples = [
        { at = 0 s, value = 0 m/s },
        { at = 1 s, value = 2 m/s },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation moving_belt {
    wall_motion {
        target = geometry.channel.region.belt
        signal = signal.belt_speed
        direction = global([1, 0, 0])
        kinematics = fixed_frame
    }
}
```

This is prescribed kinematics. Use `fluid_structure.no_slip` when wall motion must come from and feed
back into a structural solution.

## How do I model a rotating wall, impeller, or cylinder?

A single `wall_motion` direction describes translational velocity, not rigid rotation over a
surface. Rotating-wall kinematics require axis, origin, angular speed and the spatial law
$\mathbf u=\boldsymbol\omega\times(\mathbf x-\mathbf x_0)$. Use a compatible generated field or
rotating-domain formulation when admitted. Do not assign one tangential vector to an entire curved
rotor.

## How do I model suction or blowing through a wall?

Treat the porous patch as an inlet/outlet boundary with a signed normal velocity or conserved mass
flow. Confirm normal orientation and exclude it from a no-slip wall region. A porous-medium pressure
drop is a separate constitutive domain law and cannot be inferred from boundary suction alone.

## How do I set outlet pressure?

```fea
load atmospheric_outlet {
    fluid.outlet_pressure {
        target = geometry.channel.region.outlet
        value = 0 Pa
    }
}
```

The value follows the study's pressure convention (typically gauge when a zero reference is used).
Outlet pressure is a natural boundary; `fluid.pressure_reference` fixes the gauge. Keep the two roles
distinct and avoid conflicting constraints on recirculating/inflow portions of an outlet.

## How do I set absolute pressure versus gauge pressure?

State the convention in study inputs and material/state definitions. A value of `0 Pa` commonly
means zero gauge pressure; it is not zero absolute pressure. Compressible density and thermodynamic
properties usually require absolute pressure even when the momentum unknown is stored relative to a
gauge. Do not mix conventions across inlet, outlet, and material models.

## How do I model several pressure outlets?

Declare one outlet-pressure load per region. Equal values permit the solution to determine flow
split; different values impose the intended pressure difference. Inspect signed flux at every outlet
because one may become an inlet. Do not add a redundant pressure-reference constraint to every
pressure outlet unless the formulation explicitly requires it.

## Should I prescribe inlet velocity or mass flow?

Use `inlet_velocity` when the velocity profile or speed history is known. Use `mass_flow` when the
integrated mass rate is known and the formulation is responsible for constructing a compatible
profile. For incompressible flow with constant density, volume flow is $\dot V=\dot m/\rho$; do not
enter volume-per-time units into `mass_flow`.

## How do I model a pump, fan, or pressure rise?

The current closed load set provides inlet velocity, mass flow, wall motion, and outlet pressure. It
does not yet define a first-class pump/fan curve or internal pressure-jump object. Do not hide a
pump curve in an unrelated outlet pressure. A production contract needs upstream/downstream regions,
flow-vs-pressure data, rotation/control state where applicable, interpolation/extrapolation policy,
and power-balance evidence.

## How do I model backflow at a pressure outlet?

An outlet-pressure boundary supplies traction, not a one-way valve. If the solved field reverses,
the admitted formulation must define its backflow stabilization and transported-state values.
Inspect signed flux on each outlet; do not clamp reverse flow by also prescribing velocity on the
same face.

## How do I apply a volumetric flow rate?

The current typed resultant is mass flow. Convert a known volumetric rate only with a declared
density valid for the modeled state: $\dot m=\rho\dot V$. For compressible or variable-density flow,
a single conversion may be invalid; a dedicated volumetric-flow contract would need reference-state
semantics.

## How do I model buoyancy or natural convection?

Buoyancy requires gravity plus a density/temperature relation in a coupled thermal-fluid
formulation. It is not represented by an inlet velocity. Use a conjugate/natural-convection model
that admits the Boussinesq or compressible body-force law and declares its reference temperature,
thermal expansion coefficient, and gravity vector.

## How do I model wind loading on a structure?

Solve the fluid traction and transfer it through an admitted fluid-structure interface when the
flow field matters. A guessed uniform structural pressure is a separate engineering approximation,
documented under [structural pressure loads](./structural#how-do-i-apply-pressure-to-a-structural-surface).
Do not label a pressure-only structural model as CFD or two-way FSI.

## How do I model a periodic pressure drop through a channel cell?

Use `fluid.periodic` for the mapped velocity/pressure fluctuation and a separate formulation input
for the macroscopic pressure gradient or conserved flow. The current load set has no first-class
periodic pressure-jump declaration. Do not place unrelated outlet-pressure boundaries on faces that
are also periodic.

## Why is the inlet flow direction reversed?

Check the geometry face normal, the declared global direction, and the sign of mass flow. A positive
mass-flow convention can be relative to the resolved outward normal, while a velocity vector is in
its declared frame. Preview the annotation on the exact CAD face and verify signed integrated flux
before interpreting downstream results.

## Why does a pressure outlet become an inlet?

Pressure is not a one-way constraint. Recirculation or an adverse pressure field can produce
backflow. Inspect transported quantities and stabilization supplied for backflow, move the boundary
farther from the recirculation region if physically appropriate, or model the connected downstream
domain. Do not add a contradictory velocity clamp merely to force the desired sign.

## What should I inspect after applying fluid loads?

Inspect resolved face area and orientation, integrated mass flux, imbalance across all boundaries,
wall-relative velocity, pressure convention, outlet backflow, and momentum/energy residuals. For
transient signals, verify that every solver time lies inside the authored interpolation/extrapolation
contract.

An inlet-velocity excitation preserves a velocity field; mass flow preserves a resultant. Wall
motion is distinct from a structural displacement unless an FSI interface transfers the motion.
Avoid combining incompatible prescribed velocity and pressure data on one boundary.

Further reading: [fluid constraints](../constraints/fluid), [fluid studies](../physics/fluid),
[conjugate heat transfer](../physics/conjugate-heat-transfer), and
[fluid–structure interaction](../physics/fluid-structure-interaction).
