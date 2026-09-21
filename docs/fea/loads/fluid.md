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

## How do I specify total mass flow?

```fea
load commanded_flow {
    mass_flow on geometry.channel.region.inlet = 0.2 kg/s
}
```

Mass flow preserves $\dot m$, not velocity. A realization must use density and resolved area to
construct a compatible boundary field and report the achieved resultant. Check sign/orientation and
do not separately prescribe an incompatible complete velocity field on the same inlet.

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

An inlet-velocity excitation preserves a velocity field; mass flow preserves a resultant. Wall
motion is distinct from a structural displacement unless an FSI interface transfers the motion.
Avoid combining incompatible prescribed velocity and pressure data on one boundary.

Further reading: [fluid constraints](../constraints/fluid), [fluid studies](../physics/fluid),
[conjugate heat transfer](../physics/conjugate-heat-transfer), and
[fluid–structure interaction](../physics/fluid-structure-interaction).
