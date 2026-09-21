# Fluid constraints

Fluid constraints act on velocity and the pressure gauge. No-slip enforces
$\mathbf u=\mathbf u_{wall}$; slip enforces $\mathbf u\cdot\mathbf n=0$ with zero tangential
traction; a pressure reference fixes the additive pressure nullspace without by itself defining an
outlet flux law.

| Kind | Required fields | Meaning |
| --- | --- | --- |
| `fluid.velocity` | `target`, `value` | Prescribe fluid velocity |
| `fluid.moving_wall` | `target`, `value` | Prescribe wall velocity |
| `fluid.no_slip_wall` | `target` | Match velocity to a stationary wall |
| `fluid.slip_wall` | `target` | Prevent normal flow while permitting tangential slip |
| `fluid.symmetry` | `target` | Apply a symmetry-plane condition |
| `fluid.pressure_reference` | `target`, `value` | Establish the pressure datum |
| `fluid.periodic` | periodic pair/transform | Pair periodic flow boundaries |

```fea
constraint inlet_velocity {
    fluid.velocity {
        target = geometry.channel.region.inlet
        value = [1.5 m/s, 0 m/s, 0 m/s]
    }
}

constraint channel_walls {
    fluid.no_slip_wall { target = geometry.channel.region.walls }
}

constraint pressure_datum {
    fluid.pressure_reference {
        target = geometry.channel.region.outlet
        value = 0 Pa
    }
}
```

## How do I prescribe an inlet velocity profile?

Use `fluid.velocity` for a known velocity vector on a region:

```fea
constraint uniform_inlet {
    fluid.velocity {
        target = geometry.channel.region.inlet
        value = [1.5 m/s, 0 m/s, 0 m/s]
    }
}
```

This is a uniform prescribed value. Use a signal-driven `inlet_velocity` excitation for time or
frequency dependence. Use `mass_flow` when the total flow is known but its spatial profile is not.

## How do I model a moving wall in CFD?

```fea
constraint moving_lid {
    fluid.moving_wall {
        target = geometry.cavity.region.lid
        value = [1 m/s, 0 m/s, 0 m/s]
    }
}
```

This prescribes wall velocity and no-slip relative to that wall. For a time-varying wall use a
`wall_motion` excitation. For motion supplied by a structural solution use an FSI interface instead
of duplicating the displacement history manually.

## How do I apply a no-slip wall?

```fea
constraint viscous_walls {
    fluid.no_slip_wall { target = geometry.channel.region.walls }
}
```

No-slip imposes zero wall-relative velocity and is appropriate for ordinary viscous solid walls.
Check that inlet/outlet faces are excluded from a multi-face `walls` region; overlapping conditions
can overconstrain boundary DOFs.

## How do I apply a slip wall?

```fea
constraint ideal_slip_wall {
    fluid.slip_wall { target = geometry.channel.region.slip_faces }
}
```

Slip blocks normal velocity while allowing tangential motion. Use it for a deliberate inviscid or
symmetry-like wall approximation, not as a convergence workaround for a viscous boundary layer.

## How do I apply a fluid symmetry plane?

```fea
constraint centerplane {
    fluid.symmetry { target = geometry.half_channel.region.symmetry_face }
}
```

The geometry, inlet profile, other boundaries, and expected solution must all be symmetric. The
reported flow/resultants correspond to the modeled fraction and must be scaled accordingly.

## How do I set the pressure datum in incompressible flow?

`fluid.pressure_reference` sets the additive pressure gauge:

```fea
constraint pressure_reference {
    fluid.pressure_reference {
        target = geometry.channel.region.reference_face
        value = 0 Pa
    }
}
```

It is not automatically an outlet law. Use `fluid.outlet_pressure` for a natural outlet boundary.
Do not add multiple incompatible references to suppress a singular matrix.

## How do I model periodic flow through a repeating passage?

```fea
constraint repeating_passage {
    fluid.periodic {
        primary = geometry.cell.region.inlet_pair
        secondary = geometry.cell.region.outlet_pair
        translation = [50 mm, 0 mm, 0 mm]
        tolerance = 1 um
    }
}
```

Plain periodicity maps velocity/pressure fluctuations across the pair. A macroscopic pressure drop
or mass-flow driving condition remains a separate physical input; periodicity does not invent it.

An outlet-pressure natural boundary is documented under [fluid loads](../loads/fluid). Choose one
pressure datum consistent with the formulation and avoid simultaneously prescribing incompatible
velocity and pressure data on the same boundary.
