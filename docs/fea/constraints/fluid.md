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

## How do I prescribe only the normal inlet speed?

Use the signal-driven `inlet_velocity` form when the scalar speed and direction are the controlled
inputs. `fluid.velocity` prescribes the complete vector and can unintentionally suppress tangential
components. For a curved inlet, a single global vector is not the same as a uniform normal speed;
use a normal/spatial field contract admitted by the formulation.

## How do I apply a developed inlet profile?

The constant vector form is uniform. A parabolic, turbulent, measured, or mapped profile must enter
through a typed topology field or function that preserves coordinates, units, and interpolation.
RunMat does not infer “fully developed” from a region name. See
[fluid loads](../loads/fluid#how-do-i-apply-a-parabolic-or-fully-developed-inlet-profile).

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

## How do I model a rotating no-slip wall?

The wall field must satisfy $\mathbf u_w=\boldsymbol\omega\times(\mathbf x-\mathbf x_0)$. A constant
`fluid.moving_wall` vector only represents translation. Use an admitted generated field or rotating
domain contract with explicit axis, origin, angular velocity, and frame; do not assign one tangent
vector across an entire cylindrical rotor.

## How do I apply a no-slip wall?

```fea
constraint viscous_walls {
    fluid.no_slip_wall { target = geometry.channel.region.walls }
}
```

No-slip imposes zero wall-relative velocity and is appropriate for ordinary viscous solid walls.
Check that inlet/outlet faces are excluded from a multi-face `walls` region; overlapping conditions
can overconstrain boundary DOFs.

## How do I model a stationary solid wall in a moving reference frame?

State wall velocity in the formulation's declared frame. “Stationary” in the laboratory frame can
be nonzero in a rotating frame. Do not mix global and rotating-frame values without the explicit
frame transformation and angular velocity.

## How do I apply a slip wall?

```fea
constraint ideal_slip_wall {
    fluid.slip_wall { target = geometry.channel.region.slip_faces }
}
```

Slip blocks normal velocity while allowing tangential motion. Use it for a deliberate inviscid or
symmetry-like wall approximation, not as a convergence workaround for a viscous boundary layer.

## How do I model an inviscid wall or centerline?

Use `fluid.slip_wall` for an impermeable inviscid wall where tangential traction is zero. Use
`fluid.symmetry` for an actual symmetry plane. They can produce similar local velocity restrictions,
but symmetry also asserts mirrored geometry, loading, and solution.

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

## Where should I place the incompressible pressure reference?

Place it on a physically meaningful reference region admitted by the formulation, usually associated
with a pressure outlet or known datum. The value fixes only the additive gauge. Results in pressure
difference and velocity should be invariant to a consistent datum shift; absolute thermodynamic
pressure for density/material laws remains a separate convention.

## Why is pressure still singular after adding a reference?

Check disconnected fluid components, empty/unresolved targets, and periodic components not connected
to the referenced region. Every independent pressure nullspace needs one admitted datum. Adding
several overlapping references to one component can instead overconstrain it.

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

## How do I model rotationally periodic turbomachinery passages?

Use `fluid.periodic` with a complete rotational transform when the formulation admits sector
periodicity. The primary-to-secondary mapping needs origin, axis, angle, and compatible face
topology. Rotor/stator interfaces with different pitch or relative motion are not ordinary periodic
pairs and require a dedicated interface.

## How do I model a free surface, porous wall, or outlet backflow?

These are not aliases for the essential constraints on this page. A free surface needs a
multiphase/free-surface law, a porous wall needs normal resistance or mass transfer, and outlet
backflow belongs to the outlet formulation with transported-state data. Use the corresponding typed
physics contract when admitted rather than weakening a no-slip or velocity constraint.

## Why is the velocity field overconstrained at an inlet edge?

An inlet face and adjacent no-slip wall share edge nodes. The formulation must reconcile the inlet
profile with zero wall-relative velocity; a uniform nonzero profile has a discontinuity at the
corner. Use a compatible developed profile or documented weak enforcement and inspect localized
residuals. Do not apply duplicate complete vectors to the same boundary entities.

## What should I inspect after applying fluid constraints?

Inspect exact target faces, wall/inlet overlap, prescribed velocity frame, normal flux, pressure
datum ownership, periodic pair coverage, mass conservation, wall-relative velocity, and constraint
residuals. In the Desktop preview, verify motion vectors and periodic pair annotations before the
solve.

An outlet-pressure natural boundary is documented under [fluid loads](../loads/fluid). Choose one
pressure datum consistent with the formulation and avoid simultaneously prescribing incompatible
velocity and pressure data on the same boundary.

## How do I combine an inlet, walls, and outlet in one RunMat fluid model?

Use `fluid.velocity` on the inlet, `fluid.no_slip_wall` on stationary viscous walls, one physical
pressure reference when required, and `fluid.outlet_pressure` as the outlet natural boundary. Give
each role a nonoverlapping CAD region. Do not prescribe full velocity and outlet pressure on the same
face or place a wall constraint over inlet edge entities without a compatible profile.

## Should I use a slip wall or a symmetry plane?

Both suppress normal flow under common formulations, but symmetry also constrains mirrored field
behavior and is justified only when geometry, loading, and expected solution are symmetric. A slip
wall represents an impermeable boundary without tangential no-slip shear. Use `fluid.symmetry` for a
cut model and `fluid.slip_wall` for a physical inviscid or slip boundary; inspect tangential traction
and normal flux.

## How do I model wall motion from a prescribed mechanical motion?

Use `fluid.moving_wall` when wall velocity is a known input. Use
`fluid_structure.no_slip` when structural displacement and velocity are solved and must feed back
with fluid traction. Applying both to the same face creates two owners for wall kinematics unless the
coupling procedure explicitly defines one as an override.

## How do I constrain a fluid domain with several disconnected cavities?

Each disconnected incompressible pressure component needs its own datum or a boundary relation that
establishes pressure. One reference in cavity A cannot remove the constant-pressure nullspace in
cavity B. Inspect connected components after interfaces and periodic pairs resolve; do not scatter
arbitrary point references until the linear system happens to factor.

## What fluid constraint evidence should be archived?

Retain resolved boundary areas and normals, prescribed velocity frames, wall-relative velocity,
pressure-reference ownership by connected component, periodic mapping coverage, constraint
residuals, and signed mass flux. For moving walls, also retain the motion signal or producing FSI
state identity.

## How do I choose CFD boundary conditions that are mathematically well posed?

For each connected incompressible domain, prescribe enough velocity/flow information to drive the
problem and exactly one independent pressure datum or outlet relation to remove constant-pressure
freedom. Classify every remaining face as wall, symmetry, periodic, opening, or coupled interface.
Do not prescribe all velocity components and pressure on the same boundary unless the formulation
explicitly defines that mixed condition.

Run the boundary audit before meshing: report face coverage, overlaps, unclassified faces, connected
fluid components, and pressure-reference ownership. After solving, check signed mass conservation.

## How do I model a stationary pipe or duct wall?

Use `fluid.no_slip_wall` for viscous stationary walls. Use `fluid.slip_wall` only for an inviscid or
intentional zero-normal-flow approximation. A symmetry plane also imposes symmetry on transported
fields and gradients; it is not merely a low-friction wall.

Near-wall mesh and turbulence treatment remain formulation choices. A no-slip declaration does not
prove that wall shear or heat transfer is resolved; retain $y^+$ or the applicable near-wall metric,
wall force, and mesh-convergence evidence.

## How do I constrain a rotating or translating reference-frame model?

Prescribe wall velocity in the inertial frame with `fluid.moving_wall`, or use a rotating-domain
formulation whose frame owns Coriolis/centrifugal terms. Do not combine a rotating frame and the same
absolute wall motion without transforming velocities explicitly. At an interface between frames,
use the admitted sliding/mixing relation rather than a generic periodic pair.

Archive frame identity, axis, origin, angular speed, relative wall velocity, and torque/power. Check
that a wall intended to be stationary relative to the rotor has zero relative velocity.

## How do I constrain an internal-flow periodic cell?

Pair inlet and outlet geometry with `fluid.periodic`, declare the exact translation/rotation, and
supply the driving pressure-gradient, body-force, or mass-flow contract separately. Periodicity only
relates fields; it does not create pressure drop or flow.

Validate pair coverage and transformed normals, then compare mass flux and transported quantities on
both faces. The pressure decomposition (periodic fluctuation plus imposed mean gradient) must belong
to the selected formulation and be visible in evidence.
