# Fluid constraints

Fluid constraints act on velocity and the pressure gauge. No-slip enforces
$\mathbf u=\mathbf u_{wall}$; slip enforces $\mathbf u\cdot\mathbf n=0$ with zero tangential
traction; a pressure reference fixes the additive pressure nullspace without by itself defining an
outlet flux law.

| Kind | Required fields | Meaning |
| --- | --- | --- |
| `fluid.velocity` | `target`, three-component `value` in m/s or vector velocity signal | Prescribe the full velocity vector |
| `fluid.moving_wall` | `target`, three-component `value` in m/s or vector velocity signal | Prescribe the full wall-velocity vector |
| `fluid.no_slip_wall` | `target` | Match velocity to a stationary wall |
| `fluid.slip_wall` | `target` | Prevent normal flow while permitting tangential slip |
| `fluid.symmetry` | `target` | Apply a symmetry-plane condition |
| `fluid.pressure_reference` | `target`, `value` | Establish the pressure datum |
| `fluid.periodic` | periodic pair/transform | Pair periodic flow boundaries [not yet in the current release](pending:fluid.periodic) |

```fea
signal inlet_speed : velocity over constant {
    value = 1.5 m/s
}

excitation inlet_velocity {
    inlet_velocity {
        target = geometry.channel.region.inlet
        signal = signal.inlet_speed
        direction = global([1, 0, 0])
        kinematics = fixed_frame
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

`fluid.velocity` and `fluid.moving_wall` require exactly three signed velocity components.
Inline components are explicitly global coordinates, for example:

```fea
constraint inlet {
    fluid.velocity {
        target = geometry.channel.region.inlet
        value = [-1.5 m/s, 0 m/s, 0 m/s]
    }
}
```

`value = signal.<name>` instead retains a declared three-component velocity signal, including
its frame and constant/table representation. It is not converted to a speed, maximum magnitude,
or guessed axis. The formulation must explicitly support realizing that signal at the boundary;
unsupported frames or histories are rejected. Scalar constants and scalar signals are not vector
constraint inputs. The reduced channel adapter cannot realize these exact vector constraints and
rejects them; this shared input contract alone does not qualify a numerical flow solver.

For a scalar speed with an explicitly declared direction, the shared excitation form is:

```fea
signal inlet_speed : velocity over constant {
    value = 1.5 m/s
}

excitation uniform_inlet {
    inlet_velocity {
        target = geometry.channel.region.inlet
        signal = signal.inlet_speed
        direction = global([1, 0, 0])
        kinematics = fixed_frame
    }
}

load_case operating { include excitation.uniform_inlet }
```

`direction` is a dimensionless global vector. The signal carries m/s and may instead be defined over
time when a transient procedure consumes it. Use `mass_flow` when the total flow is known but its
spatial profile is not, when the selected formulation supports conservative realization.
The runtime velocity-constraint constructors likewise require a cell vector of three typed
`fea.Velocity` values (global components), or a framed three-component `fea.VelocitySignal`
also retained in the model signal inventory. They do not reinterpret one scalar velocity value
as a vector. A pressure reference remains a gauge, not a physical outlet boundary; physical
outlet pressure uses the existing `fluid.outlet_pressure` boundary operator.

## How do I prescribe only the normal inlet speed?

Use the signal-driven `inlet_velocity` form above when scalar speed and a global direction are the
controlled inputs. For a curved inlet, one global direction is not a uniform local-normal field.
The current source surface has no `normal(...)` direction or spatial-profile expression; use an
admitted topology field/generator when the selected formulation provides one. Otherwise that case
is not expressible by the current typed input and must be rejected, not approximated implicitly.

## How do I apply a developed inlet profile?

The directed excitation is uniform. A parabolic, turbulent, measured, or mapped profile must enter
through a typed topology field or function that preserves coordinates, units, and interpolation.
RunMat does not infer “fully developed” from a region name. See
[fluid loads](../loads/fluid#how-do-i-apply-a-parabolic-or-fully-developed-inlet-profile).

## How do I model a moving wall in CFD?

```fea
constraint moving_lid {
    fluid.moving_wall {
        target = geometry.cavity.region.lid
        value = 1 m/s
    }
}
```

This is the exact current constraint shape: it carries a scalar speed and no authorable direction
or frame. Use it only when the consuming formulation gives that scalar an unambiguous wall-motion
meaning. For directed or time-varying translation, use a `wall_motion` excitation with a scalar
velocity signal and `direction = global([x, y, z])`. For motion supplied by a structural solution,
use an FSI interface instead of duplicating the displacement history manually.

## How do I model a rotating no-slip wall?

The wall field must satisfy $\mathbf u_w=\boldsymbol\omega\times(\mathbf x-\mathbf x_0)$. Neither
the scalar `fluid.moving_wall` constraint nor a single-direction `wall_motion` excitation expresses
that spatial law. A rotating-wall generator with axis, origin, angular speed, and frame is not in
the current closed source surface. Use a rotating-domain/formulation input that explicitly admits
the law; otherwise rotating-wall kinematics are unsupported.

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

The current scalar wall constraint cannot author a rotating or translating coordinate frame.
“Stationary” in the laboratory frame can be nonzero in a rotating frame, so a moving-reference-frame
procedure must own that transformation explicitly. Do not infer a frame from the CAD orientation or
the region name.

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
residuals. Do not assign duplicate kinematic owners to the same boundary entities.

## What should I inspect after applying fluid constraints?

Inspect exact target faces, wall/inlet overlap, excitation direction, normal flux, pressure
datum ownership, periodic pair coverage, mass conservation, wall-relative velocity, and constraint
residuals. In the Desktop preview, verify motion vectors and periodic pair annotations before the
solve.

An outlet-pressure natural boundary is documented under [fluid loads](../loads/fluid). Choose one
pressure datum consistent with the formulation and avoid simultaneously prescribing incompatible
velocity and pressure data on the same boundary.

## How do I combine an inlet, walls, and outlet in one RunMat fluid model?

Use a directed `inlet_velocity` excitation on the inlet, `fluid.no_slip_wall` on stationary viscous
walls, one physical pressure reference when required, and `fluid.outlet_pressure` as the outlet
natural boundary. Give each role a nonoverlapping CAD region:

```fea
signal bulk_speed : velocity over constant { value = 2 m/s }

excitation pipe_inlet {
    inlet_velocity {
        target = geometry.pipe.region.inlet
        signal = signal.bulk_speed
        direction = global([1, 0, 0])
        kinematics = fixed_frame
    }
}

constraint pipe_wall {
    fluid.no_slip_wall { target = geometry.pipe.region.wall }
}

constraint outlet_gauge {
    fluid.pressure_reference {
        target = geometry.pipe.region.outlet
        value = 0 Pa
    }
}

load outlet_traction {
    fluid.outlet_pressure {
        target = geometry.pipe.region.outlet
        value = 0 Pa
    }
}

load_case operating {
    include excitation.pipe_inlet
    include load.outlet_traction
}
```

The pressure reference removes the algebraic gauge; the outlet load supplies the physical natural
boundary. Whether both are needed on the same outlet is formulation-dependent and must be admitted
explicitly. Before solving, require zero unclassified boundary area and no overlap among inlet,
wall, and outlet regions. After solving, check
$\left|\sum_\Gamma\dot m_\Gamma\right|/\sum_{in}|\dot m|$ against the study tolerance, report the
pressure drop, and compare wall shear/resultant with a mesh-refined run.

## Should I use a slip wall or a symmetry plane?

Both suppress normal flow under common formulations, but symmetry also constrains mirrored field
behavior and is justified only when geometry, loading, and expected solution are symmetric. A slip
wall represents an impermeable boundary without tangential no-slip shear. Use `fluid.symmetry` for a
cut model and `fluid.slip_wall` for a physical inviscid or slip boundary; inspect tangential traction
and normal flux.

## How do I model wall motion from a prescribed mechanical motion?

Use scalar `fluid.moving_wall` only when the consuming formulation defines its wall-speed direction
unambiguously; use directed `wall_motion` for supported prescribed translation. Use
`fluid_structure.no_slip` when structural displacement and velocity are solved and must feed back
with fluid traction. Applying both to the same face creates two owners for wall kinematics unless the
coupling procedure explicitly defines one as an override.

## How do I constrain a fluid domain with several disconnected cavities?

Each disconnected incompressible pressure component needs its own datum or a boundary relation that
establishes pressure. One reference in cavity A cannot remove the constant-pressure nullspace in
cavity B. Inspect connected components after interfaces and periodic pairs resolve; do not scatter
arbitrary point references until the linear system happens to factor.

## What fluid constraint evidence should be archived?

Retain resolved boundary areas and normals, excitation directions, wall-relative velocity,
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

Use a rotating-domain formulation whose frame owns Coriolis/centrifugal terms. The current scalar
`fluid.moving_wall` syntax has no frame, axis, or origin and therefore cannot by itself establish an
inertial-frame rotating wall. Do not combine a rotating frame and an independently prescribed
absolute wall motion without an explicit supported transformation. At an interface between frames,
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
