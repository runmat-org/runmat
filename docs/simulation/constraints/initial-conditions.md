# Initial conditions

An initial condition defines physical state at the first time sample. It is not a boundary condition:
it applies at $t=t_0$ and does not continue constraining the field unless a separate constraint says
so. For structural dynamics, the initial state participates in

$$
\mathsf M\ddot{\mathbf u}_0+\mathsf C\dot{\mathbf u}_0+
\mathsf K\mathbf u_0=\mathbf f(t_0).
$$

For transient heat transfer, $\mathbf T_0$ supplies stored energy
$\mathsf C_T\mathbf T_0$. Fluid, acoustic, electrical, and magnetic initial fields similarly must
match the state type and solution space of the consuming procedure.

| Kind | Quantity | Typical target |
| --- | --- | --- |
| `structural.displacement` | vector length | structural region or whole model |
| `structural.velocity` | vector velocity | structural region or whole model |
| `structural.angular_velocity` | vector angular velocity | structural region or whole model |
| `thermal.temperature` | scalar temperature | thermal region or whole model |
| `electrical.potential` | scalar potential | electrical region or whole model |
| `magnetic.vector_potential` | vector potential | magnetic region or whole model |
| `acoustic.pressure` | scalar pressure | acoustic region or whole model |
| `fluid.pressure` | scalar pressure | fluid region or whole model |
| `fluid.velocity` | vector velocity | fluid region or whole model |

```fea
model warmed_part {
    initial at_rest {
        structural.velocity {
            target = whole_model
            uniform = [0 m/s, 0 m/s, 0 m/s]
            frame = global
        }
    }
    initial room_temperature {
        thermal.temperature {
            target = geometry.part.region.body
            uniform = 293.15 K
        }
    }
}
```

Declaring an initial condition does not activate it. The consuming procedure selects one or more
unique `model.initial.<name>` references:

```fea
study release_test {
    geometry = geometry.part
    mesh = mesh.part
    model = model.warmed_part

    analysis response {
        structural transient {
            start_time = 0 s
            end_time = 20 ms
            time_step = 0.1 ms
            initial_conditions = [model.initial.at_rest]
        }
    }
}
```

The compiler checks that the procedure's physics family can consume every selected state kind and
rejects duplicate references. An unselected declaration has no effect on the solve.

## What exact value sources can an initial condition use?

Every initial condition requires exactly one source: `uniform`, a `sample { ... }` block, or a
`field_artifact { ... }` block. Vector kinds also require `frame = global`, `material`, `element`,
or `boundary_normal_tangent`; scalar kinds cannot declare a frame. `target` is either `whole_model`
or a CAD region reference.

Sampling is explicit about the signal, sample coordinate, and side limit:

```fea
signal measured_temperature : temperature over time {
    samples = [
        { at = 0 s, value = 293.15 K },
        { at = 10 s, value = 323.15 K },
    ]
    interpolation = linear
    extrapolation = reject
}

initial measured_start {
    thermal.temperature {
        target = geometry.part.region.body
        sample {
            signal = signal.measured_temperature
            at = 10 s
            approach = left_limit
        }
    }
}
```

`approach` is exactly `at`, `left_limit`, or `right_limit`. Signal quantity and scalar/vector shape
must match the initial-condition kind. Time-domain signals use a time-valued `at`; constant and
study-parameter signals use a dimensionless value. Frequency-domain signals cannot define physical
initial state and are rejected.

```fea
initial mapped_temperature {
    thermal.temperature {
        target = geometry.part.region.body
        field_artifact {
            artifact_id = "thermal-preload-042"
            content_digest = "sha256:..."
            topology_digest = "sha256:..."
            field_id = "temperature"
        }
    }
}
```

All four artifact strings are required and nonempty. The declaration binds identity; realization
must still verify field quantity, units, topology compatibility, coverage, and interpolation before
the procedure consumes it.

## How do I start a structural transient analysis from a displaced shape?

Use `structural.displacement` when the initial nodal configuration is known independently of the
first applied load:

The focused `initial` declarations below are written as model-body excerpts; place each one inside
the `model { ... }` block that owns it, as in the complete example above.

```fea
initial released_deflection {
    structural.displacement {
        target = geometry.spring.region.flexure
        uniform = [0 mm, -2 mm, 0 mm]
        frame = global
    }
}
```

An imposed initial displacement is released at the first transient step unless a continuing
`structural.prescribed` constraint holds it. A uniform displacement of an unconstrained whole body
is rigid translation and stores no elastic energy; a meaningful released-deflection study needs a
spatial field or constraints that produce strain.

## How do I give a structure initial linear velocity?

```fea
initial launch_velocity {
    structural.velocity {
        target = geometry.projectile.region.body
        uniform = [25 m/s, 0 m/s, 0 m/s]
        frame = global
    }
}
```

This initializes $\dot{\mathbf u}_0$. It does not apply a continuing force and does not prescribe
the later trajectory. Select the actual moving body, state the coordinate frame, and verify that
contact or supports do not contradict the initial velocity at $t_0$.

## How do I initialize a rotating component?

```fea
initial spin_state {
    structural.angular_velocity {
        target = geometry.rotor.region.solid
        uniform = [0 rad/s, 0 rad/s, 500 rad/s]
        frame = global
    }
}
```

Angular velocity initializes rotational motion about the declared frame. It is not centrifugal
loading by itself. A formulation that derives gyroscopic or centrifugal terms must explicitly
admit those terms; otherwise use the documented preload/body-force route for the intended physics.

## How do I set the initial temperature for transient heat transfer?

```fea
initial soak_temperature {
    thermal.temperature {
        target = geometry.battery.region.cells
        uniform = 318.15 K
    }
}
```

Initial temperature supplies $\mathbf T_0$ to the transient capacity term. It does not clamp the
temperature after $t_0$; use a `thermal.temperature` constraint for a controlled boundary. Use
absolute temperatures and cover every connected thermal region whose default is not appropriate.

## How do I initialize electric potential?

```fea
initial charged_state {
    electrical.potential {
        target = geometry.device.region.dielectric
        uniform = 12 V
    }
}
```

This initializes the electrical state where the admitted transient formulation stores one. It does
not replace the potential reference required to remove the field nullspace and does not imply a
terminal remains driven at 12 V.

## How do I initialize magnetic vector potential?

```fea
initial remanent_field_state {
    magnetic.vector_potential {
        target = geometry.motor.region.field_domain
        uniform = [0 T*m, 0 T*m, 1e-6 T*m]
        frame = global
    }
}
```

Vector potential is gauge-dependent. The initial field, gauge constraint, and formulation basis
must be compatible. Do not treat one uniform vector-potential value as a direct magnetic-flux-density
specification: $\mathbf B=\nabla\times\mathbf A$.

## How do I initialize acoustic pressure?

```fea
initial cavity_pressure {
    acoustic.pressure {
        target = geometry.cavity.region.volume
        uniform = 2 Pa
    }
}
```

This sets the acoustic perturbation pressure, not atmospheric absolute pressure. A second-order
transient acoustic formulation may also require a compatible initial pressure-rate state; RunMat
must reject a requested state component that the selected procedure cannot express.

## How do I initialize fluid pressure?

```fea
initial pressure_guess {
    fluid.pressure {
        target = geometry.channel.region.volume
        uniform = 0 Pa
    }
}
```

For incompressible flow, pressure is defined relative to a gauge. The initial value can seed a
transient or nonlinear solve, but a `fluid.pressure_reference` or physically equivalent admitted
condition is still required. Use the study's gauge convention consistently.

## How do I initialize a fluid velocity field?

```fea
initial developed_flow_seed {
    fluid.velocity {
        target = geometry.channel.region.volume
        uniform = [1 m/s, 0 m/s, 0 m/s]
        frame = global
    }
}
```

The initial field should satisfy impermeable-wall and inlet constraints as closely as practical.
A uniform nonzero velocity across a no-slip channel is generally not a developed solution and can
create a sharp startup transient. For a spatially resolved field, use a topology-bound field
artifact rather than splitting the volume into arbitrary constant bands.

## How do I restart from a previous FEA result?

A restart is not a collection of manually copied initial values. Continue from the retained typed
state produced by the earlier analysis so displacement, velocity, temperature, pressure, and any
constitutive/internal variables remain on the same topology and basis. If remeshing or geometry
revision is involved, require an explicit transfer operation and inspect its coverage and
conservation evidence.

The source spelling is family-qualified—`structural.displacement`, `structural.velocity`,
`structural.angular_velocity`, `thermal.temperature`, `electrical.potential`,
`magnetic.vector_potential`, `acoustic.pressure`, `fluid.pressure`, or `fluid.velocity`. Vector
kinds require `frame`; scalar kinds use the global frame implicitly. Values may be uniform, sampled
from a compatible signal at an explicit side limit, or loaded from a topology-bound field artifact.

The current authored initial-condition set does not include acceleration, stress, strain,
pressure-rate, turbulence variables, plastic/internal variables, contact status, or integrator
history. A `field_artifact` containing one visible primary field is therefore not a complete restart.
Where any omitted state affects the continuation, use a typed state edge/checkpoint supported by the
procedure; if none exists, restart is unsupported for that case.

Initial declarations carry an explicit frame for vector quantities and target either a typed region
or `whole_model`. Overlapping initial conditions must agree on shared DOFs; an unspecified field
uses only the formulation's documented default, never the first sample of a load signal.

Use nonzero structural displacement together with velocity/acceleration consistent with the intended
starting equilibrium. Use a realistic absolute temperature rather than assuming zero means ambient.
For restarted work, prefer the retained typed state from the producing analysis; a hand-authored
initial field is not interchangeable with a checkpoint carrying internal constitutive state.

## Why does my transient solution jump at the first time step?

Common causes are an initial field that conflicts with an active essential constraint, a load that
steps from zero to its first sample, or displacement/velocity/acceleration that do not satisfy the
initial equilibrium equation. Inspect the resolved initial-state coverage, the signal side limit at
$t_0$, and the first-step residual. Do not hide the discontinuity by reducing the time step unless
the physical startup is genuinely discontinuous.

## How do I initialize several disconnected parts?

Give each part an explicit region and initial declaration when their states differ. A whole-model
initial condition is suitable only when every admitted field degree of freedom starts with the same
value. Overlapping declarations must agree on shared projected entities; source order does not
provide an override rule.

```fea
initial hot_insert {
    thermal.temperature {
        target = geometry.assembly.region.insert
        uniform = 423.15 K
    }
}

initial cold_housing {
    thermal.temperature {
        target = geometry.assembly.region.housing
        uniform = 293.15 K
    }
}
```

This is the correct starting point for a thermal-contact transient. Two continuing
`thermal.temperature` constraints would instead hold both bodies at those temperatures indefinitely.

## How do I initialize a field from measured data?

Use a topology-bound field artifact whose units, coordinate frame, source topology, interpolation,
coverage, and source digest are retained. The transfer must report uncovered target measure and a
conservation or interpolation error appropriate to the field. A CSV of unlabeled node values is not
a portable initial condition because remeshing changes node identity and order.

## How do initial conditions interact with constraints at time zero?

The admitted initial state must satisfy active essential constraints. If a fixed support owns
$\mathbf u=0$, a nonzero displacement or velocity on the same constrained degrees of freedom is a
conflict. For prescribed motion, use the correct left/right signal limit at $t_0$ and initialize
velocity/acceleration consistently. RunMat should diagnose an incompatible overlap before advancing
the first step.

## How do I check initial energy and momentum?

For structures, compare $\tfrac12\dot{\mathbf u}_0^T\mathsf M\dot{\mathbf u}_0$ and
$\tfrac12\mathbf u_0^T\mathsf K\mathbf u_0$ with the intended release state, and inspect total
linear/angular momentum. For thermal work, compare initial stored energy against the reference used
by the material capacity law. For fluids and acoustics, inspect divergence/compatibility and the
pressure/velocity energy convention. These checks expose unit, frame, and whole-model targeting
errors before they become a transient startup spike.

## When should I use restart state instead of an initial condition?

Use restart state whenever the producing solve owns history that cannot be reconstructed from the
visible primary field: plastic strain, contact status, internal variables, integration state,
accumulated dissipation, or a reduced-order basis. Use initial declarations for a genuinely new
study whose required state is completely described by the authored fields. A restart must bind to
the exact producing topology and operator contract or pass an explicit governed transfer.

## How do I choose initial conditions for a static-to-transient release?

Use the converged static result as the initial displacement and internal state, and state whether
initial velocity is zero. If preload, plasticity, contact, or another history variable matters, use
a typed restart/state edge rather than copying displacement alone. The first transient step should
recover static equilibrium when the same loads remain active.

For a sudden release, make the removed load or constraint transition explicit at $t_0$ and refine
the time step around it. Check the initial force residual and the exchange between strain and kinetic
energy.

## How do I initialize a thermally equilibrated assembly?

Use a uniform or mapped initial temperature only when it describes the complete thermal state needed
by the procedure. If the assembly begins in a nonuniform steady state, compute or import that state
with provenance and then start the transient from it. Do not impose the same map as continuing
`thermal.temperature` constraints unless reservoirs actually hold those boundaries.

Verify initial stored energy and the first-step energy residual. An unexplained spike usually means
the initial field is incompatible with boundary conditions, contact, or temperature-dependent
material properties.

## How do I initialize CFD from rest or from a developed flow?

Rest means zero velocity plus a pressure field compatible with gravity/reference conditions. For a
developed-flow start, initialize velocity and pressure together from a compatible analytic,
precursor, or restart solution. Projecting velocity alone can create a large divergence correction
and misleading startup transient.

Inspect divergence, boundary compatibility, mass imbalance, and kinetic energy at the first accepted
step. For turbulence or other transported models, their internal fields also belong to the restart
state; they cannot be inferred from velocity without an explicit initialization law.

## How do I initialize a harmonic or frequency-domain analysis?

Frequency-domain procedures solve complex steady response and normally do not consume time-domain
initial conditions. Their state comes from frequency, phase convention, material/operator state, and
possibly a preload result. Use a state edge for prestress or temperature dependence; do not add an
initial displacement hoping to bias a harmonic solution.

If a procedure accepts continuation between frequency points, that is a solver initialization aid,
not physical initial data. It must not change the converged branch without explicit nonlinear path
semantics.

See [study state flow](../language/studies-analyses-and-state),
[transient structural analysis](../physics/structural), and [transient heat transfer](../physics/thermal).
