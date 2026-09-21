# Initial conditions for transient FEA

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
| `displacement` | vector length | structural region or whole model |
| `velocity` | vector velocity | structural region or whole model |
| `angular_velocity` | vector angular velocity | structural region or whole model |
| `temperature` | scalar temperature | thermal region or whole model |
| `electric_potential` | scalar potential | electrical region or whole model |
| `magnetic_vector_potential` | vector potential | magnetic region or whole model |
| `acoustic_pressure` | scalar pressure | acoustic region or whole model |
| `fluid_pressure` | scalar pressure | fluid region or whole model |
| `fluid_velocity` | vector velocity | fluid region or whole model |

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

See [study state flow](../language/studies-analyses-and-state),
[transient structural analysis](../physics/structural), and [transient heat transfer](../physics/thermal).
