# Fluid-flow analysis in RunMat: velocity, pressure, walls, and mass balance

RunMat's fluid family describes incompressible velocity–pressure problems with typed inlet velocity,
mass flow, moving/no-slip/slip walls, pressure reference, outlet pressure, symmetry, periodicity, and
initial state. Geometry regions identify the fluid volume and exact boundary faces; inlet/outlet
meaning is never inferred from face normals or names.

The `.fea` language checks steady, transient, harmonic, and nonlinear fluid procedure contracts.
The numerical library currently contains an **experimental reduced one-dimensional
velocity–pressure pipeline** using a hydraulic-diameter/friction correlation. The public typed
`.fea` executor does not connect fluid studies end to end. It is not a general Navier–Stokes finite-
element or finite-volume solver.

## What fluid equations does the contract target?

The continuum incompressible problem is

$$
\rho\left(\frac{\partial\mathbf u}{\partial t}
+\mathbf u\cdot\nabla\mathbf u\right)
=-\nabla p+\nabla\!\cdot\boldsymbol\tau+\mathbf f,
\qquad \nabla\cdot\mathbf u=0,
$$

with Newtonian $\boldsymbol\tau=\mu(\nabla\mathbf u+\nabla\mathbf u^{\mathsf T})$ where that
constitutive model applies. A full discretization must satisfy momentum and continuity and report
mass-balance evidence.

The current reduced component instead constructs an axial topology, estimates Reynolds number,
chooses

$$
f=\frac{64}{Re}\quad(Re\le2300),\qquad
f=0.3164Re^{-1/4}\quad(Re>2300),
$$

and uses the Darcy–Weisbach pressure gradient

$$
\frac{\Delta p}{L}=f\frac{\rho u|u|}{2D_h}.
$$

It iterates a relaxed pressure/velocity correction on that reduced topology and publishes momentum,
continuity, pressure-correction, velocity-correction, boundary-coverage, and mass-balance
diagnostics. It must not be used as evidence that arbitrary 3D flow, separation, turbulence, or
boundary layers have been resolved.

## Which fluid procedures are defined?

| Procedure | Required domain/state | Current numerical status |
| --- | --- | --- |
| `fluid steady` | Static velocity/pressure state | Experimental reduced channel iteration |
| `fluid transient` | Time samples and compatible initial state | Experimental reduced channel time stepping |
| `fluid harmonic` | Frequency samples and complex state | Typed contract only; no realization |
| `fluid nonlinear` | Load-factor domain | Typed contract only; no realization |

The reduced steady/transient branches may use canonical Tet4/Tet10 topology for projection, ordinal
distribution, or an implicit channel topology. Result evidence identifies the actual branch and any
unmapped fallback.

## How do I describe a steady incompressible flow study?

```fea
model channel_flow {
    assign material.water to geometry.channel.region.fluid

    constraint inlet {
        fluid.velocity {
            target = geometry.channel.region.inlet
            value = [1 m/s, 0 m/s, 0 m/s]
        }
    }
    constraint walls {
        fluid.no_slip_wall { target = geometry.channel.region.walls }
    }
    load outlet {
        fluid.outlet_pressure {
            target = geometry.channel.region.outlet
            value = 0 Pa
        }
    }
}

study steady_channel {
    geometry = geometry.channel
    mesh = mesh.channel
    model = model.channel_flow
    formulation = experimental
    analysis flow { fluid steady {} }
}
```

This is a typed authoring example; it is not currently runnable through the public `.fea` executor.
The material region must provide density and viscosity appropriate to the intended formulation.

## Should I specify inlet velocity or mass flow?

- `fluid.velocity` is an essential vector constraint.
- `inlet_velocity` is a signal-driven velocity excitation.
- `mass_flow` applies a total $\mathrm{kg/s}$ resultant and requires a projection consistent with
  boundary area and density.
- `fluid.outlet_pressure` is a natural pressure boundary.
- `fluid.pressure_reference` establishes a pressure datum.

Do not prescribe both a complete velocity and incompatible traction/pressure data on the same
boundary. A pressure reference fixes the gauge; it does not necessarily model an open outlet.

```fea
signal inlet_ramp : velocity over time {
    samples = [
        { at = 0 s, value = 0 m/s },
        { at = 0.5 s, value = 1 m/s },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation startup {
    inlet_velocity {
        target = geometry.channel.region.inlet
        signal = signal.inlet_ramp
        direction = global([1, 0, 0])
        kinematics = fixed_frame
    }
}
```

## How do no-slip, slip, symmetry, moving-wall, and periodic conditions differ?

- `fluid.no_slip_wall` matches velocity to a stationary wall.
- `fluid.moving_wall` prescribes wall velocity.
- `fluid.slip_wall` blocks normal flow but permits tangential slip.
- `fluid.symmetry` enforces the symmetry-plane condition.
- `fluid.periodic` pairs mapped regions through an identity, translation, or rotation.
- `wall_motion` is a signal-driven wall excitation; FSI-transferred wall motion instead belongs to
  a declared fluid–structure interface.

Choosing slip to improve convergence changes wall shear and pressure drop. Choosing symmetry on a
plane that is not physically symmetric changes the admissible flow.

## How do I describe transient flow?

```fea
analysis startup {
    fluid transient {
        start_time = 0 s
        end_time = 5 s
        time_step = 0.01 s
    }
}
```

The time grid must resolve boundary-signal breakpoints and the relevant convective/viscous scales.
Initial velocity and pressure are typed initial conditions. A full transient implementation would
also need a declared temporal scheme, nonlinear convergence policy, stabilization/turbulence model,
and spatial conservation evidence; those must not be inferred from the generic procedure name.

## Why is my pressure drop or mass balance wrong?

Check geometry units, cross-sectional area/hydraulic diameter, density and viscosity, inlet profile,
wall condition, outlet/reference pressure, Reynolds regime, developing length, and convergence of
both momentum and continuity. Require

$$
\left|\sum_{faces}\dot m\right|
\Big/\max\left(\sum_{inlets}|\dot m|,\epsilon\right)
$$

to meet the study tolerance. For the current reduced realization, also verify that the
Darcy–Weisbach/friction assumptions match the problem; they do not cover general 3D CFD.

See [fluid constraints](../constraints/fluid), [fluid loads](../loads/fluid),
[conjugate heat transfer](./conjugate-heat-transfer),
[fluid–structure interaction](./fluid-structure-interaction), and [current status](../status).
