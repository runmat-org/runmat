# Fluid-flow analysis in RunMat: velocity, pressure, walls, and mass balance

RunMat's fluid family describes incompressible velocity–pressure problems with typed inlet velocity,
mass flow, moving/no-slip/slip walls, pressure reference, outlet pressure, symmetry, periodicity, and
initial state. Geometry regions identify the fluid volume and exact boundary faces; inlet/outlet
meaning is never inferred from face normals or names.

The `.fea` language checks steady, transient, harmonic, and nonlinear fluid procedure contracts.
The canonical numerical route is an **experimental cell-centered Tet4 finite-volume discretization**
of incompressible laminar momentum and continuity. The native typed executor connects one fresh,
independent, direct steady step through geometry, admission, CPU execution, retained fields/state,
accuracy and publication. Missing mandatory accuracy still blocks production.

Canonical transient flow uses backward Euler with immutable original-history checkpoints and scheduled
native load blocks. Authored transient whole-study roots, remote/distributed and GPU execution remain
unsupported. Symmetry, slip and periodic declarations remain typed contracts without canonical realization.

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

The canonical operator uses owner-oriented face incidence, Newtonian viscous stresses and conservative
advective fluxes. Shared GMRES correction solves with fixed-support ILU(0) act on current velocity and pressure fluctuations;
the pressure reference is retained separately, avoiding an absolute-pressure round trip. Acceptance
checks momentum, continuity, constitutive flux and local/global physical balance.

Reduced channel/correlation operators still used by experimental coupled paths are separate from this
canonical route. They do not establish 3D CFD, CHT or FSI qualification. Arbitrary turbulence,
separation and boundary-layer accuracy require their own independent evidence and supported operators.

## Which fluid procedures are defined?

| Procedure | Required domain/state | Current numerical status |
| --- | --- | --- |
| `fluid steady` | Static velocity/pressure state | Canonical Tet4 finite volume; experimental serial native study |
| `fluid transient` | Time samples and compatible initial state | Canonical backward Euler; experimental native checkpoint/block API |
| `fluid harmonic` | Frequency samples and complex state | Typed contract only; no realization |
| `fluid nonlinear` | Load-factor domain | Typed contract only; no realization |

The canonical route admits Tet4 cell geometry, exact resolved boundary authority and compatible
cell-owned fluid properties. It rejects uncovered boundaries, unsupported constraints and invalid
sources; it does not infer an implicit channel or certify a Tet10 fluid operator.

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

This is an experimental steady authoring example; execution also requires admissible geometry,
resolved boundaries, resources and the step's accuracy evidence.
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
Initial velocity and pressure are typed initial conditions. The canonical checkpoint/block route declares backward Euler and validates original sampled sources,
physical convergence and conservation. An authored transient whole-study graph is not yet admitted;
turbulence and spatial-accuracy evidence must not be inferred from the generic procedure name.

## Why is my pressure drop or mass balance wrong?

Check geometry units, cross-sectional area/hydraulic diameter, density and viscosity, inlet profile,
wall condition, outlet/reference pressure, Reynolds regime, developing length, and convergence of
both momentum and continuity. Require

$$
\left|\sum_{faces}\dot m\right|
\Big/\max\left(\sum_{inlets}|\dot m|,\epsilon\right)
$$

to meet the study tolerance. Also inspect the original cell momentum and face-flux constitutive
residuals and independently validate velocity, pressure and wall shear under refinement. A small
residual does not establish spatial accuracy.

See [fluid constraints](../constraints/fluid), [fluid loads](../loads/fluid),
[conjugate heat transfer](./conjugate-heat-transfer),
[fluid–structure interaction](./fluid-structure-interaction), and [current status](../status).
