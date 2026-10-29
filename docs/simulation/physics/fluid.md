# Fluid-flow analysis in RunMat: velocity, pressure, walls, and mass balance

RunMat's fluid family describes incompressible velocity–pressure problems with typed inlet velocity,
mass flow, moving/no-slip/slip walls, pressure reference, outlet pressure, symmetry, periodicity, and
initial state. Geometry regions identify the fluid volume and exact boundary faces; inlet/outlet
meaning is never inferred from face normals or names.

`runmat run` executes steady and transient incompressible laminar flow from a `.fea` study. The
canonical numerical route is a cell-centred finite-volume discretization on tetrahedral cells:
conservative face fluxes, Rhie–Chow pressure–velocity coupling with nonorthogonal correction,
linear-upwind convection and the full Newtonian stress. Transient flow advances with backward
Euler over the authored time grid, every interval converged to the same physical acceptance as a
steady solve. Every run checks momentum and continuity cell by cell and publishes velocity,
pressure, vorticity, face flux, the shear stress on every face (the wall shear stress on walls),
and the mass and momentum balance of every authored boundary.

The same flow solver drives the coupled [conjugate heat transfer](./conjugate-heat-transfer) and
[fluid–structure interaction](./fluid-structure-interaction) analyses.

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
advective fluxes. Velocity and pressure are solved together: each Picard step linearizes
convection and viscosity at the current state and solves the whole momentum–continuity system
with flexible GMRES from the shared sparse layer, preconditioned by a SIMPLE block sweep
(momentum solves with ILU(0), the Rhie–Chow pressure Laplacian with smoothed-aggregation algebraic
multigrid). The number of Picard steps stays nearly constant as the mesh is refined. The solve
acts on current velocity and pressure fluctuations;
the pressure reference is retained separately, avoiding an absolute-pressure round trip. Acceptance
checks momentum, continuity, constitutive flux and local/global physical balance.

Reduced channel/correlation operators still used by coupled paths are separate from this
canonical route. They do not establish 3D CFD, CHT or FSI qualification. Arbitrary turbulence,
separation and boundary-layer accuracy require their own independent evidence and supported operators.

## Which fluid procedures are defined?

| Procedure | Required domain/state | Current numerical status |
| --- | --- | --- |
| `fluid steady` | Static velocity/pressure state | Canonical Tet4 finite volume; runs from `runmat run` |
| `fluid transient` | Time samples and compatible initial state | Canonical backward Euler; runs from `runmat run` |
| `fluid harmonic` | Frequency samples and complex state | Typed contract only; no realization |
| `fluid nonlinear` | Load-factor domain | Typed contract only; no realization |

The canonical route admits Tet4 cell geometry, exact resolved boundary authority and compatible
cell-owned fluid properties. It rejects uncovered boundaries, periodic pairs and invalid sources;
it does not infer an implicit channel. Slip walls and symmetry planes are impermeable and
shear-free: no flux crosses them and only the normal viscous stress acts on them.

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
    analysis flow { fluid steady {} }
}
```

This is a steady authoring example; execution also requires admissible geometry,
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

The time grid must resolve boundary-signal breakpoints and the relevant convective/viscous scales;
signal breakpoints are added to the authored grid. Initial velocity and pressure are typed initial
conditions; without them the flow starts from rest. Each backward-Euler interval is solved to the
study's physical tolerance. Every interval keeps its solve evidence; fields are kept where outputs
sample them, so memory follows the samples you ask for, not the number of time steps. Each kept
state recovers cold from its predecessor without solving again.

Field and scalar outputs report the final state. To follow a quantity through the transient,
sample it over time; histories publish at each sample and can be reduced over time:

```fea
output peak_pressure_history {
    value = sample(
        max(output.pressure, over = geometry.channel.region.fluid),
        over = time,
        at = [1 s, 2 s, 3 s, 4 s, 5 s],
    )
    retention = explicit_samples
    maximum_samples = 5
}
```

Sample times join the time grid. A sampled state's fields also publish beside the final ones as
`<field>.time.<n>`, counting samples from 1 in increasing time.

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
