# Conjugate heat-transfer analysis in RunMat: fluid cooling and solid conduction

Conjugate heat transfer (CHT) couples energy transport in a fluid to conduction in a neighboring
solid. Fluid and solid retain different materials and equations; a declared interface transfers
temperature and normal heat flux.

The `.fea` language checks steady, transient, and nonlinear CHT contracts. The numerical library
currently admits an experimental **partitioned reduced-flow/thermal realization** with heuristic
interface transfer. The public typed `.fea` executor does not connect it end to end, and it is not a
general 3D CHT finite-volume/finite-element solver.

## What equations define conjugate heat transfer?

The solid satisfies

$$
\rho_s c_s\frac{\partial T_s}{\partial t}
-\nabla\cdot(k_s\nabla T_s)=Q_s,
$$

while the fluid energy equation includes advection,

$$
\rho_f c_f\left(\frac{\partial T_f}{\partial t}
+\mathbf u\cdot\nabla T_f\right)-\nabla\cdot(k_f\nabla T_f)=Q_f.
$$

At a perfect interface,

$$
T_f=T_s,\qquad
-k_f\nabla T_f\cdot\mathbf n_f
=k_s\nabla T_s\cdot\mathbf n_s=q_n.
$$

For finite conductance $h_c$, RunMat's current reduced interface component enforces

$$
q_n=h_c(T_f-T_s)
$$

on an interface graph, solves an anchored thermal network, and reports temperature-jump/flux-law,
network, connectivity-coverage, and mesh-backed-connectivity residuals. The surrounding fluid is
the reduced channel realization documented in [fluid flow](./fluid).

## Which conjugate heat-transfer procedures are defined?

| Procedure | Contract | Registered realization |
| --- | --- | --- |
| `conjugate_heat_transfer steady` | Static fluid/solid temperature and flow state | Experimental single-pass partitioned projection |
| `conjugate_heat_transfer transient` | Time samples and coupled initial state | Experimental fixed-point relaxation |
| `conjugate_heat_transfer nonlinear` | Load-factor domain and coupled residual | Experimental scalar residual iteration |

Every current branch uses the reduced fluid component and heuristic interface transfer. Canonical
Tet4/Tet10 topology can improve projection provenance, but does not turn the reduced fluid component
into a full three-dimensional CHT discretization.

## How do I declare a fluid–solid thermal interface?

```fea
model cooled_part {
    assign material.aluminum to geometry.cooler.region.solid
    assign material.air to geometry.cooler.region.fluid

    interface wall_transfer {
        conjugate_heat_transfer.conductance {
            primary = geometry.cooler.region.solid_wall
            secondary = geometry.cooler.region.fluid_wall
            coefficient = 5000 W/(m^2*K)
        }
    }
}

study cooler {
    geometry = geometry.cooler
    mesh = mesh.cooler
    model = model.cooled_part
    formulation = experimental
    analysis operating_point { conjugate_heat_transfer steady {} }
}
```

Use `conjugate_heat_transfer.perfect` only when temperature continuity is justified. Use
`conductance` or `resistance` when a coating, gap, or contact produces a finite jump. Primary and
secondary are typed, visible roles; proximity never creates an interface automatically.

## What else must the model contain?

Declare a well-posed fluid velocity/pressure problem, fluid inlet temperature, solid heat sources,
thermal material capabilities on both regions, and exterior losses. A CHT mesh must resolve fluid
velocity/thermal boundary layers and solid gradients, and interface projection must cover the same
physical surface on both sides.

## Why is my predicted cooling wrong?

Check inlet mass flow and temperature, fluid properties, solid conductivity, interface conductance,
contact area, boundary-layer resolution, turbulence/heat-transfer assumptions, heat-source power,
and global energy balance. For the current reduced realization, inspect whether topology/source/
boundary transfer used canonical projection or an ordinal/unmapped fallback.

See [heat transfer](./thermal), [fluid flow](./fluid),
[interfaces](../constraints/interfaces), [thermal loads](../loads/thermal), and
[current status](../status).
