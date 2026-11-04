# Conjugate heat-transfer analysis in RunMat: fluid cooling and solid conduction

Conjugate heat transfer (CHT) solves the flow and temperature of a coolant together with
conduction in the part it cools. Fluid and solid keep their own materials and equations; a
declared interface carries temperature and heat flux across the wetted surface, so the film
coefficient comes from the flow instead of a guess.

`runmat run` executes steady conjugate heat transfer from a `.fea` study. The flow is the canonical
cell-centred finite-volume solution of the [fluid flow](./fluid) analysis on the fluid regions.
One nodal energy equation then spans every fluid and solid element: Galerkin conduction in both,
conservative advection by the solved flow in the fluid, and streamline-upwind stabilization where
the flow dominates conduction. Fluid and solid are coupled inside that single linear system, so
the interface balance holds to the solver tolerance at every node.

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

A finite conductance $h_c$ (or contact resistance $R = 1/h_c$) allows a jump,
$q_n=h_c(T_s-T_f)$.

## How does RunMat discretize the coupled energy equation?

- The solved finite-volume face fluxes are lifted into every fluid tetrahedron as its
  lowest-order Raviart–Thomas velocity. That velocity reproduces each face flux, is exactly
  divergence-free in a mass-conserving cell and has continuous normal flux between cells, so the
  summed advection rows equal the enthalpy carried through the boundary: transport neither
  creates nor loses energy.
- A perfect interface shares the conformal interface nodes. A conductance or resistance gives
  each interface node a second, fluid-side temperature joined to the solid side by
  $h_c\int(T_s-T_f)N_i\,dA$, which removes from one side exactly the heat it adds to the other.
- Flow entering the fluid must carry an authored inlet temperature; outflow leaves with its
  own temperature; walls without a thermal law are adiabatic.

## Which conjugate heat-transfer procedures are defined?

| Procedure | Contract | Execution |
| --- | --- | --- |
| `conjugate_heat_transfer steady` | Steady flow and steady temperature in fluid and solid | Runs from `runmat run` |
| `conjugate_heat_transfer transient` | Steady flow advecting a backward-Euler temperature march from the authored initial temperature | Runs from `runmat run` |
| `conjugate_heat_transfer nonlinear` | Temperature-dependent properties | Typed contract; not yet executed |

## How do I declare a fluid–solid thermal interface?

```fea
model cooled_part {
    assign material.aluminum to geometry.cooler.region.solid
    assign material.water to geometry.cooler.region.fluid

    constraint wetted { fluid.no_slip_wall { target = geometry.cooler.region.fluid_wall } }
    interface wall_transfer {
        conjugate_heat_transfer.conductance {
            primary = geometry.cooler.region.solid_wall
            secondary = geometry.cooler.region.fluid_wall
            coefficient = 5000 W/m^2/K
        }
    }
}

study cooler {
    geometry = geometry.cooler
    mesh = mesh.cooler
    model = model.cooled_part
    analysis operating_point { conjugate_heat_transfer steady {} }
    output temperature = analysis.operating_point.field.node.temperature
    output velocity = analysis.operating_point.field.cell.fluid_velocity
}
```

Use `conjugate_heat_transfer.perfect` where the coolant wets bare metal, `conductance` for a
coating and `resistance` for a gap pad. Primary and secondary are typed, visible roles;
proximity never creates an interface automatically, and every face where a fluid meets a solid
must belong to a declared interface. The fluid side of the wetted wall still needs its flow law,
usually `fluid.no_slip_wall`.

## What does a conjugate run publish?

- `field.node.temperature`: the solved nodal temperature over fluid and solid. Where an
  interface carries a jump, the node value is the solid side; the fluid side is retained with
  the result.
- `field.cell.fluid_velocity`, `field.cell.fluid_pressure` and the face volume flux of the fluid.
- Evidence for every interface: the heat entering the fluid and the solid through it (equal and
  opposite), its relative imbalance and, for a finite conductance, the largest temperature jump
  and the residual of $q=h_c\Delta T$.
- Global energy closure: applied heat, convective losses, enthalpy carried out of the fluid and
  heat drawn at prescribed temperatures, with their relative imbalance.

## Why is my predicted cooling wrong?

Check inlet flow and temperature, fluid properties, solid conductivity, interface law, contact
area, boundary-layer resolution across the passage, heat-source power, and the reported energy
closure. Laminar flow only: turbulent coolant is not modelled.

See [heat transfer](./thermal), [fluid flow](./fluid),
[interfaces](../constraints/interfaces), [thermal loads](../loads/thermal), and
[current status](../validation/status.md).
