# Fluid–structure

Fluid–structure interaction (FSI) transfers fluid traction to a structure and structural wall
motion back to the fluid. The paired interface is explicit; matching-looking CAD faces do not
create coupling.

`fea run` executes steady and transient fluid–structure studies. The fluid side is the canonical
finite-volume flow of the [fluid flow](./fluid) analysis on the deformed fluid domain; the
structure is a small-strain linear elastic solid on the solid regions of the same conformal
mesh. The two sides are iterated to agreement in every solve or time interval.

## What interface equations does FSI enforce?

At a no-slip fluid–structure interface $\Gamma_{fs}$, the continuum conditions are

$$
\mathbf u_f=\dot{\mathbf d}_s,
\qquad
\boldsymbol\sigma_f\mathbf n_f
+\boldsymbol\sigma_s\mathbf n_s=\mathbf0.
$$

RunMat applies them as follows:

- **Traction to the structure.** The pressure and viscous force of every wetted finite-volume
  face, negated, is shared between the face's structural nodes by its consistent shape-function
  weights. The structure receives exactly the force the wall exerts on the fluid.
- **Motion to the fluid.** The fluid interior follows the wall through a volume-weighted Laplace
  extension of the interface displacement, and the flow is solved on that deformed domain. In a
  transient study every face sweeps an exactly integrated volume between configurations, so cell
  volume changes satisfy the geometric conservation law, and the wetted wall moves with the
  structure's velocity while carrying exactly its swept flux.
- **Agreement.** Each coupling iteration solves the flow, loads and solves the structure, and
  relaxes the interface displacement with Aitken's dynamic factor. Iteration stops when the
  structure's response to the flow reproduces the displacement that shaped it, to the coupling
  tolerance.

## Which FSI procedures are defined?

| Procedure | Contract | Execution |
| --- | --- | --- |
| `fluid_structure_interaction steady` | Equilibrium deflection under steady flow and the flow through the deflected passage | Runs from `fea run` |
| `fluid_structure_interaction transient` | Moving-mesh backward-Euler flow and average-acceleration Newmark structure, converged every interval | Runs from `fea run` |
| `fluid_structure_interaction nonlinear` | Load-factor domain | Typed contract; not yet executed |

## How do I declare an FSI interface?

```fea
model flexible_wall {
    assign material.water to geometry.duct.region.water
    assign material.rubber to geometry.duct.region.wall
    constraint inflow {
        fluid.velocity { target = geometry.duct.region.inlet; value = [0.01 m/s, 0 m/s, 0 m/s] }
    }
    load outflow {
        fluid.outlet_pressure { target = geometry.duct.region.outlet; value = 0 Pa }
    }
    constraint mounted {
        structural.fixed { target = geometry.duct.region.mount }
    }
    interface coupled_wall {
        fluid_structure.no_slip {
            primary = geometry.duct.region.fluid_wall
            secondary = geometry.duct.region.structural_wall
        }
    }
}

study flexible_duct {
    geometry = geometry.duct
    mesh = mesh.duct
    model = model.flexible_wall
    analysis response {
        fluid_structure_interaction transient {
            start_time = 0 s
            end_time = 1 s
            time_step = 1 ms
        }
    }
    output wall_displacement = analysis.response.field.node.displacement
    output pressure = analysis.response.field.cell.fluid_pressure
}
```

The declared interface is the fluid's wall: it imposes no slip relative to the moving structure,
so the fluid needs no separate wall law there. Every face where the fluid meets the structure must
belong to a declared interface. The structure needs a fixed support (`structural.fixed`); axis-
aligned symmetry planes (`structural.symmetry`) are also admitted.

## What does an FSI run publish?

- `field.node.displacement`: structural displacement on the solid and the mesh motion of the
  fluid, on every node of the reference mesh.
- `field.cell.fluid_velocity` and `field.cell.fluid_pressure` on the deformed fluid domain, in the
  reference cell order.
- Interface evidence: every coupling iteration's displacement residual, the resultant fluid force
  on the structure, the work of the interface traction and the structural strain energy (a linear
  structure loaded at the interface stores half of that work in a steady study), and the
  equilibrium residuals of the structure and the mesh motion. A transient study reports, for every
  accepted interval, its coupling iterations and residual, the cumulative interface work, the
  strain and kinetic energy (average acceleration makes their sum equal the work), and the
  structure's dynamic residual.

## How do I choose coupling time step and convergence tolerance?

The time step must resolve both fluid transport and structural dynamics. Added-mass-dominated
cases (a light wall in a dense fluid) couple strongly and need more coupling iterations per step;
Aitken relaxation adapts the factor each iteration. The structure is small-strain: large rotations
are outside the envelope.

See [fluid flow](./fluid), [structural analysis](./structural),
[fluid constraints](../constraints/fluid), [structural constraints](../constraints/structural),
[interfaces](../constraints/interfaces), and [current status](../status).
