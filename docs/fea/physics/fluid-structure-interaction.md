# Fluid–structure interaction in RunMat: pressure, wall motion, and partitioned coupling

Fluid–structure interaction (FSI) transfers fluid traction to a structure and structural wall motion
back to the fluid. The paired interface is explicit; matching-looking CAD faces do not create
coupling.

The `.fea` language checks steady, transient, and nonlinear FSI contracts. The numerical library
currently admits an experimental **partitioned reduced-flow/reduced-structure relaxation** with a
heuristic interface transfer. The public typed `.fea` executor does not connect it end to end. It is
not a production arbitrary-geometry FSI formulation.

## What interface equations does FSI enforce?

At a no-slip fluid–structure interface $\Gamma_{fs}$, the continuum conditions are

$$
\mathbf u_f=\dot{\mathbf d}_s,
\qquad
\boldsymbol\sigma_f\mathbf n_f
+\boldsymbol\sigma_s\mathbf n_s=\mathbf0.
$$

The current reduced component samples fluid pressure onto interface faces, solves a graph structural
response $\mathsf K_\Gamma\mathbf d_\Gamma=\mathbf p_\Gamma$, feeds displacement back through
a compliance-derived pressure law, and under-relaxes pressure, traction, and displacement with a
factor of $0.65$. It iterates until pressure-feedback, displacement-transfer, traction-update, and
structural-solve residuals meet tolerance or the iteration cap is reached.

Evidence includes interface connectivity coverage, mesh-backed topology coverage, interface work,
structural strain energy, pressure–displacement law residual, and the combined two-way residual.

## Which FSI procedures are defined?

| Procedure | Contract | Registered realization |
| --- | --- | --- |
| `fluid_structure_interaction steady` | Static fluid and structural interface state | Experimental single-pass partitioned transfer |
| `fluid_structure_interaction transient` | Time samples and coupled initial state | Experimental fixed-point relaxation |
| `fluid_structure_interaction nonlinear` | Load-factor domain and coupled residual | Experimental scalar residual iteration |

The structural side of these registered branches is a reduced scalar graph response, not the full
tetrahedral structural operator used by the structural executor.

## How do I declare an FSI interface?

```fea
model flexible_wall {
    assign material.air to geometry.duct.region.fluid
    assign material.steel to geometry.duct.region.structure

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
    formulation = experimental
    analysis response {
        fluid_structure_interaction transient {
            start_time = 0 s
            end_time = 1 s
            time_step = 1 ms
        }
    }
}
```

The complete model also needs fluid inlet/outlet/wall data, structural supports and material
properties, and compatible initial states. The example declares the interface and procedure but is
not runnable through today's public `.fea` executor.

## How do I choose coupling time step and convergence tolerance?

The time step must resolve both fluid transport and structural dynamics. A partitioned scheme must
converge interface traction and displacement within every accepted time step; convergence of only
one subsolver is insufficient. Added-mass-dominated cases can require stronger coupling or a
monolithic method. The current reduced relaxation is fixed and should be qualified only against its
reported residuals.

## Why does an FSI solve oscillate or fail to converge?

Check inconsistent interface normals/coverage, nonconservative pressure projection, insufficient
structural restraint, unrealistic compliance, incompatible initial states, time steps that are too
large, and added-mass instability. Require traction balance, kinematic continuity, interface work
consistency, mass balance, structural equilibrium, and mesh/time-step convergence.

See [fluid flow](./fluid), [structural analysis](./structural),
[fluid constraints](../constraints/fluid), [structural constraints](../constraints/structural),
[interfaces](../constraints/interfaces), and [current status](../status).
