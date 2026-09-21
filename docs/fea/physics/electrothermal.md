# Electrothermal FEA in RunMat: current flow, Joule heating, and temperature

Electrothermal analysis couples electrical conduction to heat transfer. Electrical loss becomes a
volumetric thermal source; temperature may change conductivity and therefore feed back into the
electrical field.

RunMat has typed electrothermal procedure, material, state, and output contracts, but **no admitted
electrothermal numerical realization** and no public typed `.fea` execution path today. The existing
library helper computes only a bounded coupling-severity indicator from voltage, conductivity, and a
heating coefficient. It is not an electrothermal solve.

## Which electrothermal procedures are defined?

| Procedure | Contract | Registered realization |
| --- | --- | --- |
| `electrothermal steady` | Static electrical and thermal coupled state | None |
| `electrothermal transient` | Time samples and coupled initial state | None |
| `electrothermal nonlinear` | Load-factor domain and nonlinear coupled state | None |

All three are authoring and type-checking contracts. Admission fails for execution rather than
selecting an unrelated electromagnetic or thermal component.

## What equations must an electrothermal realization solve?

For electric potential $V$ and conductivity $\sigma(T)$,

$$
\nabla\cdot(\sigma(T)\nabla V)=0,
\qquad
\mathbf J=-\sigma(T)\nabla V,
$$

with Joule heating

$$
Q_J=\mathbf J\cdot\mathbf E
=\sigma(T)|\nabla V|^2,
$$

coupled to

$$
\rho c_p\frac{\partial T}{\partial t}
-\nabla\cdot(k(T)\nabla T)=Q_J+Q_{other}.
$$

A production implementation must state whether the coupling is one-way, staggered fixed-point, or
monolithic; preserve power during electrical-to-thermal projection; converge both electrical and
thermal residuals; and publish charge and energy balance. RunMat will not treat a convergence
severity heuristic as this formulation.

## How do I author an electrothermal study contract?

```fea
model heated_conductor {
    assign material.copper to geometry.busbar.region.body
    constraint ground {
        electrical.potential {
            target = geometry.busbar.region.ground_terminal
            value = 0 V
        }
    }
    load current {
        electric_current on geometry.busbar.region.input_terminal = 100 A
    }
    load cooling {
        thermal.convection {
            target = geometry.busbar.region.exterior
            ambient_temperature = 298.15 K
            coefficient = 20 W/(m^2*K)
        }
    }
}

study heated_busbar {
    geometry = geometry.busbar
    mesh = mesh.busbar
    model = model.heated_conductor
    formulation = experimental
    analysis operating_point { electrothermal steady {} }
}
```

This declares intent and can be statically checked when the complete material/geometry contract is
present. `runmat fea run` must reject it until a matching realization is admitted.

## What material and boundary data are required?

Every conductor needs conductivity, thermal conductivity, density, and specific heat; temperature-
dependent conductivity must state its interpolation/range. Electrical terminals need a well-posed
voltage/current/reference set. Thermal boundaries need explicit sinks, convection, flux, or contact
resistance. Electrical contact resistance and thermal contact resistance are distinct quantities.

## How should I validate electrothermal results?

Require electrical power balance $P=\int\mathbf J\cdot\mathbf E\,d\Omega$ to match the projected
thermal source, and require thermal energy balance against boundary loss and storage. Check mesh
resolution at current crowding/hot spots, temperature-dependent property sensitivity, contact
resistance, and convergence of the coupled iteration.

See [electromagnetics](./electromagnetic), [heat transfer](./thermal),
[electromagnetic loads](../loads/electromagnetic), [thermal loads](../loads/thermal), and
[current status](../status).
