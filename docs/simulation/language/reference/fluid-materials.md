# Incompressible fluid materials

A fluid material declares its law in the `fluid` block. A constant-viscosity
Newtonian fluid:

```fea
material water {
    mass_density = 1000 kg/m^3
    fluid {
        law = incompressible_newtonian
        dynamic_viscosity = 0.001 Pa*s
    }
}
```

Density has one authority: the material's shared `mass_density`. The fluid law
owns dynamic viscosity, not a second density. Both coefficients must be finite
and strictly positive. The law, viscosity, and density are required; no acoustic
property, imported metadata, default coefficient, or scalar inlet speed supplies
missing fluid physics.

Fluid material assignments use explicitly declared `cell` regions:

```fea
region volume : cell { select result; expect exactly_one }
// Inside the model:
assign material.water to geometry.channel.region.volume
```

Cell and Solid ownership remain distinct even when they refer to the same CAD
volume. Exact shared property compilation checks every canonical volume element
has one Cell material owner. The initial homogeneous material envelope requires
all owners to declare identical physical density and viscosity; material names
may differ. Missing, overlapping, unused, mixed Cell/Solid, and heterogeneous
ownership rejects. Coefficients are not averaged or selected by node order.

## Shear-thinning and yield-stress fluids

Generalized-Newtonian laws make the viscosity a function of the local shear rate
$\dot\gamma=\sqrt{2\mathbf D:\mathbf D}$, evaluated face by face from the solved velocity
gradient:

| `law` | Viscosity | Members |
| --- | --- | --- |
| `power_law` | $K\dot\gamma^{n-1}$ | `consistency` (Pa·s^n, SI number), `power_index`, `minimum_shear_rate` (default 1e-3 1/s) |
| `carreau` | $\mu_\infty+(\mu_0-\mu_\infty)[1+(\lambda\dot\gamma)^2]^{(n-1)/2}$ | `zero_shear_viscosity`, `infinite_shear_viscosity`, `relaxation_time`, `power_index` |
| `carreau_yasuda` | as Carreau with exponent $a$ | plus `transition_exponent` |
| `cross` | $\mu_\infty+(\mu_0-\mu_\infty)/(1+(\lambda\dot\gamma)^m)$ | `zero_shear_viscosity`, `infinite_shear_viscosity`, `time_constant`, `rate_index` |
| `bingham` | $\tau_y(1-e^{-m\dot\gamma})/\dot\gamma+\mu_p$ | `yield_stress`, `plastic_viscosity`, `regularization` (default 1000 s) |
| `herschel_bulkley` | $\tau_y(1-e^{-m\dot\gamma})/\dot\gamma+K\dot\gamma^{n-1}$ | `yield_stress`, `consistency`, `power_index`, `regularization` |
| `casson` | $(\sqrt{\tau_y(1-e^{-m\dot\gamma})/\dot\gamma}+\sqrt{\mu_p})^2$ | `yield_stress`, `plastic_viscosity`, `regularization` |

Blood as a Carreau fluid:

```fea
material blood {
    mass_density = 1060 kg/m^3
    fluid {
        law = carreau
        zero_shear_viscosity = 0.056 Pa*s
        infinite_shear_viscosity = 0.00345 Pa*s
        relaxation_time = 3.313 s
        power_index = 0.3568
    }
}
```

Yield-stress laws use Papanastasiou regularization, so unyielded regions flow with a large
but finite viscosity set by `regularization`; `minimum_shear_rate` holds a shear-thinning
viscosity finite where the fluid is at rest.


## Porous zones

A `porous` material describes a region that the surrounding fluid flows through: a filter
cartridge, a packed bed, a metal foam or a tube bank represented as a resistance. Assign it to
the porous region. The flowing fluid's density and viscosity come from the fluid material on the
rest of the domain, so the porous material carries only its resistance law.

| `law` | Momentum sink per unit volume | Members |
| --- | --- | --- |
| `darcy` | $\dfrac{\mu}{K}\,\mathbf u$ | `permeability` (m^2), `porosity` |
| `darcy_forchheimer` | $\left(\dfrac{\mu}{K}+\tfrac12\rho C_2\lvert\mathbf u\rvert\right)\mathbf u$ | `permeability` (m^2), `inertial_coefficient` (1/m), `porosity` |

Velocity in a porous zone is the superficial (Darcy) velocity, so mass flow is continuous
across the zone boundary. Uniform flow through a zone of length $L$ loses
$\Delta p = \left(\mu U/K + \tfrac12\rho C_2 U^2\right)L$. The sink is implicit in the momentum
equation and part of the solver's momentum-balance acceptance.

```fea
material pleated_media {
    porous {
        law = darcy_forchheimer
        permeability = 2e-11 m^2
        inertial_coefficient = 1.5e5 m^-1
        porosity = 0.85
    }
}
```

A material is either a fluid or a porous medium, not both.
