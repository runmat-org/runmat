# Constant incompressible Newtonian materials

The shared authored material contract can explicitly declare a constant
incompressible Newtonian law:

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

This material prerequisite is not CFD validation, native execution
admission, a turbulence closure, transient flow, or coupled CHT/FSI support.
Material compilation represents the explicit laminar envelope with zero
turbulence intensity in the existing coefficient carrier; it does not infer a
velocity, characteristic length, pressure profile, or turbulence model.
