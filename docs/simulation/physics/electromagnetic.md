# Electromagnetic FEA in RunMat: potentials, harmonic fields, currents, and losses

Electromagnetic analysis predicts electric potential, current flow, magnetic vector potential,
field intensity, flux, and frequency-dependent loss. RunMat uses distinct typed quantities for volts,
amperes, coulombs, current density, and vector potential; it does not pass them to a solver as
unitless scalar boundary values.

The `.fea` language can check static, transient, harmonic, and nonlinear electromagnetic procedure
contracts. The numerical library currently has one admitted electromagnetic realization: an
**experimental harmonic reduced edge-graph curl–curl solve** on canonical Tet4/Tet10 edge topology.
The typed `.fea` executor does not yet connect this family end to end, and no other procedure has an
admitted realization. This distinction is part of the API contract, not a documentation caveat.

## What electromagnetic equation is implemented in RunMat?

For time-harmonic magnetic vector potential $\mathbf A$ with angular frequency $\omega$, the target
curl–curl form is

$$
\nabla\times\left(\mu^{-1}\nabla\times\mathbf A\right)
+ i\omega\sigma\mathbf A - \omega^2\varepsilon\mathbf A
=\mathbf J_s,
$$

subject to vector-potential ground/gauge conditions and magnetic-insulation natural boundaries. The
implemented experimental realization builds oriented edge incidence from the canonical tetrahedral
mesh, assembles a reduced edge-graph curl–curl operator, applies frequency-dependent conductivity,
permittivity, and permeability terms, projects typed current/current-density sources to edges, and
solves the real/imaginary block system with BiCGSTAB.

In discrete form it has the structure

$$
\left(\mathsf K_{curl}-\omega^2\mathsf M_\varepsilon
+i\omega\mathsf M_\sigma\right)\mathbf a=\mathbf j.
$$

The realization evidence identifies `ReducedEdgeGraphCurlCurl`,
`ReducedFrequencyMaterialResponse`, the exact topology/source/boundary projection, and any fallback.
Ordinal, synthetic, or unmapped fallback realizations are experimental evidence outcomes and must
not be represented as full production Maxwell finite elements.

## Which electromagnetic procedures are defined?

| Procedure | Required independent domain/state | Registered realization |
| --- | --- | --- |
| `electromagnetic static` | Static sources and reference/gauge | None |
| `electromagnetic transient` | Explicit time samples and compatible initial field | None |
| `electromagnetic harmonic` | Ordered frequency samples and complex field state | Experimental reduced edge-graph curl–curl |
| `electromagnetic nonlinear` | Explicit load-factor domain | None |

The static, transient, and nonlinear rows are typed contracts only. A study requiring them fails
admission; RunMat does not reroute them through the harmonic reduced solver.

## How do I describe a harmonic electromagnetic study?

```fea
model conductor {
    assign material.copper to geometry.busbar.region.body

    constraint ground {
        electrical.potential {
            target = geometry.busbar.region.ground_terminal
            value = 0 V
        }
    }
    load drive_current {
        electric_current on geometry.busbar.region.input_terminal = 100 A
    }
}

study ac_busbar {
    geometry = geometry.busbar
    mesh = mesh.busbar
    model = model.conductor
    formulation = experimental
    analysis field {
        electromagnetic harmonic { frequencies = [50, 60, 400] Hz }
    }
}
```

This source is type-checkable when its geometry/material declarations are complete. It is not yet
runnable through the public typed `.fea` entrypoint. Frequency samples are ordered physical
frequencies and become part of result identity.

## How do I apply voltage, current, charge, or current density?

- `terminal_voltage` applies an electric-potential signal to a terminal region.
- `electric_current` applies a total current and preserves its resultant during projection.
- `electric_charge` applies a total charge.
- `current_density` applies amperes per square metre and preserves intensity.
- `magnetic_vector_potential` applies a vector-potential signal.

Use direct `load` syntax for constant data and `signal` plus `excitation` for frequency- or
time-dependent data. The excitation retains exact target, domain, shape, direction/frame where
applicable, projection rule, and active analysis/case membership.

```fea
signal coil_current : electric_current over frequency {
    samples = [
        { at = 50 Hz, value = 10 A },
        { at = 400 Hz, value = 8 A },
    ]
    interpolation = linear
    extrapolation = reject
}

excitation coil_drive {
    electric_current {
        target = geometry.motor.region.coil_terminal
        signal = signal.coil_current
    }
}
```

## What electrical and magnetic references are required?

Potential and vector-potential systems have nullspaces unless the physical problem supplies a
reference/gauge. RunMat exposes these explicitly:

- `electrical.potential` and `electrical.potential_reference` establish voltage data;
- `electrical.equipotential` represents one unknown potential across a conductor;
- `electrical.floating_conductor` keeps an isolated conductor ungrounded;
- `electrical.integral_current` constrains total current;
- `magnetic.vector_potential_ground` and `magnetic.coulomb_gauge` control the vector-potential nullspace;
- periodic and phase-shifted periodic constraints pair repeating boundaries;
- `magnetic.insulation` is a natural boundary.

A numerical penalty is not a substitute for declaring the physical reference. The experimental
low-level solver reports boundary anchor, localization, penalty contribution, conditioning, flux
divergence, source realization, and source-region coverage evidence.

## What material properties does electromagnetic analysis need?

The harmonic reduced realization consumes electrical conductivity, relative permittivity, relative
permeability, and frequency-dependent loss parameters from region-assigned material capabilities.
Every participating solid needs unambiguous material ownership. Conductivity alone does not imply
permeability, permittivity, or temperature dependence.

For Joule heating, the electromagnetic loss field must pass through an explicit electrothermal
coupling contract. It must not be recreated from a scalar “power” setting on the thermal side.

## Why does my electromagnetic solve fail or produce suspect fields?

Check for:

- absent/nonzero typed current or current-density source;
- no gauge or reference anchor;
- source or boundary regions that do not project to canonical edge/boundary topology;
- missing material coverage or nonphysical conductivity/permeability/permittivity;
- inadequate mesh resolution through skin depth and geometric gaps;
- frequency sampling too coarse near resonant or dispersive behavior;
- high complex equilibrium residual, flux-divergence error, or conditioning indicators;
- fallback source/boundary realization in the result evidence.

For a full Maxwell formulation, mesh spacing should be assessed against wavelength and skin depth,

$$
\lambda=\frac{2\pi}{\omega\sqrt{\mu\varepsilon}},\qquad
\delta=\sqrt{\frac{2}{\omega\mu\sigma}},
$$

but the current reduced realization must be qualified on its own evidence rather than assumed to
inherit production edge-element convergence properties.

See [electromagnetic constraints](../constraints/electromagnetic),
[electromagnetic loads](../loads/electromagnetic), [electrothermal analysis](./electrothermal),
[typed units](../language/types-units-and-expressions), and [current status](../status).
