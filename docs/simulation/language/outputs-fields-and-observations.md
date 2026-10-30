---
description: How a .fea study names result fields by step, location and quantity, reduces them with max, mean, flux and interpolate, and publishes outputs, histories and assessments.
keywords: [FEA outputs, result fields, von Mises stress, nodal displacement, interpolate, flux, sound power, field quantities, field locations]
doc_type: language-reference
examples: [cantilever-static]
---
# Outputs, fields and observations

A study publishes what its `output` declarations ask for: whole fields, reduced quantities such as a peak stress over a region, values sampled at a point, and histories over time or frequency. Each output has a stable name, a type and a unit, and assessments turn outputs into pass or fail verdicts.

## Name a field

```fea fragment
output displacement = analysis.static.field.node.displacement
output von_mises = analysis.static.field.element.von_mises_stress
```

A field reference has four parts, `analysis.<step>.field.<location>.<quantity>`: the producing step, where the values live, and the physical quantity. The checker rejects a location or quantity the step's family and procedure do not produce, so a typo is an error at its line, not an empty result after the solve.

**Locations:**

| Location | Values at |
| --- | --- |
| `node` | Mesh nodes; element values are averaged to nodes where the quantity is recovered (stress, flux). |
| `element` | One value per element (the element average). |
| `integration_point` | Quadrature points, where stresses and fluxes are computed. |
| `cell` | Finite-volume cells (fluid). |
| `face`, `edge` | Mesh faces and edges, such as tractions on a boundary. |
| `interface` | Contact and coupled interfaces. |
| `connector`, `constraint`, `reference_point` | Connector forces, constraint reactions and reference-point results. |
| `mode` | One value per mode: frequencies, participation. As a study `output` [not yet in the current release](pending:output.mode-location). |
| `global` | One value for the whole step: residuals, balances. |

**Quantities, by family:**

| Family | Quantities |
| --- | --- |
| Structural | `displacement`, `rotation`, `velocity`, `acceleration`, `strain`, `stress`, `von_mises_stress`, `reaction_force`, `reaction_moment`, `strain_energy` |
| Beams and shells | `beam_axial_force`, `beam_shear_force`, `beam_bending_moment`, `beam_torsion_moment`, `beam_bending_stress`, `beam_torsion_stress`, `shell_membrane_force`, `shell_bending_moment`, `shell_transverse_shear`, `shell_von_mises` |
| Contact | `contact_gap`, `contact_pressure` |
| Thermal | `temperature`, `temperature_gradient`, `heat_flux`, `heat_rate`, `thermal_energy_storage_rate_density` |
| Electric | `electric_potential`, `electric_field`, `electric_displacement`, `electric_energy_density`, `electric_current`, `electric_current_density`, `electric_charge`, `joule_power_density`, `electric_edge_circulation` |
| Magnetic | `magnetic_vector_potential`, `magnetic_flux_density`, `magnetic_field_intensity`, `magnetic_edge_circulation` |
| Acoustic | `acoustic_pressure`, `acoustic_particle_velocity`, `acoustic_intensity`, `sound_pressure_level` |
| Fluid | `fluid_pressure`, `fluid_velocity`, `vorticity`, `wall_shear_stress`, `mass_flow_rate`, `volume_flow_rate` |
| Modal | `modal_frequency`, `modal_participation`, `eigenvalue` |
| Solver checks | `residual`, `balance`, `constraint_violation` |

`analysis.static.field.node.von_mises_stress` is the nodal von Mises stress: at each node, the volume-weighted average of the surrounding elements' stress tensors, with von Mises taken from the averaged tensor. `recover(…, to = node, using = volume_weighted_average)` names the same recovery explicitly. The element field (`field.element.von_mises_stress`) is each element's own value.

## Reduce a field

```fea source=cantilever-static.fea
    output tip_deflection = max(magnitude(analysis.static.field.node.displacement), over = geometry.beam.region.loaded_end)
```

Output operations are typed: each one accepts certain field locations, quantities and shapes and returns a typed result. `magnitude` keeps the field's locations and turns a vector into a scalar; `max` over a region returns an attributed extremum, whose value stays linked to the entity, position, step and field it came from (`.value` gives the scalar alone).

| Operations | Purpose |
| --- | --- |
| `max`, `min`, `mean`, `rms`, `integral`, `resultant`, `moment`, `norm` | Reduce a field over a region. |
| `magnitude`, `component`, `principal`, `von_mises` | Transform a field pointwise. |
| `real`, `imag`, `phase`, `level` | Read a complex (harmonic) result; `level` gives a decibel level. |
| `flux(field, through = region)` | $\int_\Gamma \mathbf v\cdot\mathbf n\,dA$ of a vector field over a face region. |
| `sound_power(step, through = region)` | The flux of active acoustic intensity $\tfrac12\mathrm{Re}(p\,\mathbf v^*)$: mean power leaving through the faces. |
| `interpolate(field, at = point)`, `interpolate(field, at = [point, …])` | Sample a field at a point, or at an ordered list of points along a path. |
| `sample`, `envelope`, `absmax`, `variance` over `time` or `frequency` | Histories and spectra of a quantity. |
| `recover(field, to = <location>, using = <method>)` | Move a field to another location. |

Signatures and examples for every operation are in [operations and output queries](./reference/operations-and-output-queries.md).

## Sample a field at a point

```fea fragment
output top_stress = interpolate(analysis.static.field.node.von_mises_stress, at = [100 mm, 50 mm, 100 mm])
output tip_deflection = interpolate(analysis.static.field.node.displacement, at = [200 mm, 0 mm, 50 mm])
```

`interpolate(field, at = point)` returns the value at one point in model coordinates. For a nodal field it evaluates the shape functions of the element that contains the point, so the value follows the gradient to the point; compare this with a surface stress from beam or plate theory. For an element field it returns the containing element's value. A point on a shared face or edge reads one of the elements that touch it. A point outside the meshed solid is an error; it is never moved to the nearest surface.

## Histories and retention

Use an output block when an output also needs a sampling or retention policy:

```fea fragment
output tip_history {
    value = sample(analysis.response.field.node.displacement, over = time, at = [0 s, 10 ms, 20 ms])
    retention = explicit_samples
    maximum_samples = 3
}
```

The checker verifies that the sampling domain belongs to the field (time for a transient step, frequency for a harmonic step), that sample locations have the right type and unit, and that the declared bound holds the requested samples.

## Assess an output

```fea source=cantilever-static.fea
    assessment tip_deflection_matches_beam_theory {
        value = output.tip_deflection
        minimum = 0.19 mm
        maximum = 0.21 mm
        evidence = authored_samples
    }
```

An assessment compares an output with a `minimum`, a `maximum` or both, in the output's unit, and publishes a pass or fail verdict with the run. `runmat reassess` evaluates revised assessments against a saved result without solving again. See [operations and output queries](./reference/operations-and-output-queries.md).

## Read results in `.m` code

Fields are published separately from rendering. After a run, `.m` code loads the result, selects a field and plots or post-processes it:

```matlab
results = fea.results(run);
stress = fea.field(results, fea.StructuralField.vonMises());
plot(stress);
```

`plot` takes the field object, not a field name string.

## Large results

Fields and histories are content-addressed, partitioned result objects. Each part records its study, step, field, sample and mesh identity, so reading one field or one sample loads only that data. Distributed reductions combine typed partitions in a fixed order, so a reduction gives the same value however the work was split, and the result records where each part ran.

## Outputs across a sweep

Inside a sweep, `case.output.<name>` refers to one case's output of the study. `collect`, case `min` and `max`, and `retain_each` gather outputs across cases; see [inputs, sweeps and cases](./inputs-sweeps-and-cases.md).
