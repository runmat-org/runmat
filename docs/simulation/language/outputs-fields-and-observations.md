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
| Acoustic | `acoustic_pressure`, `acoustic_particle_velocity`, `acoustic_intensity`, `acoustic_port_intensity` (plane-wave power density through an incident or radiation port face), `sound_pressure_level` |
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
| `pressure_drop(step, from = region, to = region)` | Area-weighted mean fluid pressure on the `from` faces minus the mean on the `to` faces, in Pa, on a fluid step. |
| `transmission_loss(step, incident = region, transmitted = region)` | $10\log_{10}(W_\mathrm{inc}/W_\mathrm{tr})$ in dB per frequency on an acoustic harmonic step, from the incident and transmitted port powers. |
| `interpolate(field, at = point)`, `interpolate(field, at = [point, …])` | Sample a field at a point, or at an ordered list of points along a path. |
| `sample`, `envelope`, `absmax`, `variance` over `time` or `frequency` | Histories and spectra of a quantity. |
| `recover(field, to = <location>, using = <method>)` | Move a field to another location. |

Signatures and examples for every operation are in [operations and output queries](./reference/operations-and-output-queries.md).

## Sample a field at a point

```fea fragment
output top_stress = interpolate(analysis.static.field.node.von_mises_stress, at = [100 mm, 50 mm, 100 mm])
output tip_deflection = interpolate(analysis.static.field.node.displacement, at = [200 mm, 0 mm, 50 mm])
```

`interpolate(field, at = point)` returns the value at one point in model coordinates. For a nodal field it evaluates the shape functions of the element that contains the point, so the value follows the gradient to the point; compare this with a surface stress from beam or plate theory. For an element field it returns the containing element's value. A point on a shared face or edge reads one of the elements that touch it. A point outside the meshed solid is an error; it is never moved to the nearest surface. On a beam or shell mesh, the point is located on the containing line or surface element ([not yet in the current release](pending:output.interpolate-line-and-surface-meshes)).

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
    assessment deflection_limit {
        value = output.tip_deflection
        maximum = 0.5 mm
        evidence = authored_samples
    }
```

An assessment compares an output with a `minimum`, a `maximum` or both, in the output's unit, and publishes a pass or fail verdict with the run. `runmat reassess` evaluates revised assessments against a saved result without solving again. See [operations and output queries](./reference/operations-and-output-queries.md).

## Display a field

A `display` statement sets how result views show one field output: the snapshot posters, the saved scenes and the live result view.

```fea fragment
output von_mises = analysis.static.field.element.von_mises_stress
display von_mises {
    unit = MPa
    range = [0 MPa, 12 MPa]
    precision = 3
    hide = [geometry.arm.region.payload]
    context = [geometry.arm.region.stand]
    deformation = 200
}
```

| Member | Value | Effect |
| --- | --- | --- |
| `unit` | A unit of the field's quantity: `MPa`, `degC`, or a quoted compound unit such as `"kW/m^2"` | The legend's range, limit and notes are written in this unit. |
| `range` | Two values in the field's quantity, minimum below maximum: `[0 MPa, 12 MPa]` or `[0, 12] MPa` | The colour scale is fixed to this range. Values outside it take the end colours, and the legend states the field's true extremes beyond it, such as "▲ 14.2 MPa above the range". Two runs with the same range share one colour scale. |
| `precision` | An integer from 1 to 6 | The legend writes its numbers with this many significant digits. |
| `hide` | A list of solid regions of the study geometry: `[geometry.arm.region.payload]` | The view leaves these bodies out. Their values do not count toward the legend's extremes. |
| `context` | A list of solid regions of the study geometry | The view draws these bodies in translucent grey with no field colour. Their values do not count toward the legend's extremes. |
| `deformation` | A positive dimensionless number: `200` | The view draws the deformed shape with the displacement multiplied by this factor. Two runs with the same factor are exaggerated equally. Without it, each run chooses a scale from its own displacement. |

Each member is optional, and a `display` statement needs at least one. The named output must be a field output of the study. The checker refuses a unit or range in another quantity, a range whose minimum is not below its maximum, a precision outside 1 to 6, a deformation that is zero, negative or has a unit, and a `hide` or `context` entry that is not a solid region of the study geometry. A region may appear in `hide` or in `context`, not in both.

Each solved body is a solid region the model assigns a material to, named by its region name. Result scenes tag each body's surface with that name, so a viewer can hide or isolate a body after the run. The legend's extremes, its colour range and its overflow markers are taken over the bodies the view draws in field colour. For example, hiding a stiff payload stand-in leaves the arm's own peak on the bar instead of a value at the joint between the two bodies.

Without a `unit`, the legend uses the unit the study already writes that quantity in. The checker looks, in order, at the study's assessment bounds, its analysis blocks, and the loads, constraints, interfaces, excitations and initial conditions of its model. The first value of the field's quantity sets the unit. A thermal study whose model sets `reference_temperature = 20 degC` shows temperatures in °C, with or without a `display` statement. A study that writes no value of that quantity shows SI units. Without a `range`, the colour scale runs from the field's lowest to its highest value.

When the study assesses the shown quantity, the legend states the run's result first, then the limit, then the factor between them: "Peak 17.7 MPa · limit 184 MPa · factor 10.4". A minimum criterion reads "Lowest …". A temperature states its margin instead of a factor: "Peak 63.4 °C · limit 85 °C · margin 21.6 K".

The view of an assessed quantity colours the field the assessment reads, at the location the assessment reads it. A peak stress over element von Mises values is drawn element by element, so the top of the bar and the result on the assessment line are the same number when the view draws the assessed region. The line names the location: "Peak 76.3 MPa (element) · limit 184 MPa · factor 2.41". A view of the same quantity at another location names its location on the legend, such as "Von Mises stress, nodal average".

Each published assessment carries the same line in its `summary_line` field, next to the raw result and limit. Publication writes it in the unit and precision of the display of the field output the assessed value reads. Without such a display, it uses the unit the study writes that quantity in, and otherwise SI. Reports and the Desktop print this field, so the legend, a report and the Desktop show the same text for one run. An assessment with a function rule has no `summary_line`.

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
