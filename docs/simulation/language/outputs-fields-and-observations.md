# Outputs, fields and observations

An analysis publishes its canonical terminal result and evidence. `output` declarations request
additional fields, reductions, histories, or retained intermediate state with stable names.

## Select a field

```fea
output displacement = analysis.static.field.node.displacement
output von_mises = analysis.static.field.element.von_mises_stress
```

The reference identifies the producing analysis, field location, and physical quantity. Invalid
family, procedure, state, location, or field combinations fail checking. A name that happens to
look like a field is not resolved at runtime.

## Transform and reduce

```fea
output displacement_magnitude = magnitude(output.displacement)
output peak_displacement = max(
    output.displacement_magnitude,
    over = geometry.bracket.region.body,
)
```

Output operations are typed overloads in RunMat's canonical operation catalog. `magnitude`
preserves the field domain and changes the component shape. A regional `max` returns an attributed
extremum: the value remains linked to the entity, position, sample, analysis, and source field that
produced it.

`flux(field, through = region)` integrates the outward normal component of a vector field over a
face region, $\int\mathbf v\cdot\mathbf n\,dA$. For a harmonic acoustic analysis,
`sound_power(analysis.sound, through = region)` is the flux of its active intensity
$\tfrac12\mathrm{Re}(p\,\mathbf v^*)$, the mean acoustic power leaving through the faces:

```fea
output radiated_power = sound_power(analysis.sound, through = geometry.speaker.region.hemisphere)
output intensity = analysis.sound.field.element.acoustic_intensity
```

## Sample a field at a point

```fea
output top_stress = interpolate(
    analysis.static.field.element.von_mises_stress,
    at = [100 mm, 50 mm, 100 mm],
)
output tip_deflection = interpolate(analysis.static.field.displacement, at = [200 mm, 0 mm, 50 mm])
```

`interpolate(field, at = point)` returns the field value at one point in model coordinates. For a
nodal field, RunMat evaluates the shape functions of the element that contains the point. For an
element field, it returns the native value of the element that contains the point. A point on a
shared face or edge reads one of the elements that touch it. A point outside the meshed solid is an
observation error; RunMat does not clamp it to the nearest surface.

`analysis.static.field.node.von_mises_stress` is the nodal von Mises field: the volume-weighted
average of the element values around each node, the same field `recover(...,
to = node, using = volume_weighted_average)` names. Interpolating it gives a value that follows
the stress gradient to the sample point, which is the comparison to make against a surface stress
from beam or plate theory; the element field gives the containing element's average.

This is intentionally separate from rendering. After a run, ordinary RunMat code selects the
field and plots it:

```matlab
results = fea.results(run);
stress = fea.field(results, fea.StructuralField.vonMises());
plot(stress);
```

`plot` does not choose an FEA field by string. The same ordinary `mesh(...)` entrypoint displays
solver or CAD-reference meshes without changing numeric MATLAB-compatible `mesh` behavior.

## Output policy blocks

Use a block when an output needs sampling or retention policy in addition to its value:

```fea
output tip_history {
    value = sample(
        analysis.response.field.node.displacement,
        over = time,
        at = [0 s, 10 ms, 20 ms],
    )
    retention = explicit_samples
    maximum_samples = 3
}
```

The checker verifies that the sampling domain belongs to the source field, that sample locations
have the right type and units, and that the declared bound can contain the requested result.

## Large results

Fields and histories are content-addressed, partitioned result objects. Publication keeps study,
analysis, field, sample, mesh, workload, and numerical-receipt identity on every manifest and page.
Selecting one field or sample does not require loading unrelated payloads into the driver process.

Distributed reductions operate on typed partitions and use deterministic fan-in. The published
evidence distinguishes a partition-local reduction from a gathered one and records the placement
and accelerator facts that actually occurred.

## Sweep results

Inside a sweep, `case.output.<name>` refers to a declared output of the source study. Attributed
extrema expose `.value` when only the physical scalar is needed. See
[Inputs, sweeps, and cases](./inputs-sweeps-and-cases) for `collect`, case extrema, and
`retain_each`.
