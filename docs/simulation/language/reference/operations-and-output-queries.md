# Operations and output queries

FEA expressions resolve through RunMat's canonical operation catalog. One public operation name
may have numeric `.m` overloads and FEA field overloads, but both use one identity, documentation
entry, completion item, and lowering registry.

## Field operations

`magnitude` converts a typed vector field to a scalar field on the same domain. Complex fields
require an explicit representation such as magnitude, phase, real component, or imaginary
component before an operation that assumes real ordering or sign.

`component` selects a typed component. `recover` performs an explicit supported element-to-node
recovery and records the recovery method. Neither operation changes the underlying physical
quantity.

## Spatial reductions

`min` and `max` over a region return attributed extrema. `sum`, `mean`, integral, resultant, norm,
and moment operations enforce field location, physical quantity, geometric measure, weighting, and
reference-position rules. A boundary integral and a nodal sum are different overloads with
different evidence.

Named argument `over` selects the typed reduction domain. Moment operations require an explicit
reference point. Vector and tensor extrema are rejected until reduced to an ordered scalar
representation.

Two port operations read a pair of face regions on one step. `pressure_drop(step, from =
inlet, to = outlet)` returns the area-weighted mean fluid pressure on `from` minus the mean on
`to`, in Pa; it requires a fluid step. `transmission_loss(step, incident = inlet, transmitted =
outlet)` returns $10\log_{10}(W_\mathrm{inc}/W_\mathrm{tr})$ in dB at each frequency of an
acoustic harmonic step, where $W_\mathrm{inc}$ is the incident power at the `incident` port and
$W_\mathrm{tr}$ the power leaving through the `transmitted` port. The `incident` region must carry
an incident wave and the `transmitted` region a radiation (anechoic) boundary; a run with no
positive power at either port fails with that reason. Both operations refuse a region that is not
a face region.

```fea source=fluid-plane-poiseuille.fea
    output pressure_drop = pressure_drop(analysis.flow, from = geometry.channel.region.inlet, to = geometry.channel.region.outlet)
```

In this example the channel is 3 m long and its walls are 0.1 m apart (L/h = 30). The
fully developed drop is $12\mu U L/h^2 = 36$ Pa. The run reads within 5% of that value. The
uniform inlet adds an entrance loss of about 0.9 h of equivalent channel length, which is about
3% of the developed drop.

## Sampling and series

`sample` selects explicit positions on a field's independent domain. Interpolation requires one
point or a nonempty typed path. History and series reductions preserve sample coordinates,
weighting, event attribution, and source identity. Output policy blocks bound retained samples and
storage.

## Sweep operations

`collect(case.output.<name>)` retains compact case values. `retain_each` keeps each referenced
output object. Case `min` and `max` retain the governing case coordinates. Sweep operations accept
only outputs declared by the source study.

## Acceptance assessments

An `assessment` inside a study compares a declared real scalar output with a
dimensionally compatible bound. Use `maximum`, `minimum`, or both for an inclusive
range. The evidence policy is required.

```fea
output spatial_peak = max(
    magnitude(analysis.response.field.node.displacement),
    over = geometry.bracket.region.body,
)
output peak_history = sample(
    output.spatial_peak,
    over = time,
    at = [0 s, 0.05 s, 0.1 s],
)
output peak = max(output.peak_history, over = time)
assessment deflection {
    value = output.peak
    maximum = 1 mm
    evidence = authored_samples
}
```

This example checks the three declared times. Choose sampling coordinates that
cover the engineering requirement; three samples alone do not establish the peak
throughout a transient. Spatial reduction occurs before sampling, and the time
reduction preserves the governing sample and entity.

`authored_samples` compares the declared output when its numerical result is
acceptable. `certified_bound` additionally requires a certified absolute error
bound. Unavailable evidence or uncertainty crossing the limit produces an
indeterminate result. Acceptance status is separate from solver convergence.

Study setup should preserve a user's criterion while checking output coverage and
available execution evidence. A final-time output cannot answer a peak-over-time
requirement. Do not replace such a requirement with a final-time comparison or
claim continuous coverage from a sampled maximum.

### Imported assessment functions

Use an imported function when a comparison needs custom logic. The function receives
the scalar output first, followed by the declared parameters. Values are passed in
canonical SI units: `1 mm` is passed as `0.001`, for example. A function assessment
cannot also declare `minimum` or `maximum`.

```fea
import function within_limit from "./within_limit.m"

// Inside a study, after declaring output.peak:
assessment deflection {
    value = output.peak
    function = function.within_limit
    parameters = [1 mm]
    evidence = authored_samples
}
```

```matlab
function result = within_limit(value, limit)
arguments
    value (1,1) double
    limit (1,1) double
end
result = struct('evaluated', true, 'accepted', value <= limit);
end
```

The checked function must have a complete signature with real scalar inputs and a
scalar record containing exactly two logical fields, `evaluated` and `accepted`.
It cannot capture workspace values or require ambient capabilities such as file
access or randomness. The executable must match the frozen imported source.

`evaluated = false` produces an indeterminate verdict. A thrown exception or an
invalid return also produces an indeterminate verdict, with an execution diagnostic.
A function that has not run is unavailable, not failed. The logical return does not
certify an error bound or establish coverage between authored samples. Checking the
source establishes its language contract; execution still requires a host that
admits the portable function and its bounded task resources.

### Reassessing saved results

To compare existing results against revised criteria, save the revised assessment
declarations and reassess the original result publication. This produces a separate
immutable assessment report. It does not solve again, replace the original report,
or apply revised geometry, materials or loads to the old numerical result.

```sh
runmat reassess bracket.fea --study bracket \
    --original-definition sha256:<original-definition-digest> \
    --result-manifest <saved-study-result-manifest> --json
```

Use the original definition digest and content-addressed manifest recorded by the
solve. Changed bounds or compatible custom assessment functions can reuse that
publication. Each referenced output must retain its original quantity, region,
sampling and producer. Changing an output while keeping its name requires new
numerical evidence. Imported functions still require checked frozen sources and
bounded executable admission. Missing evidence remains indeterminate.

## Diagnostics

Operation errors report the resolved argument types, physical dimensions, locations, available
overloads, and invalid named argument. There is no fallback to a string-dispatched solver helper.
An operation is not part of the language unless its catalog descriptor, type checker, lowering,
editor metadata, and documentation are all present.

## Examples

Field operations compose inside an output expression while preserving quantity and location:

```fea
output displacement = analysis.solve.field.node.displacement
output displacement_magnitude = magnitude(output.displacement)
output peak_displacement = max(
    output.displacement_magnitude,
    over = geometry.bracket.region.body,
)
```

Use a block when an output also needs retention policy:

```fea
output tip_history {
    value = sample(
        source = output.tip_displacement,
        over = time,
        at = [0 s, 0.5 s, 1 s],
    )
    precision = real64
    compression = lossless
    retention = explicit_samples
    maximum_samples = 2000
    partial_access = manifest_and_chunks
    purpose = publication
}
```

Sweep expressions consume outputs declared by the source study:

```fea
sweep thickness_study over study.static_check {
    axis wall_thickness {
        thin = 2 mm
        nominal = 3 mm
        thick = 4 mm
    }
    output peak_by_case = collect case.output.peak_stress.value
    output governing_case = max(case.output.peak_stress.value, over = cases)
    output stress_fields = retain_each case.output.stress
}
```
