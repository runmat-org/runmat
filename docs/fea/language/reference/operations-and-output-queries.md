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

## Sampling and series

`sample` selects explicit positions on a field's independent domain. Interpolation requires one
point or a nonempty typed path. History and series reductions preserve sample coordinates,
weighting, event attribution, and source identity. Output policy blocks bound retained samples and
storage.

## Sweep operations

`collect(case.output.<name>)` retains compact case values. `retain_each` keeps each referenced
output object. Case `min` and `max` retain the governing case coordinates. Sweep operations accept
only outputs declared by the source study.

## Diagnostics

Operation errors report the resolved argument types, physical dimensions, locations, available
overloads, and invalid named argument. There is no fallback to a string-dispatched solver helper.
An operation is not part of the language unless its catalog descriptor, type checker, lowering,
editor metadata, and documentation are all present.
