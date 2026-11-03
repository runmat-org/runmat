# Types, units and expressions

The checker assigns a type to every input, reference, literal, and operation before a study can be
planned. Physical quantities carry dimensions, so a length cannot satisfy a force input merely
because both are numbers.

## Scalar and physical types

Common scalar types are `boolean`, `integer`, `real`, and `string`. Physical types include
`length`, `time`, `frequency`, `mass`, `force`, `pressure`, `temperature`, `energy`, `power`,
`velocity`, and `acceleration`. Model parameters may also use declaration types such as `material`
and topology types such as `region<solid>` or `region<face>`.

```fea
model bracket(region body : region<solid>, region pad : region<face>) {
    input body_material : material
    input service_load : force = 1 kN
}
```

The region topology is part of the type. A face region cannot be supplied for `body`, even if it
contains one entity.

## Quantities and units

A quantity is a number followed by a unit expression:

```fea
input thickness : length = 8 mm
input duration : time = 12 ms
input load : force = 1.5 kN
input density : mass_density = 2700 kg/m^3
```

Compatible units are normalized for computation while the source unit remains available for
diagnostics and presentation. Unit expressions may use multiplication, division, and integer
powers. A bare number is dimensionless; it does not acquire the unit expected by its destination.

Temperature uses absolute and interval-aware units. The checker rejects arithmetic that confuses
an absolute temperature with a temperature difference.

## Lists and records

Lists are ordered and homogeneous after type resolution. Records have named fields:

```fea
let direction = [0, -1, 0]
let sample = { at = 10 ms, value = 1 kN }
```

Vector quantities can place the unit after the list, as in `[0, -1, 0] kN`. A direction is
dimensionless and should be wrapped in the frame operation expected by the field, for example
`global([0, 0, 1])`.

## References

References name a declaration and, where applicable, an owned member:

```fea
geometry.bracket
geometry.bracket.region.support
material.aluminum
model.bracket.constraint.fixed_support
analysis.static.field.node.displacement
```

The checker resolves references by lexical scope and module imports. An unresolved or ambiguous
reference is an error; there is no string-based lookup at solver time.

## Expressions and operations

Expressions are immutable. They support typed arithmetic, comparisons, lists, records, references,
and calls to cataloged operations. Output operations use the same canonical operation identities
as RunMat where the meaning agrees:

```fea
output displacement = analysis.static.field.node.displacement
output displacement_magnitude = magnitude(output.displacement)
output peak_displacement = max(
    output.displacement_magnitude,
    over = geometry.bracket.region.body,
)
```

An FEA overload defines the accepted field location, quantity, shape, named arguments, and result.
The language server and compiler read the same operation descriptor. The parser does not accept an
arbitrary function name and defer interpretation to a solver.

## Inputs and local values

`input` declares externally bindable data. `let` gives a name to a local expression:

```fea
input service_load : force {
    default = 1 kN
    minimum = 0 N
    maximum = 5 kN
}

let proof_load = 1.5 * input.service_load
```

Inputs are immutable. A default is optional; without one, every consuming study or sweep case must
supply a value. Bounds use the same type and physical dimension as the input.
