---
description: The .fea type system and units table, with every accepted unit spelling, unit expressions, temperatures, vectors, lists, records, references, operators, inputs and let bindings.
keywords: [.fea types, units table, unit expressions, physical quantities, temperature units, delta_degC, vectors, references, expressions]
doc_type: language-reference
---
# Types, units and expressions

The checker gives every input, reference, literal and operation a type before a study is planned. Physical values carry a dimension, so a length cannot stand in for a force because both are numbers, and a pressure written in `MPa` and one written in `Pa` are the same quantity.

## Types

Inputs, model parameters and `let` bindings are declared with a type: `input thickness : length = 8 mm`.

| Group | Types |
| --- | --- |
| Plain values | `boolean` (also `bool`), `integer`, `real`, `scalar`, `dimensionless`, `complex`, `string` |
| Geometry and kinematics | `length`, `displacement`, `area`, `volume`, `angle`, `rotation`, `velocity`, `acceleration`, `angular_velocity`, `angular_acceleration` |
| Mass and time | `mass`, `mass_density`, `density`, `time`, `frequency`, `mass_flow_rate`, `volume_flow_rate` |
| Mechanics | `force`, `moment`, `pressure`, `stress`, `body_force_density`, `energy`, `power` |
| Heat | `temperature`, `temperature_difference`, `heat_flux`, `heat_generation_density`, `heat_transfer_coefficient`, `thermal_conductivity` |
| Electromagnetics | `electric_potential`, `electric_current`, `electric_current_density`, `electric_charge`, `electric_charge_density`, `magnetic_flux_density`, `magnetic_vector_potential` |
| Declarations | `geometry`, `mesh`, `material`, `model`, `study`, `execution` |
| Regions | `region<solid>`, `region<face>`, `region<edge>`, `region<vertex>`, `region<cell>`, `region<anything>` |

The region's topology is part of its type: a face region cannot be passed where `region<solid>` is expected, even when it holds one entity.

```fea fragment
model bracket(region body : region<solid>, region pad : region<face>) {
    input body_material : material
    input service_load : force = 1 kN
}
```

## Units

A quantity is a number followed by a unit: `8 mm`, `1.5 kN`, `2700 kg/m^3`. These unit symbols are accepted, and combine into unit expressions with `*`, `/` and integer powers `^`:

| Dimension | Units |
| --- | --- |
| Length | `m`, `mm`, `um`, `nm`, `cm`, `km`, `in` (also `inch`), `ft` |
| Time | `s`, `ms`, `us`, `ns` |
| Frequency | `Hz`, `kHz`, `MHz`, `GHz` |
| Mass | `kg`, `g` (also `gram`) |
| Force | `N`, `kN` |
| Pressure and stress | `Pa`, `kPa`, `MPa`, `GPa` |
| Energy | `J` |
| Power | `W`, `mW`, `kW`, `MW` |
| Temperature | `K`, `degC` (absolute), `delta_degC` (difference) |
| Current | `A`, `mA`, `kA` |
| Charge | `C` |
| Potential | `V`, `mV`, `kV`, `MV` |
| Magnetic flux density | `T`, `mT` |
| Angle | `rad`, `deg` |
| Memory (execution resources) | `B`, `KiB`, `MiB`, `GiB` |

Derived units are written as expressions:

| Quantity | Written as |
| --- | --- |
| Density | `kg/m^3` |
| Thermal conductivity | `W/m/K` |
| Specific heat | `J/kg/K` |
| Heat transfer coefficient | `W/m^2/K` |
| Heat flux | `W/m^2` |
| Dynamic viscosity | `Pa*s` |
| Electrical conductivity | `A/V/m` (siemens per meter) |
| Current density | `A/m^2` |
| Electric field | `V/m`, `kV/mm` |
| Thermal expansion coefficient | `K^-1` |
| Specific acoustic impedance | `Pa*s/m` |
| Permeability (porous) | `m^2` |

A bare number is dimensionless; it never takes on the unit its destination expects, so `youngs_modulus = 200e9` is an error and `poisson_ratio = 0.3` is correct. Compatible units are converted for computation, and the authored unit is kept for diagnostics and display.

### Temperatures

`K` and `degC` are absolute temperatures, and `delta_degC` is a temperature difference. `20 degC` is 293.15 K, while `20 delta_degC` is a 20 K rise. The checker rejects arithmetic that mixes them up, such as adding two absolute temperatures, and a `temperature_difference` input does not accept `20 degC`.

## Lists, vectors and records

Lists are ordered and, after type resolution, homogeneous. Records have named fields:

```fea fragment
let direction = [0, -1, 0]
let sample = { at = 10 ms, value = 1 kN }
```

A vector quantity is written either with a unit on each component, `[0 mm, 0 mm, 5 mm]`, or with one unit after the list, `[0, 0, -100000] Pa`. Both forms are accepted by loads and geometry operations; in orientation and frame members (`origin = [0, 0, 0] m`) the trailing-unit form is [not yet in the current release](pending:language.vector.trailing-unit). A direction is dimensionless and is wrapped in the frame it is expressed in, such as `global([0, 0, 1])`.

## References

A reference names a declaration and, where it applies, an owned member. Every segment is checked:

```fea fragment
geometry.bracket
geometry.bracket.region.support
material.aluminum
model.bracket.constraint.fixed_support
model.response_spectrum.design
load_case.service
analysis.static.state.basis
analysis.static.field.node.displacement
output.peak_stress
input.service_load
```

References resolve through lexical scope and the module's imports. An unresolved or ambiguous reference is a check error; nothing is looked up by string at solve time.

## Expressions and operators

Expressions are immutable. They combine literals, references, lists, records and calls to cataloged operations with these operators:

| Operator | Meaning |
| --- | --- |
| `+`, `-`, `*`, `/`, `^` | Arithmetic, with units: `1.5 * input.service_load` is a force, `input.force / input.area` a pressure. |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | Comparisons of quantities of the same dimension. |
| `a..b` | An integer range, such as `workers = 4..32`. |

Operations are called by name with positional and named arguments, such as `max(magnitude(output.displacement), over = geometry.bracket.region.body)`. Each operation's overloads fix the accepted field location, quantity, shape, named arguments and result type; an unknown operation or an argument of the wrong kind is a check error. The catalog of output operations is in [operations and output queries](./reference/operations-and-output-queries.md).

## Inputs and `let`

`input` declares a value that a study, a sweep case or the command line can set; `let` names a local expression:

```fea fragment
input service_load : force {
    default = 1 kN
    minimum = 0 N
    maximum = 5 kN
}

let proof_load = 1.5 * input.service_load
```

Inputs are immutable. A default is optional; without one, every study binding or sweep case must supply the value. Bounds have the input's type and dimension, and a value outside them is an error. Any name can be declared, including names that are also keywords, such as `input load : force` [not yet in the current release](pending:language.keyword-names). See [inputs, sweeps and cases](./inputs-sweeps-and-cases.md).
