---
description: Values in RunMat design files are quantities with units, numbers, strings, lists, records, references and expressions. Learn inputs, let values, tolerances and the units each file kind reads.
keywords: [design values, units, quantities, inputs, parameters, let, tolerance, expressions, mm, deg, kohm, design parameters]
---
# Values, units and expressions

A design value is a number with its unit, a reference to another value, or an expression of them. The compiler checks the dimension of every value against where it is used: a sketch coordinate takes a length, a revolve angle takes an angle, a resistor value takes a resistance. Values are stored in SI units, so `80 mm`, `8 cm` and `0.08 m` are the same length.

## Value forms

| Form | Example | Meaning |
|---|---|---|
| Quantity | `80 mm`, `30 deg`, `10 kohm` | A number and its unit. A space separates them. |
| Number | `2.5`, `4` | A dimensionless value, such as a count or a ratio. |
| String | `"PL-100"`, `"R0402-10K"` | Text: part numbers, file paths, names that are not plain words. |
| Word | `cut`, `third_angle`, `power` | A choice from a fixed set, such as `mode = cut`. |
| List | `[0 mm, 40 mm]`, `[feature.round, feature.slot]` | An ordered list; points and vectors are lists. |
| Record | `{ body = feature.body.body, part_number = "PL-100" }` | Named fields. |
| Reference | `input.width`, `let.half`, `feature.body.face.end`, `net.GND` | Another item's value or entity, by kind and name. |
| Expression | `input.width / 2 + 5 mm`, `-input.offset` | Arithmetic on values. |
| Tolerance | `10 mm ± 0.1 mm`, `10 mm ± [0.02 mm, 0 mm]`, `10 kohm ± 1 %` | A nominal value with symmetric or `[plus, minus]` bounds. |
| Function call | `camBaseRadius(lift = input.lift, rise = 120 deg)` | A value computed by a `.m` function. See [MATLAB functions in design values](/docs/design-language/matlab-functions). |

## Inputs

An input is a parameter of the design. Configurations, runs and sweeps set inputs without editing the rest of the file.

```part
runmat 0.7
std 1

part spacer {
    input outer : length = 12 mm
    input bore : length = { default = 6.4 mm, minimum = 3 mm, maximum = input.outer }
    input height : length = 8 mm

    sketch ring on origin.xy {
        rim = circle(center = [0 mm, 0 mm], radius = input.outer / 2)
        hole = circle(center = [0 mm, 0 mm], radius = input.bore / 2)
    }
    feature body = extrude(profile = sketch.ring, distance = input.height)

    output spacer = { body = feature.body.body, part_number = "SP-12" }
}
```

- `input <name> : <kind> = <default>` declares an input. The kind is required; the default is optional.
- The record form `{ default = …, minimum = …, maximum = … }` adds bounds. A value outside them is an error at the input. A bound may itself be an input, as `maximum = input.outer` is above.
- A reference to an input is `input.<name>`.

## Let values

`let <name> = <value>` names a value inside one file. Other statements read it as `let.<name>`, or by its bare name inside an expression. Lets are evaluated in dependency order, so a let may use another let written below it. A cycle (`let a = let.b`, `let b = let.a`) is an error.

```part
runmat 0.7
std 1

part tab {
    input width : length = 40 mm
    let half = input.width / 2
    let height : length = let.half + 5 mm

    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])
        l2 = line(start = [input.width, 0 mm], end = [input.width, let.height])
        l3 = line(start = [input.width, let.height], end = [0 mm, let.height])
        l4 = line(start = [0 mm, let.height], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s, distance = 3 mm)

    output tab = { body = feature.body.body }
}
```

A `let` may state its kind (`let height : length = …`); the checker then compares the value's dimension with it.

## Expressions

| Operator | Meaning | Dimension rule |
|---|---|---|
| `a + b`, `a - b` | sum, difference | `a` and `b` have the same dimension |
| `a * b` | product | dimensions multiply: `length * length` is an area |
| `a / b` | quotient | dimensions divide: `length / length` is dimensionless |
| `-a` | negation | unchanged |
| `( … )` | grouping | unchanged |

Multiplication and division bind tighter than addition and subtraction, and `±` binds loosest: `20 mm + 2 * 5 mm ± 0.1 mm` is `30 mm` with a tolerance of `0.1 mm`.

## Tolerances

A toleranced value carries its nominal and its bounds. Geometry and solvers use the nominal; checks read the bounds. In a `.part`, the `size` and `location` statements attach toleranced values to faces as PMI, which drawings and STEP exports carry (see [Outputs, regions and published values](/docs/mcad/part/regions-and-publish#pmi)). In a `.circuit`, a component's toleranced value sets the spread a tolerance study samples (see [Simulation sections and measures](/docs/ecad/circuit/simulation)).

| Written | Nominal | Bounds |
|---|---|---|
| `25 mm ± 0.1 mm` | 25 mm | 24.9 mm to 25.1 mm |
| `10 mm ± [0.02 mm, 0 mm]` | 10 mm | 10 mm to 10.02 mm |
| `10 kohm ± 1 %` | 10 kΩ | 9.9 kΩ to 10.1 kΩ |

## Units

### Mechanical files

`.part`, `.assembly` and `.drawing` values take these kinds and units:

| Kind (as written after `:`) | Units |
|---|---|
| `length`, `distance` | `m`, `cm`, `mm`, `um` (also `µm`), `in` |
| `area` | `m^2`, `mm^2` |
| `angle` | `deg`, `rad` |
| `mass` | `kg`, `gram` |
| `real`, `dimensionless`, `count`, `integer` | a bare number |

A bare number where a length is expected is an error: write `5 mm`, not `5`. Board coordinates in a `.circuit.layout` are the exception: a bare number there is millimetres.

### Electrical files

`.circuit` and `.circuit.layout` values take SI units with the SI prefixes `T`, `G`, `M`, `k`, `m`, `u` (also `µ`), `n`, `p` and `f`:

| Quantity | Unit | Examples |
|---|---|---|
| Resistance | `ohm` (also `Ohm`, `Ω`) | `10 kohm`, `4.7 Mohm`, `0.5 mohm` |
| Capacitance | `F` | `100 nF`, `10 uF` |
| Inductance | `H` | `2.2 uH` |
| Voltage | `V` | `3.3 V`, `22.2 V` |
| Current | `A` | `20 A`, `1 mA` |
| Frequency | `Hz` | `48 kHz` |
| Time | `s` | `5 ns`, `14 us` |
| Power, energy, charge | `W`, `J`, `C` | `62.5 mW` |
| Conductance | `S` | `1 mS` |
| Length | `m` | `0.2 mm`; board lengths also take `mil` |
| Temperature | `K`, `degC` | `25 degC` |
| Tolerance | `%` | `± 1 %` |

The circuit printer writes quantities in SI units (`10000 ohm`), so a file reads back bit for bit; any of the forms above reads in.

## Strings and words

A plain word (`power`, `F.Cu`, `GND`) needs no quotes. A name that is not a plain word is quoted: a net called `+3V3` is written `"net.+3V3"` in a reference, and a pin `1` of `C1` is `"C1.1"`. File paths and part numbers are always strings.
