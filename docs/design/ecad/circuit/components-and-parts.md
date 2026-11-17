---
description: Reference for components in a RunMat circuit. Placing library parts by MPN, values with tolerances, values computed in .m functions, component classes and properties, power for the thermal map, automatic designators and renames, and parts declared in the file.
keywords: [schematic components, library parts, MPN, component value, tolerance, designators, annotate, renamed_from, part declaration, component properties]
---
# Components and library parts

`component <designator> = { part = "<mpn>" }` places a part. The part comes from the parts library or from a `part "<mpn>" = { … }` declaration in the file; see [The parts library](/docs/ecad/parts-library).

| Field | Meaning |
|---|---|
| `part` | the library part, by MPN |
| `value` | the component's value with its tolerance: `value = 10 kohm ± 1 %` |
| `classes` | component classes, for board rules and rooms |
| `properties` | typed properties |
| `variants` | how the component is fitted in each variant |
| `power` | the dissipation the board's thermal map uses, in W |
| `renamed_from` | the old designator after a rename, so the board keeps the placement |

A designator ending in `?` (`R?`) is numbered automatically.

## Values

A value carries its unit and, optionally, its tolerance: `value = 10 kohm ± 1 %`, `value = 100 nF ± 10 %`. A value can also come from a `.m` function in the circuit's folder, written as a call with named arguments and the unit the result is read in:

```circuit fragment
sheet main {
    component R3 = { part = "R0402-1K", value = ledResistor(vs = 3.3 V, vf = 2 V, i = 2 mA) as ohm }
}
```

The circuit keeps the call and recomputes the value when an argument changes. See [MATLAB functions in design values](/docs/design-language/matlab-functions#circuit-values).

## Designators and renames

Annotate numbers every designator written with `?`. Renaming a component writes `renamed_from = <old>`, so the board keeps its footprint's placement and routing under the new designator.

## Change part, footprint and pins

Right-click a component to change what it is. Each command writes the component's statement in the `.circuit`; Update board takes the change to the board.

- **Change part** gives the selected components another part by MPN. Pins map by name, so the new part must have every pin the schematic connects; a part that lacks one is refused, naming the pin.
- **Footprint** chooses another package. A component named by its symbol (`entity`) takes the package as written. A component placed as an orderable part takes the library part with the same symbol, the same value (resistance, capacitance or inductance) and that package: `R0402-10K` with footprint `R0603` becomes `R0603-10K`. When the library has no such part, the message lists the packages the value comes in.
- **Swap pins/gates** trades the connections of two pins of one gate (`1, 2`), or of two gates (`B, C` on a quad op-amp). The library decides what may swap: pins or gates with the same non-zero swap group, such as a resistor's two terminals or the four amplifiers of a quad op-amp. The schematic's connections and the board's pad nets change together, in every channel of a repeated sheet.

## Parts in the file

`part "<mpn>" = { … }` at the circuit's top level declares a part the file uses: its pins and pads, package, parameters, datasheet and SPICE model. A declared part replaces a built-in part of the same MPN. See [The parts library](/docs/ecad/parts-library#part-declarations).
