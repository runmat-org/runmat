---
description: The RunMat parts library. Find orderable parts with their pins, packages, parameters, datasheets and SPICE models, embed them in a circuit as part declarations, and preview their symbols and footprints.
keywords: [parts library, component library, MPN, part search, footprints, symbols, SPICE models, datasheets, embedded part definitions]
---
# The parts library

A circuit names each component's part by MPN: `component U1 = { part = "LDO-3V3-SOT23" }`. The part comes from the parts library or from a declaration in the circuit itself, so a circuit carries every part it uses and opens the same anywhere.

## Placing a part

Searching the library returns orderable parts with their manufacturer, package, pins with pad numbers and electrical types, parameters, datasheet and SPICE model. Placing one writes two things into the circuit, as the smallest text change:

- a `part "<mpn>" = { … }` declaration, unless the circuit or the built-in library already has that MPN;
- a `component U<n> = { part = "<mpn>" }` on the selected sheet.

The library's preview shows the part's symbol as the schematic places it, its footprint (pads with pin names, courtyard, silkscreen and fabrication layers) and its body.

## Part declarations

```circuit fragment
part "BUCK-60V-3A" = {
    manufacturer = "Sample Catalog",
    description = "60 V 3 A synchronous buck",
    category = regulator,
    package = "SOT23-6",
    datasheet = "https://example.com/buck.pdf",
    params = { vin_max = "60 V" },
    pins = [
        { name = BST, pad = "1", type = passive },
        { name = GND, pad = "2", type = power_in }
    ],
    spice = { model = BUCK60, library = "models/buck.lib", nodes = [VIN, SW, FB, GND] }
}
```

| Field | Meaning |
|---|---|
| `manufacturer`, `description`, `category`, `datasheet` | what the part is |
| `package` | a footprint generator string (`"SOT23-6"`), or a `.kicad_mod` file read in place |
| `params` | parameters for selection and checks |
| `pins` | each pin's name, pad and electrical type (`passive`, `power_in`, `power_out`, `input`, `output`, `bidirectional`, `no_connect`, …); pins sharing a name become one pin on several pads |
| `spice` | the SPICE model: `model` (a subcircuit or model name), `library`, and `nodes` in model-pin order |

A declared part replaces a built-in part of the same MPN. A part with a `spice` model simulates with no `model` statement in the simulation section; a section's own `model` wins, so one section can run a switching model and another an averaged one of the same part.

## Part data from suppliers

A workspace can connect its own supplier accounts to search live stock and pricing, under each supplier's terms. Library packages that a team shares between projects (`use library`) [not yet in the current release](pending:design.ecad.use-library).
