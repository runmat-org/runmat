---
description: The RunMat parts library. Find orderable parts with their pins, packages, parameters, datasheets and SPICE models, embed them in a circuit as part declarations, and preview their symbols and footprints.
keywords: [parts library, component library, MPN, part search, footprints, symbols, SPICE models, datasheets, embedded part definitions, library .circuit]
---
# The parts library

A circuit names each component's part by MPN: `component U1 = { part = "LDO-3V3-SOT23" }`. The part comes from the parts library or from a declaration in the circuit itself, so a circuit carries every part it uses and opens the same anywhere.

## Finding parts in Search

Search in the Desktop's left rail finds parts along with files and docs. Type an MPN, a description or a parameter (`buck 36 V`, `10k 0402`), and the results come in groups:

- **Project:** the project's files and the RunMat docs.
- **Library:** parts from KiCad symbol libraries opened beside the open circuit. Choose **Open KiCad library** in the group's header and name a `.kicad_sym` file relative to the circuit. The library is read in place, and its parts are found without any network request.
- **Vendors:** orderable parts from vendors, with each vendor's stock, price breaks and lifecycle. This group appears only after you check **Search part vendors** under the search field.

**Search part vendors** starts off in every session. While it's off, nothing you type in Search is sent to a vendor. The chevron at the right of that row opens a menu of the part vendors your account can use, each with its own checkbox. When the box is checked, Search sends your search only to the vendors checked in that menu, through the RunMat server. Search remembers which vendors you checked; it doesn't remember the box. If you aren't signed in, or the server or your plan doesn't include vendor search, the box is unavailable and its tooltip says why.

Select a part to see its symbol, footprint and 3D body, its parameters, datasheet and offers. **In stock only** in the Vendors header hides parts that no vendor holds in stock.

## Placing a part

Select a part in Search and choose **Place** to put it on the open circuit's shown sheet. Each part found in Search carries its manufacturer, package, pins with pad numbers and electrical types, parameters, datasheet and SPICE model. Placing writes two things into the circuit, as the smallest text change:

- a `part "<mpn>" = { … }` declaration, unless the circuit or the built-in library already has that MPN;
- a `component U<n> = { part = "<mpn>" }` on the selected sheet.

A vendor part with no pinout has no symbol yet, so Place stays off for it. Open a KiCad library that holds its symbol and place it from the Library group.

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
| `package` | the footprint: a generator string for a standard package (`"QFN24_4mm_P0.5_EP2.6"`), or the name of a `footprint` declaration (`package = gps_lcc12`); see [Footprints](/docs/ecad/footprints) |
| `symbol` | optional: the name of a `symbol` declaration the part draws with; without it the symbol is generated from the pins |
| `body` | optional: the part's 3D body, a STEP file or an MCAD `.part`, over its footprint's own |
| `params` | parameters for selection and checks |
| `pins` | each pin's name, pad and electrical type (`passive`, `power_in`, `power_out`, `input`, `output`, `bidirectional`, `no_connect`, …); pins sharing a name become one pin on several pads |
| `nc` | pads the part leaves without a pin on purpose, such as `nc = ["7", "8"]` |
| `spice` | the SPICE model: `model` (a subcircuit or model name), `library`, and `nodes` in model-pin order |

A declared part replaces a built-in part of the same MPN. A part with a `spice` model simulates with no `model` statement in the simulation section; a section's own `model` wins, so one section can run a switching model and another an averaged one of the same part.

## Footprints

A part's `package` string names a standard package, and RunMat generates its land pattern from the IPC-7351B equations, so a standard part needs no footprint file: `"R0402"`, `"SOIC8"`, `"QFN24_4mm_P0.5_EP2.6"`, `"SOT23-6"`, `"PinHeader1x04"`. [Footprints](/docs/ecad/footprints) lists every family with its string and pad numbering.

A package no family covers, such as a module or an odd connector, gets a `footprint` declaration in a library file under `electrical/library/footprints/`, and the part names it: `package = gps_lcc12`.

## Libraries

A `.circuit` file with no `circuit` block is a library of `footprint`, `symbol` and `part` declarations. A circuit uses one with `use "../library/parts.circuit"`, and its parts are then placed by MPN like built-in ones. Before declaring a new part, check whether a library the project uses already has it. [Footprints](/docs/ecad/footprints#libraries) covers how names resolve and the checks a library passes.

## Part data from vendors

Vendor data comes through the RunMat server, which holds any vendor keys and caches results. An organization can connect its own vendor accounts to search live stock and pricing, under each vendor's terms; those vendors appear in Search for the organization's members.
