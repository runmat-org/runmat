---
description: The RunMat parts library. Find orderable parts with their pins, packages, parameters, datasheets and SPICE models, embed them in a circuit as part declarations, and preview their symbols and footprints.
keywords: [parts library, component library, MPN, part search, footprints, symbols, SPICE models, datasheets, embedded part definitions]
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
| `package` | a footprint generator string (`"QFN32_5mm_P0.5_EP3.3"`; see [Footprints from generator strings](#footprints-from-generator-strings)), or a `.kicad_mod` file read in place for interop |
| `params` | parameters for selection and checks |
| `pins` | each pin's name, pad and electrical type (`passive`, `power_in`, `power_out`, `input`, `output`, `bidirectional`, `no_connect`, …); pins sharing a name become one pin on several pads |
| `spice` | the SPICE model: `model` (a subcircuit or model name), `library`, and `nodes` in model-pin order |

A declared part replaces a built-in part of the same MPN. A part with a `spice` model simulates with no `model` statement in the simulation section; a section's own `model` wins, so one section can run a switching model and another an averaged one of the same part.

## Footprints from generator strings

A part's `package` string names a standard package. RunMat generates its land pattern from the IPC-7351B equations, with pads, courtyard, silkscreen with a pin-1 mark, fab outline and a 3D body, so a standard part needs no footprint file. Letters are not case-sensitive. Dimensions are in mm, and the `mm` after a body size is optional.

| Package family | String | Example |
|---|---|---|
| Chip resistor, capacitor, inductor, diode, fuse, LED (imperial size code) | `R`, `C`, `L`, `D`, `F` or `LED`, then the size | `R0402`, `C0603`, `LED0805` |
| SOIC (3.9 mm body; add `W` for the 7.5 mm wide body) | `SOIC<pins>` / `SOIC<pins>W` | `SOIC8`, `SOIC16W` |
| TSSOP | `TSSOP<pins>` | `TSSOP20` |
| QFP (square body) | `QFP<pins>_<body>mm_P<pitch>` | `QFP48_7mm_P0.5` |
| QFN (square body, optional exposed pad) | `QFN<pins>_<body>mm_P<pitch>[_EP<pad>]` | `QFN32_5mm_P0.5_EP3.3`, `QFN16_3mm_P0.5` |
| BGA (full array) | `BGA<balls>_<rows>x<cols>_P<pitch>[_BALL<diameter>]` | `BGA256_16x16_P1.0_BALL0.5` |
| SOT-23 (3, 5 or 6 pins) | `SOT23-<pins>` | `SOT23-3`, `SOT23-6` |
| SOT-223 | `SOT223` | `SOT223` |
| Pin header | `PinHeader<cols>x<rows>[_P<pitch>]` (pitch defaults to 2.54) | `PinHeader1x04`, `PinHeader2x05_P2.54` |

Add `@A` for the most-material density (hand soldering, rework) or `@C` for the least-material density (dense boards). The default is nominal: `C0402@A`, `QFN32_5mm_P0.5_EP3.3@C`.

Pads number in the package's standard order: pin 1 at the top left, counter-clockwise for QFN and QFP, rows A, B, C… and columns 1, 2, 3… for BGA. The exposed pad is the last pad, so a QFN32 with an exposed pad has pads 1–33. A part maps its pins to these pad numbers.

Use a generator string whenever a family above fits; most parts need nothing else. A package no string covers (a module, an odd connector) needs a custom footprint: today that is a `.kicad_mod` file read in place, and native footprint declarations in a library `.circuit` replace it.

## Part data from vendors

Vendor data comes through the RunMat server, which holds any vendor keys and caches results. An organization can connect its own vendor accounts to search live stock and pricing, under each vendor's terms; those vendors appear in Search for the organization's members. Library packages that a team shares between projects (`use library`) [not yet in the current release](pending:design.ecad.use-library).
