---
description: Bring existing electronics designs into RunMat. KiCad projects, symbol and footprint libraries with their SPICE fields, SPICE model libraries, and the formats still to come.
keywords: [KiCad import, import schematic, import PCB, symbol library, footprint library, SPICE models, migrate PCB design]
---
# Import

RunMat reads existing electronics designs into `.circuit` and `.circuit.layout` sources, staged for review before anything is written.

| Source | Becomes |
|---|---|
| A KiCad project | a `.circuit` with its sheets, components, nets and classes, and a `.circuit.layout` with its stackup, footprints, copper and rules |
| A KiCad symbol library (`.kicad_sym`) | catalog parts, whose packages are the `.kicad_mod` footprints their Footprint property names, read in place from `<Lib>.pretty/` beside the library |
| KiCad simulation fields on a symbol | the part's SPICE model: a subcircuit or a device model card, with its nodes in model-pin order |
| An XML schematic (`.sch`) and board (`.brd`) in the common XML format, version 6 and later | a `.circuit` with a component and a `part` declaration for each placed part, nets joined by name across sheets, net classes and variants; a `.circuit.layout` whose outline, footprints, tracks, vias, zones and limits come from the board; and a `.kicad_mod` footprint per package in `<name>-footprints/` |
| SPICE model libraries (`.lib`, `.cir`, `.sp`) | models a part or a simulation section names; libraries in other common dialects read in, their parasitics becoming their own elements |

Import runs from the agent (`design.import`) and the Desktop. A footprint file named by a part's `package` is listed as a dependency of the circuit, and a missing file is a diagnostic.

## Other formats

Each import ends with a report of what carried over, what was approximated and what was left out, and the staged change lists it as notes. Schematics and boards from the other common tools, in their binary, text and JSON formats, read in the same way [not yet in the current release](pending:design.ecad.import.other-tools). A KiCad round trip that checks hierarchy, symbols, footprints, net classes and rules on a corpus of real projects gates every release [not yet in the current release](pending:design.ecad.import.kicad-roundtrip).
