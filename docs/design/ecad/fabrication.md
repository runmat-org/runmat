---
description: Fabrication outputs and netlists from a RunMat board. Presets, the layer table, settings saved with the board, Gerber X2 with a job file, Excellon drill files and drill map, fabrication and assembly drawings, BOM and pick-and-place per variant, IPC-2581, ODB++, the pre-flight checks, and KiCad, SPICE, EDIF and IPC-D-356A netlists.
keywords: [Gerber, Gerber X2, Excellon, drill files, pick and place, BOM, assembly drawing, IPC-2581, ODB++, IPC-D-356, netlist, fabrication outputs, PCB manufacturing files, fabrication presets]
---
# Fabrication outputs and netlists

A board's fabrication set is written in one step from its `.circuit.layout`: copper, mask, legend and drill data, the fabrication and assembly drawings, and the assembly files of each variant. Every output comes from the same geometry, so the Gerber and ODB++ copies of a board agree. The Desktop's Fabrication outputs panel, `design.export` and `runmat-design export --format fabrication` read the same settings from the board and make the same files. [Make fabrication outputs](/docs/ecad/guides/make-fabrication-outputs) walks through the panel.

## Outputs

| Output | Word | Writes |
|---|---|---|
| Gerber | `gerber` | Gerber X2, one file per layer, and the `.gbrjob` job file with the stackup |
| Drill | `drill` | Excellon drill files, plated per drill span and non-plated |
| Drill map | `drill_map` | the drill symbols and table, as Gerber and PDF |
| Fabrication drawing | `drawing` | drill table, board size, the layer and stackup table and notes, as Gerber and PDF |
| Test netlist | `ipc356` | IPC-D-356A, for bare-board electrical test |
| Pick and place | `pick_place` | the centroid file of each variant |
| BOM | `bom` | the bill of materials of each variant |
| Assembly drawing | `assembly_drawing` | one sheet per side with component outlines and designators, the bottom seen from below, as Gerber and PDF |
| IPC-2581 | `ipc2581` | IPC-2581C |
| ODB++ | `odb` | ODB++, as a job tree or a `.tgz` |

The assembly drawings mark pin 1 and the polarity of polarized parts [not yet in the current release](pending:design.ecad.fab.assembly-drawing).

## Presets

| Preset | Outputs |
|---|---|
| `fab_house` (the default) | `gerber`, `drill`, `drill_map`, `drawing`, `ipc356` |
| `assembly` | `pick_place`, `bom`, `assembly_drawing` |
| `full` | every output |
| `custom` | the outputs listed |

## The layer table

The layer table lists the board top to bottom: paste, legend, mask, the stackup's copper layers with their weight and the dielectrics between them, the bottom's layers, the courtyards and the outline. Each row gives its material, its thickness, the Gerber file it makes and whether it is on by default; the courtyards are off. A request names the layers to write, and the job file lists exactly those.

`design.export` with `previewLayer` draws one layer as its Gerber makes it (flashes, strokes and regions, the mask as its openings) and writes nothing; the panel shows it when a layer row is selected.

## Settings saved with the board

The board keeps the settings of its last fabrication export in a `fabrication` statement, written only for the fields that differ from the defaults:

```circuit.layout fragment
fabrication = { preset = custom, outputs = [gerber, drill, drill_map, drawing, ipc356, pick_place, bom], layers = ["F.Cu", "B.Cu", "F.Mask", "B.Mask", "F.Silkscreen", "Edge.Cuts"], variants = ["", lite], folder = "fab/sensor-B", zip = true, gerber = { units = inch, decimals = 5 }, pick_place = { sides = separate, units = mm }, bom = { columns = [designators, quantity, value, mpn], group_by_value = false }, revision = "B" }
```

| Field | Sets |
|---|---|
| `preset`, `outputs` | the outputs to make |
| `layers` | the Gerber layers to write, by layer name; every layer but the courtyards when left out |
| `variants` | the variants that get assembly files (`""` is the default variant); every variant when left out |
| `folder` | the output folder, relative to the board's folder; `fab/<board>`, or `fab/<board>-<revision>` with a revision, by default |
| `zip` | pack the set into one `.zip` beside the files (on by default) |
| `through_hole` | list through-hole parts in the pick-and-place file |
| `gerber = { units, decimals }` | `mm` with 5 or 6 decimals (4.5 or 4.6 coordinates), or `inch` with 5 or 6 (2.5 or 2.6), for the layers, the drill map and the drawings |
| `pick_place = { sides, units }` | `both` sides in one file, `separate` files, or `top` or `bottom` only; `mm` or `inch` |
| `bom = { columns, group_by_value }` | the columns (`designators`, `quantity`, `value`, `package`, `mpn`, `manufacturer`), and one line per value or, with `group_by_value = false`, one line per designator |
| `revision` | the board revision, written into the Gerber and job files' project attribute and the fabrication drawing |

## Pre-flight

Before it writes, an export can run the pre-flight: the counts of DRC findings, ERC findings, unrouted connections and schematic changes the board hasn't taken, from the same checks as `runmat-design check` without the simulations. The Desktop runs it on Generate and asks before exporting a board with issues.

## The command line

```sh
runmat-design export board/sensor.circuit.layout --format fabrication
runmat-design export board/sensor.circuit.layout --format fabrication --preset assembly --variant "" --variant lite
runmat-design export board/sensor.circuit.layout --format fabrication --outputs gerber,drill --gerber-units inch --no-zip --out fab
```

The command starts from the board's saved settings, and each flag overrides one. It writes to the settings' folder unless `--out` names another.

| Option | Effect |
|---|---|
| `--preset <name>` | `fab_house`, `assembly`, `full` or `custom` |
| `--outputs <list>` | the outputs, by word (`drill-map` and `drill_map` are the same) |
| `--layers <list>` | the Gerber layers, by name |
| `--variant <name>` | the assembly files of a variant; repeat for several; `""` is the default variant |
| `--include-through-hole` | through-hole parts in the pick-and-place file |
| `--zip`, `--no-zip` | pack the set or not |
| `--gerber-units mm\|inch`, `--gerber-decimals 5\|6` | the Gerber coordinate format |
| `--pick-place-sides both\|separate\|top\|bottom`, `--pick-place-units mm\|inch` | the pick-and-place files |
| `--bom-columns <list>`, `--bom-ungrouped` | the BOM's columns, and one line per designator |
| `--revision <text>` | the board revision |
| `--remember` | save these settings into the board |

Values print with engineering prefixes (`12 pF`, `10 kΩ`) in BOMs and netlists.

## Netlists

`design.export` writes a circuit's netlist as `kicad_netlist`, `spice`, `edif` (EDIF 2.0.0) or `ipc356`, for other tools and test fixtures.
