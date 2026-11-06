---
description: Fabrication outputs and netlists from a RunMat board. Gerber X2 with a job file, Excellon drill files and drill map, the fabrication drawing, BOM and pick-and-place per variant, IPC-2581, ODB++, and KiCad, SPICE, EDIF and IPC-D-356A netlists.
keywords: [Gerber, Gerber X2, Excellon, drill files, pick and place, BOM, IPC-2581, ODB++, IPC-D-356, netlist, fabrication outputs, PCB manufacturing files]
---
# Fabrication outputs and netlists

A board's fabrication set is written in one step from its `.circuit.layout`: copper, mask, silkscreen and drill data, a fabrication drawing, and the assembly files of each variant. Every output comes from the same geometry, so the Gerber and ODB++ copies of a board agree.

```sh
runmat-design export board/sensor.circuit.layout --format fabrication --out fab
runmat-design export board/sensor.circuit.layout --format fabrication --outputs gerber,drill --out fab
runmat-design export board/sensor.circuit.layout --format fabrication --variant "" --variant lite --out fab
```

| Folder | Holds |
|---|---|
| `gerber/` | Gerber X2, one file per layer, and the `.gbrjob` job file with the stackup |
| `drill/` | Excellon drill files per span and for non-plated holes, and the drill map |
| `drawing/` | the fabrication drawing: drill symbols and table, board size, the layer and stackup table and notes, also as PDF |
| `assembly/` | the BOM and the pick-and-place file of each variant |
| `ipc2581/` | IPC-2581C |
| `odb/` | ODB++, as a job tree or a `.tgz` |
| `netlist/` | IPC-D-356A, for bare-board electrical test |

| Option | Effect |
|---|---|
| `--outputs <list>` | Write only these: `gerber`, `drill`, `drill-map`, `drawing`, `pick-place`, `bom`, `ipc2581`, `odb`, `ipc356`. |
| `--variant <name>` | The assembly files of these variants; `""` is the default variant. |
| `--include-through-hole` | List through-hole parts in the pick-and-place file. |

Values print with engineering prefixes (`12 pF`, `10 kΩ`) in BOMs and netlists. The Desktop's Fabrication outputs command writes the same set into a folder, with one switch per output.

## Netlists

`design.export` writes a circuit's netlist as `kicad_netlist`, `spice`, `edif` (EDIF 2.0.0) or `ipc356` for other tools and test fixtures.

## Assembly drawings

An assembly drawing per side, with component outlines, designators and polarity marks, joins the set [not yet in the current release](pending:design.ecad.fab.assembly-drawing).
