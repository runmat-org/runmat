---
description: The design tools the RunMat agent, MCP clients and scripts call over parts, assemblies, drawings, circuits and boards, and the runmat-design command line for checks and exports.
keywords: [agent tools, MCP, design.check, design.inspect, design.export, design.propose_edit, command line, CLI, automation, CI]
---
# Agent tools and the command line

One set of design tools works on every design file kind. The in-app agent calls them, MCP clients see them in the server's tool list, and the `runmat-design` command line runs the checks and exports from scripts. Each tool is a pure function of its input and the files it reads, so the editor, the agent, the server and a CI job get the same answer from the same files.

## Tools

| Tool | What it does |
|---|---|
| `design.check` | Regenerates a file and returns the editor's diagnostics: root causes, the features that failed because of each, source ranges, and the resolved inputs. |
| `design.inspect` | Named faces, regions, published values, inputs, features, mates, the BOM, mass properties, components, nets and rooms, filtered by name or path. `set` regenerates with input values for one run without editing the file: `{"set": {"thickness": "3.5 mm"}}`. |
| `design.render` | A PNG of a view, by named view or camera, in a display mode. |
| `design.drc`, `design.erc` | A board's and a circuit's rule violations: rule, objects, location and source range. |
| `design.simulate` | Runs a circuit's simulation section and returns its measures against their limits. |
| `design.import` | KiCad projects and STEP files in, as RunMat sources staged for review. |
| `design.export` | STEP, STL, 3MF, glTF, DXF and DWG, PDF drawing sheets, KiCad out, a board's fabrication set, and KiCad, SPICE, EDIF and IPC-D-356A netlists. |
| `design.overview` | What a file is, its state, the files linked to it, the decisions its guides still need, and next steps. |
| `design.propose_edit` | A source change or an editor command with its values, staged as a diff with the diagnostics after it. Nothing is written until the change is kept. |

An agent works in three steps: read with `design.inspect` and `design.check`, change with `design.propose_edit`, then confirm with `design.check` on the staged result. A proposed change shows in the editor as a diff with Keep and Undo.

## Inputs for one run

`design.inspect` with `set` answers a what-if without editing the design: the part regenerates with the given input values, and the rest of the file stays as written. An unknown name is an `unknown_input` error that lists the declared inputs. `design.export` takes `set` the same way, so a study's geometry can come from a part at chosen input values [not yet in the current release](pending:design.tools.export-set).

## The command line

```sh
runmat-design check <file>... [--json] [--no-simulate] [--root <dir>]
runmat-design export <file> --format <format> [options]
```

`check` is described in [Checks and diagnostics](/docs/design-language/checks-and-diagnostics#checking-from-the-command-line). `export` writes files to `--out` (the current folder by default) and lists them on stdout:

| Format | Writes |
|---|---|
| `step` | STEP AP242 of a part or an assembly, with PMI |
| `3mf`, `stl`, `gltf` | meshes of a part or an assembly |
| `dxf`, `dwg` | drawing sheets and sheet-metal flat patterns |
| `pdf` | drawing sheets |
| `bom` | a bill of materials |
| `kicad` | a KiCad project from a circuit and its board |
| `fabrication` | a board's fabrication set (see [Fabrication outputs and netlists](/docs/ecad/fabrication)) |

| Option | Effect |
|---|---|
| `--configuration <name>` | Export one configuration. |
| `--all-configurations` | A product family: a part's STEP, or every drawing sheet as DXF, once per configuration. |
| `--sheet <name>` | One drawing sheet. |
| `--outputs <list>` | With `fabrication`: `gerber`, `drill`, `drill-map`, `drawing`, `pick-place`, `bom`, `ipc2581`, `odb`, `ipc356`. |
| `--variant <name>` | With `fabrication`: the assembly files of one variant (repeat for several; `""` is the default variant). |
| `--include-through-hole` | With `fabrication`: through-hole parts in the pick-and-place file. |
| `--root <dir>` | The workspace root. |

The workspace is the file's folder and everything under it, unless `--root` names another.
