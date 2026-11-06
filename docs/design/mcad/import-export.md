---
description: Exchange RunMat designs with other tools. STEP AP242 with PMI and part numbers in and out, DXF and DWG drawings and flat patterns, PDF sheets, 3MF, STL and glTF meshes, and BOMs.
keywords: [STEP, AP242, PMI, import STEP, export STEP, DXF, DWG, PDF drawing, 3MF, STL, glTF, BOM export, CAD interoperability]
---
# Import and export

RunMat reads STEP into a part and writes every common mechanical exchange format. Exports come from the editor's Export command, from the agent (`design.export`) and from the command line (`runmat-design export`), all through the same writers.

## Import

| Format | Becomes |
|---|---|
| STEP | an `import` feature in a part: `feature housing = import(source = "supplier/housing.step", digest = …)`. The body keeps the file's face names, read as `feature.housing.face("Top")`, and the digest pins the file's content. AP242 semantic PMI comes in as the part's PMI. |
| KiCad projects | circuits and boards; see [Import](/docs/ecad/import) |

Direct edits (`direct.move_faces`, `direct.replace_faces`, `direct.delete_faces`) change an imported body as features in the history, so the supplier's file stays untouched; see [Features](/docs/mcad/part/features#direct-edits).

## Export

| Format | From | Carries |
|---|---|---|
| `step` | a part or an assembly | AP242 geometry, the assembly structure, part numbers (a configured part travels under its own part number), region and output labels as face and product names, and semantic PMI |
| `dxf`, `dwg` | a drawing, or a sheet-metal part's flat pattern | sheets with views, annotations and tables |
| `pdf` | a drawing | sheets |
| `3mf` | a part or an assembly | meshes, for printing |
| `stl` | a part or an assembly | a mesh |
| `gltf` | a part or an assembly | meshes, for viewers and the web |
| `bom` | an assembly | the bill of materials |

```sh
runmat-design export bracket.part --format step --out exports
runmat-design export bracket.part --format step --all-configurations --out exports
runmat-design export bracket.drawing --format pdf
runmat-design export enclosure.part --format dxf --out flat
```

Region labels become STEP face names, which is how a study selects them: a region `label = "Mounting bores"` in the part is `select name "Mounting bores"` in the `.fea` file.

## Studies of a design

A study reads a part or an assembly's exported STEP today. A `.fea` geometry that names the design directly, `geometry arm { part = arm, inputs = { length = 92 mm } }`, regenerates it at the given inputs with no export step [not yet in the current release](pending:design.fea.part-geometry).
