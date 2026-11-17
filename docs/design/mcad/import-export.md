---
description: Exchange RunMat designs with other tools. STEP AP242 with PMI and part numbers in and out, DXF and DWG drawings and flat patterns, PDF sheets, 3MF, STL and glTF meshes, and BOMs.
keywords: [STEP, AP242, PMI, import STEP, export STEP, DXF, DWG, PDF drawing, 3MF, STL, glTF, BOM export, CAD interoperability]
---
# Import and export

RunMat reads STEP into a part and writes every common mechanical exchange format. Exports come from the editor's Export command, from the agent (`design.export`) and from the command line (`runmat design export`), all through the same writers.

## Import

| Format | Becomes |
|---|---|
| STEP | an `import` feature in a part: `feature housing = import(source = "supplier/housing.step", digest = …)`. The body keeps the file's face names, read as `feature.housing.face("Top")`, and the digest pins the file's content. AP242 semantic PMI comes in as the part's PMI. |
| KiCad projects | circuits and boards; see [Import](/docs/ecad/import) |

Direct edits (`direct.move_faces`, `direct.replace_faces`, `direct.delete_faces`) change an imported body as features in the history, so the supplier's file stays untouched; see [Features](/docs/mcad/part/features#direct-edits).

### Recognize features

**Recognize features** on the Manage tab turns an imported body into features you can edit: a sketch and an extrude, cuts, holes, rounds and chamfers. Pick the body, and set the match tolerance (0.01 mm by default). The command recognizes, in this order:

- the body as a prism: two flat caps at each end of one axis (x, y or z) and side faces parallel to it. The bottom cap's outline becomes the sketch, and openings that run through the body become loops in it;
- holes: cylinders along that axis that run through the body, or open at a cap and end in a flat bottom or a drill point. Holes of the same size, depth and bottom on the same cap become one `hole` feature with a point per hole;
- pockets and steps: a flat floor between the caps, facing one of them, becomes a cut from that cap, with the floor's loops as its sketch;
- rounds and chamfers on the outline's corners: an arc tangent to the two lines it joins becomes a `fillet` with its radius, and a short line that cuts a convex corner at equal distances becomes a `chamfer`. Corners with the same radius or distance share one feature.

For a 60 x 40 x 10 mm plate with two holes and two rounded corners, the import

```
feature step = import(source = "supplier/plate.step", digest = sha256_9f2c…)
```

becomes

```part fragment
sketch outline on origin.xy {
    l1 = line(start = [0 mm, 0 mm], end = [60 mm, 0 mm])
    l2 = line(start = [60 mm, 0 mm], end = [60 mm, 40 mm])
    l3 = line(start = [60 mm, 40 mm], end = [0 mm, 40 mm])
    l4 = line(start = [0 mm, 40 mm], end = [0 mm, 0 mm])
}
feature base = extrude(profile = sketch.outline, distance = 10 mm)
feature corner_round = fillet(edges = [edges_between(feature.base.face.side(outline.l1), feature.base.face.side(outline.l2)), edges_between(feature.base.face.side(outline.l2), feature.base.face.side(outline.l3))], radius = 4 mm)
feature holes = hole(face = feature.base.face.end, at = { h1 = point(12 mm, 20 mm, 10 mm), h2 = point(48 mm, 20 mm, 10 mm) }, diameter = 6 mm, through_all)
```

Before it writes anything, the command rebuilds the features and compares the result with the import. The volumes must agree within the tolerance times the body's surface area, and each imported face needs a rebuilt face of the same kind whose centroid lies within the tolerance and whose area agrees within the tolerance times the face's perimeter. References to the imported body, such as an output's `body`, move to `feature.base.body`.

A body is converted whole or not at all, because the new features can't be built on the parts of an import that would stay. When a face fits none of the patterns, such as a round along a cap edge, a draft or a freeform surface, or when the rebuilt body doesn't match, the import stays as it is and the panel lists each face that was left and why. The STEP file stays in the project, and undo restores the `import` line.

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
runmat design export bracket.part --format step --out exports
runmat design export bracket.part --format step --all-configurations --out exports
runmat design export bracket.drawing --format pdf
runmat design export enclosure.part --format dxf --out flat
```

Region labels become STEP face names, which is how a study selects them: a region `label = "Mounting bores"` in the part is `select name "Mounting bores"` in the `.fea` file.

## Studies of a design

A study reads a part or an assembly's exported STEP today. A `.fea` geometry that names the design directly, `geometry arm { part = arm, inputs = { length = 92 mm } }`, regenerates it at the given inputs with no export step [not yet in the current release](pending:design.fea.part-geometry).
