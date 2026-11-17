---
description: Reference for what a RunMat part file exposes to other files and to manufacturing. Outputs and part numbers, materials, regions, published values, semantic PMI (datums, GD&T, size and location tolerances, surface finish), properties and links.
keywords: [part output, part number, material, regions, representation, defeature, midsurface, beam, publish, PMI, GD&T, datum, flatness, parallelism, tolerance, surface finish, coating, weld symbol, 3D annotation, model-based definition, MBD, annotation view, auto-dimension, traceability, links]
---
# Outputs, regions and published values

A part's feature history makes geometry. The statements on this page say what that geometry is to the rest of the project: which body is the part and what it is made of, which faces carry names, which values other files may read, and what the manufacturing tolerances are. This bracket uses each of them.

```part
runmat 0.7
std 1

part bracket {
    input width : length = 80 mm
    input depth : length = 50 mm
    input thickness : length = 10 mm

    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])
        l2 = line(start = [input.width, 0 mm], end = [input.width, input.depth])
        l3 = line(start = [input.width, input.depth], end = [0 mm, input.depth])
        l4 = line(start = [0 mm, input.depth], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = input.thickness)
    feature mount = hole(face = feature.body.face.end, at = { a = point(15 mm, 25 mm, 10 mm), b = point(65 mm, 25 mm, 10 mm) }, diameter = 6.6 mm, through_all)
    feature round = fillet(edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.end)], radius = 1.5 mm)

    output bracket = { body = feature.body.body, part_number = "BR-100", material = "material.aluminum_6061" }

    region top : face = { select = [feature.body.face.end], expect = exactly_one }
    region mounting : face = { select = [feature.mount.face.wall(a), feature.mount.face.wall(b)], expect = at_least_one, label = "Mounting bores" }
    publish top = region(region = top)
    publish thickness = value(input = thickness)

    datum A = { on = feature.body.face.start }
    datum B = { on = feature.body.face.side(s1.l1) }
    tolerance top_flat = { characteristic = flatness, on = [feature.body.face.end], tolerance = 0.05 mm }
    tolerance top_parallel = { characteristic = parallelism, on = [feature.body.face.end], tolerance = 0.1 mm, datums = [{ label = A, modifier = rfs }] }
    size bore = { on = [feature.mount.face.wall(a)], size = 6.6 mm ± [0.1 mm, 0 mm] }
    location bore_to_edge = { from = feature.body.face.side(s1.l4), to = feature.mount.face.wall(a), distance = 15 mm ± 0.1 mm }
    finish seat = { on = [feature.body.face.start], ra = 1.6 um }

    configuration long = { width = 120 mm, part_number = "BR-120" }
    configuration plain = { suppress = [feature.round], part_number = "BR-100-P" }
    configuration thick = { features = { body = { distance = 12 mm } }, part_number = "BR-100-T" }

    link verifies feature.body to "checks/bracket_static.fea#check.max_stress"
}
```

## Outputs

`output <name> = { … }` declares a part the file delivers: the body, and the identity the BOM, title blocks and STEP carry. A file may declare several outputs, such as a fluid body beside the solid it fills.

| Field | Meaning |
|---|---|
| `body` | The body: `feature.<name>.body`. |
| `part_number` | The part number. A configuration can set its own. |
| `material` | The material, as `"material.<name>"`. Mass properties read its density. |
| `color` | The display color, `"#RRGGBB"` or `"#RRGGBBAA"`, when it differs from the material's appearance. |
| `label` | The product name in a STEP export, when it differs from the output's name. A study selects the body by it. |
| `properties` | Typed properties, such as `density = 1450 kg/m^3` for a body whose material the density table does not know. |

`material = { body = feature.<name>.body, material = "material.<name>" }` sets the material of one body; bodies with no assignment take their output's material. Mass properties match the material name's words against a table of common alloys, plastics and composites (`aluminum_6061`, `steel_300m`, `carbon_prepreg`, `nylon`); a `density` property on the output overrides the table.

## Regions

A region names a selection and checks its count. The selection forms and `expect` values are in [Names and selectors](/docs/design-language/names-and-selectors#regions). Regions travel with the geometry: a STEP export names the faces by the region's `label` (or its name), so a study writes `select name "Mounting bores"` and keeps working when the part changes.

To add a region in the editor, run **Simulate › Region**. Give it a name, choose the kind (faces, edges, bodies or vertices), pick the items in the viewport and choose how many the region expects. The command writes one `region` line into the part and the region appears in the Outline's Regions folder. Picks of another kind, a name the part already uses, or a count that `expect = exactly_one` does not allow are refused in the bottom strip, and nothing is written.

A study reads the part's face regions by name. With the `top` region above, this study geometry selects the face:

```fea fragment
geometry bracket {
    file = "./bracket.part"
    region top : face { select name "top"; expect exactly_one }
}
```

A study reads a body by its output's name (`select name "bracket"`). Edge and vertex regions name items for drawings and assemblies; a study does not read them yet.

## Analysis representations

A representation is simplified geometry for meshing. The part's source declares it, and RunMat builds it from the regenerated part each time it is drawn. The design's features stay as they are.

```part fragment
representation simple : defeatured = { max_fillet = 2 mm, max_hole = 8 mm, max_chamfer = 1 mm }
representation shell : midsurface = { body = feature.body.body }
representation spar : beam = { body = feature.body.body }
```

| Kind | What RunMat builds | Refused when |
|---|---|---|
| `defeatured` | The bodies with every fillet at or below `max_fillet` (radius), hole at or below `max_hole` (diameter) and chamfer at or below `max_chamfer` (distance) removed. The kernel extends the neighboring faces over each gap. A hole is a hole feature's faces or any full concave cylinder, so cut and imported holes count too. | No threshold is set. |
| `midsurface` | The kernel's mid-surface of the body, from every pair of opposite parallel planar faces closer than half the width of their overlap, with each pair's wall thickness. | The body has no such pair: it isn't a thin wall. |
| `beam` | The centerline of a prismatic body between the centroids of its two end faces, and the end section's area and principal second moments of area. | The body isn't prismatic: the message names the face that isn't parallel to the line between the end faces, or says the end faces differ in area. |

`body` names the body a mid-surface or beam is made from; without it the part must have one body. `keep_regions` (on by default) keeps every face a region holds, so a study's loads and supports keep their faces; a fillet or hole kept this way is listed with the region's name. A fillet, hole or chamfer the kernel cannot remove stays, and is listed with the kernel's reason.

In the editor, **Simulate › Defeature** adds a `defeatured` representation and **Simulate › Representation** adds any of the three. Each is built before it is written, so a beam of a body that isn't prismatic is refused and nothing is written. The new representation is drawn at once. The **Representation** menu in the bottom strip switches between **full** (the design) and each representation; beside it, the strip lists what the drawn one removed, its wall thicknesses or its beam section, or why it could not be built.

Each representation has a row in the Outline's **Representations** folder. Select a defeatured representation's row to change its thresholds or **Keep named regions** in Properties; RunMat rebuilds it at once and refuses a change it cannot build. A mid-surface or beam has no thresholds to edit. Right-click a row and choose **Delete** to remove the representation. If it was the one drawn, the strip goes back to **full**.

A study reads a part in its full representation. A `.fea` geometry cannot yet select a part's `defeatured`, `midsurface` or `beam` representation, so meshing one of them in a study is not available yet.

## Published values

`publish` makes an item readable from other files by a stable name:

| Form | Publishes | Read elsewhere as |
|---|---|---|
| `publish top = region(region = top)` | a region | an assembly's attachment, a drawing's references |
| `publish thickness = value(input = thickness)` | an input's value, with its unit | `part.bracket.publish.thickness` |
| `publish half = let(name = half)` | a let's value | `part.bracket.publish.half` |

A skeleton part publishes the frames and values a whole assembly is laid out from; parts and assemblies read them, and a change to the skeleton flows to every reader. Across project partitions a read is pinned to a released revision (`use part skeleton from "//wing/skeleton" @ "R3" as sk`).

## PMI

PMI is part of the part, attached by the same references as features and regions. Drawings show it (a GD&T frame on a view names a tolerance here rather than restating it), and STEP AP242 exports carry it as semantic PMI.

| Statement | Declares | Fields |
|---|---|---|
| `datum A = { on = … }` | a datum feature | `on`: a face |
| `tolerance <name> = { … }` | a geometric tolerance | `characteristic`, `on`, `tolerance`, `datums`, `diametral`, `modifier`, `projected` |
| `size <name> = { … }` | a toleranced size or a fit | `on`, `size` (a toleranced value), `fit` (such as `"H7"`) |
| `location <name> = { … }` | a toleranced distance between two features | `from`, `to`, `distance` |
| `finish <name> = { … }` | a surface finish | `on`, `ra`, `removal` (`any`, `required`, `prohibited`), `process` (the method, such as `"ground"`), `lay` |
| `treatment <name> = { … }` | a coating or treatment | `on` (faces; none for the whole part), `specification` |

Characteristics: `straightness`, `flatness`, `circularity`, `cylindricity` (form; no datums); `profile_of_line`, `profile_of_surface`; `parallelism`, `perpendicularity`, `angularity` (orientation); `position`, `concentricity`, `symmetry` (location); `circular_runout`, `total_runout`. Orientation, location and runout tolerances need at least one datum. A datum reference is `{ label = A, modifier = rfs }`, with the material condition `rfs`, `mmc` or `lmc`.

The check resolves every attachment against the regenerated part and checks the semantics: datum labels unique and declared, form tolerances without datums, orientation and location tolerances with them, and tolerance values of the right dimension.

## 3D annotations

A part shows its PMI in the 3D model, with notes and weld symbols beside it (model-based definition). Each annotation is an `annotation` statement in the drawing's annotation form, placed in the model instead of on a sheet. A datum symbol, GD&T frame, dimension, finish or coating names the PMI item it shows, so the part keeps one declaration that its drawings, its model view and STEP AP242 all read. Annotations attach to the part's face and edge names and follow the geometry through every rebuild; one whose reference stops resolving is an error on the annotation.

```runmat
runmat 0.7
std 1

part plate {
    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])
        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])
        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = 10 mm)
    feature h = hole(face = feature.body.face.end, at = { a = point(60 mm, 20 mm, 10 mm) }, diameter = 8 mm, through_all)

    datum A = { on = feature.body.face.start }
    datum B = { on = feature.body.face.side(s1.l4) }
    tolerance position1 = { characteristic = position, on = [feature.h.face.wall(a)], tolerance = 0.2 mm, diametral = true, modifier = mmc, datums = [{ label = A, modifier = rfs }, { label = B, modifier = rfs }] }
    size size1 = { on = [feature.h.face.wall(a)], size = 8 mm ± 0.1 mm }
    treatment coating1 = { specification = "anodize type II, black" }

    annotation datum1 = datum_symbol(label = "A", view = front_pmi)
    annotation datum2 = datum_symbol(label = "B", view = front_pmi)
    annotation gdt1 = gdt(tolerance = position1, view = front_pmi, at = [0, 0, 25])
    annotation dim1 = dimension(style = diameter, references = [feature.h.face.wall(a)], pmi = size1)
    annotation coating1 = coating(treatment = coating1)
    annotation weld1 = weld(on = edges_between(feature.body.face.side(s1.l1), feature.body.face.end), symbol = "fillet", size = 5 mm, side = both, tail = "GMAW")
    annotation note1 = note(text = "Break sharp edges", leader = feature.body.face.side(s1.l1))

    view front_pmi = { from = [0, -1, 0], up = [0, 0, 1] }
}
```

| Statement | Shows | Fields |
|---|---|---|
| `datum_symbol(label)` | a datum feature symbol | the `datum` it shows |
| `gdt(tolerance)` | a feature control frame | the `tolerance` it shows |
| `dimension(style, references, pmi)` | a hole's diameter (`diameter`) or the distance between two faces (`linear`), with the `size` or `location` that tolerances it | `references`: one face or two |
| `surface_finish(on)` | the finish on a face | the `finish` on `on` |
| `coating(treatment)` | a coating's specification | the `treatment` it shows |
| `weld(on, symbol, size, side, all_around, field, length, pitch, tail)` | a weld symbol on an edge | `symbol`: `fillet`, `groove`, `plug` or `spot`; `side`: `arrow`, `other` or `both`; `length` and `pitch` for an intermittent weld; `tail`: the process |
| `note(text, leader)` | a note, with a leader to a face, edge or vertex | `text`, `leader` |

Every annotation takes `view`, the annotation view it is filed under, and `at`, its label's offset from the point it is attached to in millimetres along the part's axes. Without `at`, the label sits off the part, away from its center.

An annotation view (`view <name> = { from, up }`) is a saved view direction: `from` points from the model toward the eye, `up` stays upright, and `default = true` opens the part on it. Selecting a view in the Outline turns the model to it and shows only its annotations.

### Annotate commands

| Command | Writes |
|---|---|
| `annotate.datum` | a datum on a face or axis (labels are capital letters other than I, O and Q) and its symbol |
| `annotate.gdt` | a geometric tolerance on faces and its frame; form tolerances take no datums, orientation, location and runout tolerances need them, and a material condition goes only on straightness, orientation tolerances and position |
| `annotate.dimension` | a size (one face: a hole's or boss's diameter) or a location (two parallel faces, a face and a hole, or two holes) at the measured value, with no tolerance, a symmetric, bilateral or limit tolerance, or a fit, and its dimension |
| `annotate.auto-dimension` | datums A, B and C (the picked faces, else the part's datums, else its largest planar face and the largest faces square to it), then for each hole its diameter and either its location from each datum (plus and minus scheme) or a position frame at maximum material condition (geometric scheme); each other planar face its distance from a parallel datum, or a profile of a surface. Run again, it adds only what is missing. |
| `annotate.surface-finish` | a finish (Ra in micrometres, the material-removal requirement, the method and the lay) on faces and its symbol |
| `annotate.coating` | a treatment on faces, or the whole part, and its note |
| `annotate.weld-symbol` | a weld symbol on each picked edge |
| `annotate.note` | a note, with or without a leader |
| `annotate.annotation-view` | an annotation view from a standard orientation or the current view, with the picked annotations (or every annotation not yet in a view) filed under it |

Select an annotation in the Outline or the model to edit its definition in Properties: the change rewrites the PMI item and the annotation in place, under the same names. Drag a label in the model to move it. Delete removes the annotation with the PMI item only it shows; a datum that a frame uses stays until the frame is deleted or changed.

## Links

`link <kind> <item> to "<target>"` records a traceable relation from an item in this file to another file:

| Kind | Meaning |
|---|---|
| `verifies` | A study or check verifies the item: `link verifies feature.body to "checks/bracket_static.fea#check.max_stress"`. |
| `derived_from` | The item was derived from the target, such as a supplier model. |
| `supersedes` | The item replaces the target. |
| `supplier_doc` | The target is the supplier's document for the item. |

The item is a feature (`feature.body`), a region, a published name or an output.
