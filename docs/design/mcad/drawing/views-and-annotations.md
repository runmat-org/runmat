---
description: Reference for RunMat drawing files. Sheet formats and title blocks, standard, projected, section, detail, auxiliary and broken-out views, dimensions, hole callouts, GD&T frames from the part's PMI, surface finish and weld symbols, wrapped notes and general notes, balloons, level marks, plan grids and door swings, and BOM, hole, revision, bend, weld and cut-list tables.
keywords: [engineering drawing, 2D drawing, title block, section view, detail view, dimensions, GD&T, feature control frame, surface finish symbol, BOM table, hole table, revision table, cut list, third angle projection]
---
# Drawing views and annotations

A `.drawing` file lays out sheets of views of a part or an assembly, and annotates them by reference to the model. A dimension names the faces it measures and a GD&T frame names a tolerance in the part's PMI, so the drawing follows the model as it changes and a reference that stops resolving is an error on the annotation.

```part file=bracket.part
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

```drawing file=bracket.drawing
runmat 0.7
std 1

drawing bracket {
    format a3 = {
        size = { size = iso, name = A3 },
        title_block = [
            { label = "Title", source = { source = literal, text = "Mounting bracket" } },
            { label = "Part number", source = { source = property, name = part_number } },
            { label = "Material", source = { source = property, name = material } },
            { label = "Scale", source = { source = scale } },
            { label = "Sheet", source = { source = sheet_number } },
            { label = "Rev", source = { source = revision } }
        ]
    }

    sheet "1" = { format = a3, scale = [1, 1], projection = third_angle }

    view front = standard(orientation = front, sheet = "1", model = { part = bracket }, at = [150, 200], display = hidden_lines)
    view top = projected(from = front, direction = up, sheet = "1", model = { part = bracket }, at = [150, 100], display = hidden_lines)
    view iso = standard(orientation = isometric, sheet = "1", model = { part = bracket }, at = [330, 120], display = visible_lines)

    annotation thickness = dimension(style = linear, references = [feature.body.face.start, feature.body.face.end], view = front)
    annotation bore_size = hole_callout(hole = feature.mount.face.wall(a), view = top)
    annotation flat = gdt(tolerance = top_flat, view = front)
    annotation datum_a = datum_symbol(label = A, view = front)
    annotation seat = surface_finish(on = feature.body.face.start, view = front)
    annotation n1 = note(text = "PART {part_number} REV {revision}", view = top)

    table holes = hole(view = top, origin = feature.body.face.side(s1.l4), sheet = "1", at = [20, 40])
}
```

## Formats and sheets

`format <name> = { size, title_block }` defines a sheet format:

- `size = { size = iso, name = A3 }` for an ISO size, `size = ansi` for an ANSI size, or `custom` dimensions.
- `title_block` lists fields, each a label and a source: `literal` text, a `property` of the model (`part_number`, `material`, `description`, `drawn_by`), the `revision`, the `sheet_number`, the `sheet_count`, the `scale`, or an `image`: `{ label = "", source = { source = image, file = "logo.svg" } }` draws an SVG logo's outlines in its cell. A property reads the active configuration's value, so each configuration's drawing carries its own part number.
- The title block is laid out once (ISO 7200: 180 mm wide with 10 mm rows, a second column once it has more than four fields, and the projection symbol beside it) and drawn the same way on the sheet and in every export.

`sheet "<name>" = { format, scale = [1, 2], projection = third_angle }` adds a sheet; `projection` is `first_angle` or `third_angle`.

## Views

`view <name> = <kind>(…, sheet, model, at, scale, display)`. Every view names its sheet, its model (`model = { part = bracket }`, `model = { assembly = drone, from = ".." }`, `model = { part = bracket, from = "../bracket.part" }`), its position on the sheet in millimeters and its display (`visible_lines`, `hidden_lines` or `shaded`), and optionally its own scale. `exclude` leaves occurrences out of an assembly view, and `exploded` shows a named exploded state.

| Kind | Parameters | Shows |
|---|---|---|
| `standard` | `orientation` (`front`, `back`, `top`, `bottom`, `left`, `right`, `isometric`, `dimetric`, `trimetric`) or `frame` | a model view |
| `projected` | `from`, `direction` (`up`, `down`, `left`, `right`) | a view projected from another |
| `section` | `from`, `plane`, `label`, `aligned` | a cut through the model by a face or datum plane |
| `detail` | `from`, `center`, `radius`, `label` | an enlarged circle of another view |
| `auxiliary` | `from`, `edge`, `label` | a view perpendicular to a model edge |
| `broken_out` | `from`, `boundary`, `depth` | material removed inside a boundary |

A view's `from` names the folder that holds the design (`".."`) or the design's file (`"../bracket.part"`). With the file form, `part =` is optional, and the view shows the part that file declares. A drawing kept in a `drawings/` folder reaches the bracket one level up either way:

```drawing file=drawings/bracket_shop.drawing
runmat 0.7
std 1

drawing bracket_shop {
    format a3 = {
        size = { size = iso, name = A3 },
        title_block = [{ label = "Sheet", source = { source = sheet_number } }]
    }
    sheet "1" = { format = a3, scale = [1, 1], projection = third_angle }
    view front = standard(orientation = front, sheet = "1", model = { part = bracket, from = "../bracket.part" }, at = [150, 165], display = hidden_lines)
    view top = standard(orientation = top, sheet = "1", model = { part = bracket, from = ".." }, at = [150, 80], display = hidden_lines)
}
```

```drawing fragment
view a_a = section(from = front, plane = feature.body.face.start, label = A, sheet = "1", model = { part = bracket }, at = [100, 100], display = hidden_lines)
view b = detail(from = a_a, center = feature.body.face.end, radius = 20 mm, label = B, sheet = "1", model = { part = bracket }, at = [300, 100], scale = [2, 1], display = hidden_lines)
```

## Annotations

`annotation <name> = <kind>(…, view, at)`:

| Kind | Parameters | Draws |
|---|---|---|
| `dimension` | `style`, `references`, `pmi`, `group`, `reference`, `condition` | a dimension: `linear`, `aligned`, `angular`, `radius`, `diameter`, `ordinate`, `baseline` or `chain`. `pmi` names a `size` in the part, whose tolerance the dimension shows; `reference` makes it a driven dimension in parentheses. |
| `hole_callout` | `hole` | the hole's size, depth and form from the hole feature |
| `gdt` | `tolerance` | the feature control frame of a tolerance in the part's PMI |
| `datum_symbol`, `datum_target` | `label` | a datum symbol or target from the part's PMI |
| `surface_finish` | `on` | the finish symbol of the face's `finish` |
| `weld` | `on`, `symbol`, `size`, `all_around`, `field` | a weld symbol |
| `note` | `text`, `leader`, `width` | text; `{part_number}`, `{revision}` and other `{property}` fields fill in. The text wraps at spaces to `width` (a length on the sheet, such as `60 mm`), otherwise to the room it has; `\n` starts a new line |
| `balloon` | `occurrence`, `table` | a balloon numbered by a BOM table |
| `center_mark`, `centerline` | `on`, `between` | center marks and centerlines |
| `level` | `label`, `on` or `datum`, `occurrence` | a level mark: a line from a face's or a datum plane's height to a target symbol, the label above the line and the elevation below it |
| `grids` | `occurrence`, `datums`, `labels` | grid lines across a plan, each label in a bubble, and a row of dimensions between neighboring grids |
| `door_swing` | `door`, `hinge`, `swing` | a door's leaf open 90° from its hinge and the quarter arc it sweeps |

Annotations read PMI from the part instead of restating it: change the flatness tolerance in the `.part` and the frame on every drawing changes with it.

### Annotations on assembly views

On an assembly view, a reference names a face of one occurrence's part. A dimension lists the occurrence of each reference in `occurrences`, one path per reference and in the same order. A level mark names its face's occurrence in `occurrence`, or names a datum plane that occurrence publishes in `datum` (a skeleton's `second_floor`). The drawing measures each face where the assembly places that occurrence, so moving a wall in the assembly moves its dimension and changes the value.

```drawing fragment
annotation overall_ns = dimension(style = linear, references = [feature.top1.face.start, feature.top1.face.start], occurrences = [[{ name = framing }, { name = wall_a_l1 }], [{ name = framing }, { name = wall_d_l1 }]], view = first)
annotation slab = level(label = "T.O. SLAB", on = feature.slab.face.end, occurrence = [{ name = foundation }, { name = slab }], view = south, at = [60, 253])
annotation floor_2 = level(label = "2ND FLOOR", datum = second_floor, occurrence = [{ name = skeleton }], view = south)
```

A level mark prints the level's height above the model's origin in the sheet's units. In feet and inches the elevation always carries its feet: `EL. 0'-0"`, `EL. -0'-8"`, `EL. 10'-1 3/4"`.

`grids` reads the datum planes an occurrence's part publishes (`publish grid_1 = datum(…)` in a skeleton) and pairs each with a label. Each grid line runs across the view as a chain line and ends in a bubble beyond the view's dimension rows. Grids measured left to right put their bubbles on the side the view's horizontal dimensions use; grids measured bottom to top use the vertical dimensions' side. The dimensions between neighboring grids join the view's dimension rows as a chain.

`door_swing` draws from the door occurrence's own geometry. The leaf is the door's thinnest body across the wall (the slab, not the frame), and the leaf is as wide as that body. `hinge = start` hangs the leaf at the door's x = 0, `hinge = end` at the far end of the opening. `swing = positive` opens it toward the door's +y, `swing = negative` toward -y.

```drawing fragment
annotation grids_ew = grids(occurrence = [{ name = skeleton }], datums = [grid_1, grid_2, grid_3, grid_4, grid_5], labels = ["1", "2", "3", "4", "5"], view = first)
annotation swing_d1 = door_swing(door = [{ name = envelope }, { name = d1_a_l1_2 }], hinge = start, swing = positive, view = first)
```

## Tables

`table <name> = <kind>(…, sheet, at)`. `at` is the table's corner on the sheet when you place it by hand, `[0, 0]` included; leave it out and the table takes its standard place.

| Kind | Lists |
|---|---|
| `bom(view, include_fasteners)` | the bill of materials of an assembly view, with item numbers the balloons use |
| `hole(view, origin)` | every hole in a view with its position from an origin face, size and form |
| `revision(rows)` | revisions: `rows = [{ rev = "B", description = "…", date = "2026-11-05", by = "NA" }]` |
| `notes(rows, width)` | the general notes: a numbered list under "NOTES:", one note per row, each wrapping with a hanging indent in a column `width` wide (the title block's width when unset) |
| `bend(view)` | the bends of a sheet-metal flat pattern |
| `weld(view)` | the welds in a view |
| `cut_list(view)` | a weldment's structural members, one row per group of identical members: item, quantity, profile, material, cut length and end angles |

```drawing fragment
table bom = bom(view = front, include_fasteners, sheet = "1")
table revs = revision(rows = [{ rev = "A", description = "First release", date = "2026-11-05", by = "NA" }], sheet = "1", at = [300, 280])
```

### General notes

Sheet-wide notes go in one `notes` table, with one note per row. Write each note as one string; the sheet wraps it. Unless placed with `at`, the list sits on the title block's top edge and grows upward. A `{property}` field fills in from the model of the sheet's first view.

```drawing fragment
table notes = notes(
    rows = [
        "MATERIAL: {material}.",
        "BREAK SHARP EDGES 0.2 TO 0.5 MM; DEBURR ALL HOLES BOTH SIDES.",
        "GENERAL TOLERANCES: ISO 2768-mK WHERE NO INDIVIDUAL TOLERANCE IS SHOWN."
    ],
    sheet = "1"
)
```

### Notes on the sheet

Notes, tables and views never overprint each other, the title block or the border:

- A note placed with `at` wraps to its `width`. Without a width, it wraps at the room to its right, up to the border, the title block or the next item.
- A note with no room at its `at` moves to the nearest free spot, and the check warns: "Note n2 at [20, 30] has no room there: …".
- A note or table that fits nowhere on the sheet stays where it was put, and the check warns with what to change: "Note n2 doesn't fit on the sheet: it needs 200 x 60 mm of free space …".

The sheet sets note text in capitals, as ASME Y14.2 and ISO 3098 lettering expect, with filled `{property}` fields included: `MATERIAL: {material}.` prints `MATERIAL: ALUMINUM 6061.` for a part whose material is `Aluminum 6061`. A unit symbol written after a number keeps the case it was written in, so `3 mm` and `25kg` print as written. The file and the model's properties keep their own case.

The sheet, the PDF and the DXF break notes at the same words. Each line is its own text entity in the DXF and its own line of text in the PDF.

## Laid out for you

The Auto-drawing command (`drawing.auto`) writes a drawing for a part or an assembly: three orthographic views, a section through the first hole, overall dimensions, hole callouts and tables, at the largest scale that fits the chosen sheet (A4 to A2, ANSI B or D). Fasteners are left out of assembly views. Auto-balloon numbers an assembly view's occurrences against its BOM table. The result is an ordinary `.drawing` to edit.

## Exports

`runmat design export bracket.drawing --format pdf` writes the sheets as PDF, and `--format dxf` or `dwg` as CAD drawings, one file per sheet (`--sheet <name>` for one). See [Import and export](/docs/mcad/import-export).

## Not yet built

- **Drawing standards:** ISO and ASME presets for line weights, text, arrowheads and tolerance display [not yet in the current release](pending:design.mcad.drawing.standards).
