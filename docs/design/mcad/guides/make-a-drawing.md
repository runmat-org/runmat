---
description: Make an engineering drawing in RunMat MCAD. Lay out a sheet and title block, place standard, projected, section and isometric views, dimension the part from its faces, show its holes and tolerances from the part's PMI, and export PDF and DXF.
keywords: [make a drawing, engineering drawing tutorial, 2D drawing, title block, section view, dimensions, hole callout, GD&T, PDF export, DXF export, auto drawing]
---
# Make a drawing

This guide draws the motor plate from [Model a part](/docs/mcad/guides/model-a-part): one A3 sheet with three orthographic views, a section through the bore, an isometric view, dimensions, hole callouts, a flatness frame and a title block. Every annotation names the model geometry it describes, so the drawing follows the part when the part changes.

## 1. The part and its PMI

The plate carries its own tolerances and finish as PMI (product manufacturing information). The drawing displays them; it does not restate them.

```part file=motor_plate.part
runmat 0.7
std 1

part motor_plate {
    input size : length = 60 mm
    input thickness : length = 5 mm

    sketch outline on origin.xy {
        bottom = line(start = [0 mm, 0 mm], end = [input.size, 0 mm])
        right = line(start = bottom.end, end = [input.size, input.size])
        top = line(start = right.end, end = [0 mm, input.size])
        left = line(start = top.end, end = bottom.start)
    }
    feature body = extrude(profile = sketch.outline, distance = input.thickness)
    feature bore = hole(face = feature.body.face.end, at = { c = point(30 mm, 30 mm, 5 mm) }, diameter = 22 mm, through_all)
    feature mount = hole(
        face = feature.body.face.end,
        at = { a = point(14 mm, 14 mm, 5 mm), b = point(46 mm, 14 mm, 5 mm), c = point(46 mm, 46 mm, 5 mm), d = point(14 mm, 46 mm, 5 mm) },
        through_all,
        spec = { standard = iso, size = M3 }
    )

    output motor_plate = { body = feature.body.body, part_number = "MP-060", material = "material.aluminum_6061" }

    datum A = { on = feature.body.face.start }
    tolerance seat_flat = { characteristic = flatness, on = [feature.body.face.start], tolerance = 0.05 mm }
    size bore = { on = [feature.bore.face.wall(c)], size = 22 mm ± [0.021 mm, 0 mm] }
    finish seat = { on = [feature.body.face.start], ra = 1.6 um }
}
```

## 2. A sheet and its title block

A `format` sets the sheet size and the title block's fields. A field reads literal text or a property of the model, so the part number and material come from the part's `output`, and each configuration's drawing shows its own.

```drawing file=motor_plate.drawing
runmat 0.7
std 1

drawing motor_plate {
    format a3 = {
        size = { size = iso, name = A3 },
        title_block = [
            { label = "Title", source = { source = literal, text = "Motor plate" } },
            { label = "Part number", source = { source = property, name = part_number } },
            { label = "Material", source = { source = property, name = material } },
            { label = "Scale", source = { source = scale } },
            { label = "Sheet", source = { source = sheet_number } },
            { label = "Rev", source = { source = revision } }
        ]
    }
    sheet "1" = { format = a3, scale = [2, 1], projection = third_angle }

    view front = standard(orientation = top, sheet = "1", model = { part = motor_plate }, at = [120, 180], display = hidden_lines)
    view side = projected(from = front, direction = right, sheet = "1", model = { part = motor_plate }, at = [260, 180], display = hidden_lines)
    view a_a = section(from = front, plane = feature.bore.face.wall(c), label = A, sheet = "1", model = { part = motor_plate }, at = [120, 70], display = hidden_lines)
    view iso = standard(orientation = isometric, sheet = "1", model = { part = motor_plate }, at = [320, 80], scale = [1, 1], display = shaded)

    annotation width = dimension(style = linear, references = [feature.body.face.side(outline.left), feature.body.face.side(outline.right)], view = front)
    annotation thickness = dimension(style = linear, references = [feature.body.face.start, feature.body.face.end], view = side)
    annotation bore_size = dimension(style = diameter, references = [feature.bore.face.wall(c)], pmi = bore, view = front)
    annotation bolts = hole_callout(hole = feature.mount.face.wall(a), view = front)
    annotation flat = gdt(tolerance = seat_flat, view = side)
    annotation datum_a = datum_symbol(label = A, view = side)
    annotation seat = surface_finish(on = feature.body.face.start, view = side)
    annotation n1 = note(text = "BREAK ALL SHARP EDGES. PART {part_number}.", view = iso)

    table holes = hole(view = front, origin = feature.body.face.side(outline.left), sheet = "1", at = [300, 260])
}
```

The title block's projection symbol shows the sheet's `third_angle` projection. A field can also hold a logo, `{ label = "", source = { source = image, file = "logo.svg" } }`, drawn from the SVG's outlines in the sheet and every export.

## 3. Views

- `standard(orientation = top)` looks down on the plate. `projected(from = front, direction = right)` projects a side view from it and stays aligned with it.
- `section(from = front, plane = …, label = A)` cuts through the bore and hatches the cut faces, with the cutting line drawn on the front view.
- Each view sets its position on the sheet in millimeters, and may set its own `scale`.

Moving a view on the canvas writes its new `at`. Views of an assembly add `exclude` to leave occurrences out and `exploded` to show an exploded state. See [Drawing views and annotations](/docs/mcad/drawing/views-and-annotations#views).

## 4. Dimensions and annotations

A dimension names the faces it measures, and its value is measured from the model. `pmi = bore` makes the bore's diameter show the tolerance written in the part. The hole callout reads the M3 clearance size from the hole feature, and the flatness frame reads `seat_flat`. Change the tolerance in the part and every drawing that shows it changes.

In the Desktop, Projected view takes a view picked on the sheet and places the new view where you click; it stays aligned with its parent. Smart dimension takes the edges you pick in a view and places the dimension where you click. Here a side view joins the hinge mount's front view, and the clevis height is dimensioned:

<!-- design:clip:mcad/drawing -->

A reference that stops resolving after an edit, such as a dimension to a face that was removed, is an error on that annotation, with the face it lost.

## 5. Let RunMat lay it out

The Auto-drawing command lays out a drawing for a part or an assembly in one step: three views, a section through the first hole, overall dimensions, hole callouts and tables, at the largest scale that fits the sheet. Start from it and move, add or remove views and annotations as needed. Auto-balloon numbers an assembly's occurrences against its BOM table.

## 6. Export

```sh
runmat-design export motor_plate.drawing --format pdf --out drawings
runmat-design export motor_plate.drawing --format dxf --out drawings
```

PDF, DXF and DWG write one file per sheet; `--sheet 1` writes one sheet, and `--all-configurations` writes every sheet once per configuration of the part.

## Next

- [Sheet metal](/docs/mcad/guides/sheet-metal) adds a flat-pattern view with its bend table.
- [Drawing views and annotations](/docs/mcad/drawing/views-and-annotations) lists every view, annotation and table.
