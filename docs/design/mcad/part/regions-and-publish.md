---
description: Reference for what a RunMat part file exposes to other files and to manufacturing. Outputs and part numbers, materials, regions, published values, semantic PMI (datums, GD&T, size and location tolerances, surface finish), properties and links.
keywords: [part output, part number, material, regions, publish, PMI, GD&T, datum, flatness, parallelism, tolerance, surface finish, traceability, links]
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
| `finish <name> = { … }` | a surface finish | `on`, `ra`, `process` |

Characteristics: `straightness`, `flatness`, `circularity`, `cylindricity` (form; no datums); `profile_of_line`, `profile_of_surface`; `parallelism`, `perpendicularity`, `angularity` (orientation); `position`, `concentricity`, `symmetry` (location); `circular_runout`, `total_runout`. Orientation, location and runout tolerances need at least one datum. A datum reference is `{ label = A, modifier = rfs }`, with the material condition `rfs`, `mmc` or `lmc`.

The check resolves every attachment against the regenerated part and checks the semantics: datum labels unique and declared, form tolerances without datums, orientation and location tolerances with them, and tolerance values of the right dimension.

## Links

`link <kind> <item> to "<target>"` records a traceable relation from an item in this file to another file:

| Kind | Meaning |
|---|---|
| `verifies` | A study or check verifies the item: `link verifies feature.body to "checks/bracket_static.fea#check.max_stress"`. |
| `derived_from` | The item was derived from the target, such as a supplier model. |
| `supersedes` | The item replaces the target. |
| `supplier_doc` | The target is the supplier's document for the item. |

The item is a feature (`feature.body`), a region, a published name or an output.
