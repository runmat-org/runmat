---
description: Design a sheet-metal part in RunMat MCAD. Start from a base flange with its thickness, bend radius and K-factor, add edge flanges, corner reliefs and formed features, unfold it to a flat pattern, and draw the flat pattern with its bend table for the shop.
keywords: [sheet metal, sheet metal tutorial, base flange, edge flange, bend radius, K-factor, bend allowance, corner relief, louver, flat pattern, bend table, DXF flat pattern]
---
# Sheet metal

This guide designs a vent cover bent from 1.5 mm sheet: a 120 × 80 mm top with three flanges, relief cuts at the corners, two louvers, and its flat pattern drawn for the laser cutter and the press brake.

## 1. The base flange and its rules

A sheet-metal part starts with a base flange. A closed sketch makes a flat plate; an open chain of lines makes a bent profile, with a bend at each corner. The flange carries the rules every later bend uses:

- `thickness`: the sheet thickness;
- `bend_radius`: the inside bend radius;
- `rule`: how a bend develops into the flat. `k_factor(k = 0.44)` puts the neutral axis at 0.44 of the thickness from the inside face.

The bend allowance of a bend through angle $\theta$ with inside radius $R$, thickness $T$ and K-factor $K$ is

$$
BA = \theta \left( R + K\,T \right),
$$

so a 90° bend with $R$ = 2 mm, $T$ = 1.5 mm and $K$ = 0.44 develops 4.18 mm of flat.

### Gauges and bend tables

A shop orders sheet by gauge and bends it to its own bend table. The part keeps both as data, in one line that every sheet body of the part follows:

```
sheet_metal = { gauges = steel, gauge = "16", bend_table = din6935 }
```

**Gauge** on the Sheet Metal tab picks the gauge table and the gauge. The gauge sets each sheet body's thickness, inside bend radius and, when its rule is a K-factor, its K-factor. The built-in tables are:

| Table | Gauges | Source |
|---|---|---|
| `steel` | 3 to 28 | Manufacturers' Standard Gauge for sheet steel (16 gauge is 0.0598 in, 1.519 mm) |
| `aluminum` | 8 to 24 | Brown & Sharpe gauge, $d = 0.005\,\text{in} \times 92^{(36-n)/39}$ (ASTM B258), to four decimal places of an inch |

A built-in row bends at an inside radius equal to its thickness, with the DIN 6935 K-factor for that ratio, 0.325. A shop's own table is a CSV file in the project with the columns `gauge,thickness_mm,bend_radius_mm,k_factor`; choose **A table in the project** and name the file. An empty gauge goes back to each feature's own thickness and radius.

**Bend table** sets the rule every bend develops by:

- **K-factor**: one K-factor for every bend (`k = 0.38`).
- **DIN 6935**: the K-factor from each bend's radius-to-thickness ratio, $K = \tfrac{1}{2}\left(0.65 + 0.5 \log_{10} \tfrac{R}{T}\right)$, and $K = 0.5$ above $R/T = 5$.
- **Bend table in the project**: a CSV file with the columns `thickness_mm,radius_mm,angle_deg,allowance_mm`. A bend takes the allowance of the row for its thickness and radius, interpolated in angle. The flat pattern of a bend with no row for its thickness and radius fails, and the error names both.
- **Each feature's own rule**: the part line is removed, and each base flange's `rule` applies again.

A table file is part of what the part depends on: saving a change to it rebuilds the part and its flat pattern. A file that is missing or doesn't parse is reported in the bottom strip with its line number.

## 2. Flanges, reliefs and formed features

```part file=vent_cover.part
runmat 0.7
std 1

part vent_cover {
    sketch s_plate on origin.xy {
        e1 = line(start = [0 mm, 0 mm], end = [120 mm, 0 mm])
        e2 = line(start = [120 mm, 0 mm], end = [120 mm, 80 mm])
        e3 = line(start = [120 mm, 80 mm], end = [0 mm, 80 mm])
        e4 = line(start = [0 mm, 80 mm], end = [0 mm, 0 mm])
    }
    feature base = sheet_metal.base_flange(
        profile = sketch.s_plate,
        rules = { thickness = 1.5 mm, bend_radius = 2 mm, rule = k_factor(k = 0.44) }
    )
    feature front = sheet_metal.edge_flange(
        edges = [edges_between(feature.base.face.sheet(plate, ordinal = 1), feature.base.face.sheet_edge(e1))],
        length = 20 mm,
        angle = 90 deg,
        length_reference = tangent
    )
    feature side = sheet_metal.edge_flange(
        edges = [edges_between(feature.base.face.sheet(plate, ordinal = 1), feature.base.face.sheet_edge(e2))],
        length = 20 mm,
        angle = 90 deg,
        length_reference = tangent
    )
    feature back = sheet_metal.edge_flange(
        edges = [edges_between(feature.base.face.sheet(plate, ordinal = 1), feature.base.face.sheet_edge(e3))],
        length = 20 mm,
        angle = 90 deg,
        length_reference = tangent
    )
    feature relief = sheet_metal.corner_relief(
        corners = [feature.base.face.sheet(plate, ordinal = 1)],
        relief = rectangular(size = 3 mm)
    )
    feature louver1 = sheet_metal.form_tool(
        tool = louver(length = 50 mm, width = 6 mm, height = 4 mm),
        face = feature.base.face.sheet(plate, ordinal = 1),
        at = point(45 mm, 40 mm, 1.5 mm),
        rotation = 90 deg
    )
    feature louver2 = sheet_metal.form_tool(
        tool = louver(length = 50 mm, width = 6 mm, height = 4 mm),
        face = feature.base.face.sheet(plate, ordinal = 1),
        at = point(75 mm, 40 mm, 1.5 mm),
        rotation = 90 deg
    )
    feature flat = sheet_metal.flat_pattern(fixed_face = feature.base.face.sheet(plate, ordinal = 1))
    output vent_cover = { body = feature.base.body, part_number = "VC-120", material = "material.steel_dc01" }
}
```

- **Faces of a sheet** are named by the sketch entity they come from: `feature.base.face.sheet(plate, ordinal = 1)` is the plate's top face (ordinal 0 is the bottom), and `sheet_edge(e2)` is the side face along sketch line `e2`. An edge flange names its edge as the edge between those two faces.
- **`length_reference = tangent`** measures the flange length from the tangent point of the bend. `outer` and `inner` measure from the outside or inside face.
- **`corner_relief`** cuts a relief where two flanges meet, so the corner folds without tearing.
- **`form_tool`** presses a louver or a dimple into a face at a point.

Other operations add hems, jogs, miter flanges, sketched and lofted bends, tabs and slots, and closed or broken corners. See [Features](/docs/mcad/part/features#sheet-metal-weldments-and-molds).

`convert_to_sheet_metal` turns a thin solid, such as a shelled box, into sheet metal:

```part fragment
feature sheet = sheet_metal.convert_to_sheet_metal(
    fixed_face = feature.body.face.start,
    bend_edges = [],
    rules = { thickness = 1.5 mm, bend_radius = 1 mm, rule = k_factor(k = 0.44) }
)
```

- The **fixed face** is a flat face with straight edges and a face one sheet thickness behind it. Its outline becomes the plate.
- At each **bend edge** of the fixed face, the wall standing on that edge becomes a bend of the sheet's radius and a flat. The face the wall stood on stays where it was: pick the outer floor and the outer wall faces stay put; pick the inner floor and the inner faces stay put. With `bend_edges = []`, every edge with a wall bends. Walls on edges left out of a non-empty list are removed.
- Where two bent walls meet at a corner, each stops short of the corner by the other's bend setback, and the plate's corner inside both setbacks is cut away. The square relief appears in the folded part and in its flat pattern. For 90° walls the setback is the bend radius when the inner floor is fixed, and the bend radius plus the thickness when the outer floor is fixed.
- The converted plate unfolds, folds and has a flat pattern like a plate with edge flanges.

## 3. The flat pattern

`flat_pattern(fixed_face = …)` unfolds the part with the chosen face held still. The flat body sits beside the folded one and is kept out of the folded display. Each bend's development comes from the base flange's rule, so a change to the thickness or the K-factor changes the flat.

<!-- design:clip:mcad/sheet-metal -->

## 4. Draw it for the shop

A `flat_pattern` view draws the developed outline with its bend lines and a note on each bend (direction, angle, radius and allowance). A `bend` table lists the bends in the order of their lines.

```drawing file=vent_cover_flat.drawing
runmat 0.7
std 1

drawing vent_cover_flat {
    format a3 = {
        size = { size = iso, name = A3 },
        title_block = [
            { label = "Title", source = { source = literal, text = "Vent cover, flat pattern" } },
            { label = "Part number", source = { source = property, name = part_number } },
            { label = "Scale", source = { source = scale } }
        ]
    }
    sheet "1" = { format = a3, scale = [1, 1], projection = third_angle }
    view flat = flat_pattern(feature = flat, sheet = "1", model = { part = vent_cover }, at = [200, 170], display = visible_lines)
    table bends = bend(view = flat, sheet = "1", at = [20, 120])
}
```

Export the flat pattern as DXF for the cutter:

```sh
runmat design export vent_cover.part --format dxf --out flat
runmat design export vent_cover_flat.drawing --format pdf --out drawings
```

## 5. Check it for manufacture

Evaluate's design-for-manufacture check runs the sheet-metal rule pack, which checks bend radius over thickness among its rules, and reports each failure on its feature. See [Design for manufacture](/docs/mcad/evaluate#design-for-manufacture).

## Next

- [Make a drawing](/docs/mcad/guides/make-a-drawing) covers sheets, views and annotations in full.
- [Product families and configurations](/docs/mcad/guides/configurations) makes thicker or longer versions of the cover.
