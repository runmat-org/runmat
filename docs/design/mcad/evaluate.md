---
description: Evaluate a RunMat part or assembly. Mass, center of gravity and inertia from materials, interference and clearance between parts, weldment cut lists, and design-for-manufacture checks for machining, sheet metal, molding and additive.
keywords: [check geometry, wall thickness map, curvature, zebra stripes, surface continuity, deviation, short edges, invalid faces, self-intersection, mass properties, center of gravity, moment of inertia, interference detection, clearance check, cut list, weldment, DFM, design for manufacturing, wall thickness, draft analysis]
---
# Evaluate

Evaluate measures a part or an assembly in its active configuration. The Desktop lists each result under the canvas; the agent reads the same results through `design.inspect`, and checks can gate on them.

| Command | Works on | Result |
|---|---|---|
| `evaluate.mass-properties` | a part or an assembly | mass, center of gravity and inertia about it |
| `evaluate.interference` | an assembly | pairs of occurrences whose solids overlap, with the overlap volume |
| `evaluate.clearance` | an assembly | pairs closer than a gap you set |
| `weldment.cut-list` | a weldment part | its structural members, grouped |
| `evaluate.draft` | a part | each face's draft angle for a pull direction, in bands, with the model colored by band |
| `evaluate.section-properties` | planar faces of a part, or a sketch | area, centroid, second, polar and principal moments of area |
| `evaluate.check-geometry` | a part | invalid faces, edges and vertices, open and short edges, gaps and self-intersections, each on a named face, edge or vertex |
| `evaluate.thickness` | a part | the wall thickness over every face, with the model colored by it and the faces thinner or thicker than a target |
| `evaluate.curvature` | a part | the curvature over every face, with the model colored by it and each face's smallest radius |
| `evaluate.zebra` | a part | black and white bands that show where faces meet tangentially, and each edge's continuity (G0, G1, G2) |
| `evaluate.deviation` | edges of a part | the angle between the faces along each edge |
| `evaluate.dfm` | a part | the part's manufacturing features checked by a process's rule pack and a company pack, with each rule's limit and measured value |
| `evaluate.cost` | a part | the cost per part and per batch from the shop's rates you enter |
| `simulate.check` | a part or an assembly | the study checks: densities, overlaps and open contacts, before a study meshes |

## Mass properties

A part weighs its output bodies at their density: the output's `density` property when it has one, else its material's density from a table of common alloys, plastics and composites matched by the words in the material's name (`material.aluminum_7075`, `steel_300m`, `carbon_prepreg`, `nylon`, `fr4`). An assembly adds every placed part occurrence at its world placement: its center of gravity is mass-weighted, and its inertia is taken about that center by the parallel-axis theorem,

$$
I_{cg} = \sum_i \left( I_i + m_i \left( |d_i|^2 E - d_i d_i^{T} \right) \right),
$$

where $m_i$ and $I_i$ are each occurrence's mass and inertia about its own center, $d_i$ is the offset of that center from the assembly's, and $E$ is the identity. The result lists parts with no density, so a missing material never hides as a light assembly.

<!-- design:clip:mcad/evaluate -->

## Draft analysis

Draft analysis takes a pull direction (an origin plane or axis, or a picked face) and a minimum draft angle, 1° by default, and sorts every face by its angle to the pull:

| Band | Color | Meaning |
|---|---|---|
| Positive | green | drafts toward the pull by at least the minimum |
| Negative | blue | drafts away from the pull by at least the minimum, so it releases from the other mold half |
| Needs draft | red | some of the face is within the minimum of parallel to the pull; a warning |
| Straddles | amber | crosses the parting line, so part of the face drafts each way |

The canvas colors each face by its band, with a legend of the face counts, and the result's rows name each face.

## Section properties

Section properties takes planar faces of one body, which must lie in one plane, or a sketch, or one region of a sketch, and reports:

| Property | Unit | Meaning |
|---|---|---|
| Area | mm² | the region's area; a sketch's regions add, and their holes subtract |
| Centroid | mm | the area's center, in world coordinates |
| Ixx, Iyy | mm⁴ | second moments of area about the x and y axes through the centroid, $I_{xx} = \int y^2 \, dA$ |
| Ixy | mm⁴ | the product of area, $\int x y \, dA$ |
| Polar moment J | mm⁴ | $I_{xx} + I_{yy}$, about the normal through the centroid |
| Principal moments I1, I2 | mm⁴ | the largest and smallest second moments over every axis through the centroid |
| Angle | degrees | from the x axis to the axis of I1 |

For a face, x is world X projected into the face's plane (world Y when the plane is square to X) and y is the normal crossed with x; for a sketch, x and y are the sketch's own axes. The result's header shows both axes. A face's properties come from the kernel's exact face; a sketch's are integrated from its boundary, with 4096 points round a circle, so a circle's area is within 2 parts in a million of $\pi r^2$. A curved face is refused.

## Check geometry

Check geometry finds what makes a later feature, an export or a mesh fail, on the exact geometry of every body of the part, or of the bodies you pick:

| Finding | Severity | What it means |
|---|---|---|
| Invalid face, edge or vertex | error | the entity's geometry or topology breaks the kernel's validity rules, for example a face whose boundary doesn't lie on its surface |
| Invalid shell or solid | error | the body's structure is wrong (orientation, closure) while each face, edge and vertex is valid |
| Open edge | error | an edge of a solid that bounds only one face, so the solid isn't closed there |
| Self-intersecting face | error | a face crosses itself or another face of the body; found only by a strict check, which takes longer |
| Short edge | warning | an edge shorter than the length you set, 0.01 mm by default |
| Edge gap, vertex gap | warning | an edge or vertex whose tolerance, the gap it bridges between its faces, is over the limit you set, 0.01 mm by default |
| Smallest radius | information | the smallest radius of curvature on the body, and the face it is on |
| Largest edge gap, largest vertex gap | information | the largest tolerances on the body |

Each row names its body and its face, edge or vertex; selecting a row selects that entity in the canvas. Faces with an error or a warning are colored red or amber while the result is open. The result's header lists the settings it ran with, such as `Check: strict` and `Short edges under 0.01000 mm`.

The agent runs the same check with `design.propose_edit` and the command `evaluate.check-geometry`, and the CLI with `runmat design run`; the result's `data.findings` lists each finding with its problem, its entity's canonical reference and its value in metres.

## Interference and clearance

Interference finds pairs of occurrences whose solids share volume. Clearance finds pairs whose surfaces come closer than a gap, such as 0.5 mm between moving parts. Each pair names both occurrences by path (`arm_front_left/motor`), so a pick in the result selects them in the canvas.

A check after a change can be limited to the occurrences the change added or moved: `design.inspect` with the `interference` section and `changed` (occurrence names or paths, `*` globs) checks only the pairs those occurrences are in. The answer lists the overlaps the change brings and leaves out overlaps the assembly already had between other parts, such as a screw modelled at its nominal size in a tapped hole. Bodies and pair results are kept between checks, so a second check after a small fix regenerates only the parts the fix changed.

```json
{ "path": "/drone.assembly", "sections": ["interference"], "changed": ["skid_*", "tube_*"] }
```

## Study checks

A study that names a part or an assembly checks the design before it meshes, and the Desktop runs the same checks on demand:

- **Mass:** every part occurrence has a density, from its `density` property or its material, so the study can assign its material and its loads from gravity and acceleration are right. A part with no density is an error.
- **Fit:** no two occurrences share material, since an overlap meshes as one region with doubled stiffness and mass. An overlap is an error.
- **Contacts:** pairs closer than the contact gap (0.1 mm by default) that don't touch are warnings, since a bonded or contact pair that never meets carries no load until the gap closes. Pairs that touch, with a gap of 1 µm or less, are listed as contacts for information.

The summary reads `ready to mesh` with the count of warnings, or the count of errors to fix first.

## Cut lists

A weldment's cut list groups identical structural members: item, quantity, profile, material, cut length, start and end angles, and mass each. The same list can be placed on a drawing as a `cut_list` table; see [Drawing views and annotations](/docs/mcad/drawing/views-and-annotations#tables).

## Design for manufacture

**DFM check** (`evaluate.dfm`) checks the part for a manufacturing process. The part's manufacturing features are measured first: its bodies (size, thinnest and thickest wall, faces short of draft, undercut faces, overhangs, enclosed voids), holes (diameter, depth over diameter, edge distance, thread depth), pockets (depth over width, sharp internal corners), fillets, sheets (thickness, bend radius over thickness, K-factor), flanges and ribs. The rule pack for the process then checks them:

| Process | Rules |
|---|---|
| Machining | blind hole depth ≤ 10 diameters, hole diameter ≥ 1 mm, thread depth ≤ 3 diameters, hole edge distance ≥ 0.5 diameters, no sharp internal pocket corners, pocket depth ≤ 4 widths, wall ≥ 0.8 mm |
| Sheet metal | inside bend radius ≥ 1 thickness, K-factor 0.25 to 0.5, flange length ≥ 4 thicknesses, punched hole diameter ≥ 1 thickness, hole edge distance ≥ 2 thicknesses |
| Molding | no face short of 0.5° draft, no undercut face, wall 0.8 to 4 mm, thickest wall ≤ 2 × thinnest, rib thickness ≤ 0.6 × wall, rib height ≤ 3 × wall, fillet radius ≥ 0.5 mm |
| Additive | wall ≥ 0.8 mm, hole diameter ≥ 2 mm, no overhang steeper than 45° from the build direction, no enclosed void, size within 250 × 250 × 300 mm |

Molding's pull and additive's build direction are the Top plane's normal unless you pick a plane, axis or face. Each rule is a row for each item it applies to, with the rule, the item, the limit with its unit, the measured value and the result. Failures come first: a failed rule that blocks a release is an error, one that blocks publishing a warning, and the rest information. A rule whose measurement the part doesn't have reads `Not measured`. Selecting a row selects the feature it is about.

A company pack tightens or extends the rules. Name a JSON file in the project, such as `company/dfm` for `company/dfm.json`, beside the part or at the project's root, and its checks run as well:

```json
{
  "name": "acme.machining",
  "version": "1",
  "checks": [
    { "name": "thick_wall", "select": "part", "attribute": "min_wall", "op": "ge",
      "expected": { "is": "quantity", "text": "1.5 mm" }, "gate": "release" }
  ]
}
```

`select` picks items by kind (`part`, `hole`, `hole.blind`, `pocket`, `sheet`, `flange`, `rib`, `fillet`), `attribute` names the measurement, `op` is one of `lt`, `le`, `gt`, `ge`, `eq` and `ne`, and `gate` is `release`, `publish` or `none`. A failed check is also a `cad.dfm.check_failed` diagnostic on the feature.

## Cost estimate

**Cost estimate** (`evaluate.cost`) works out the part's cost for a process and a quantity from rates you enter. It uses no built-in prices: every number is an input or is measured from the part, and the results header lists every input.

| Input | Unit | Used by |
|---|---|---|
| Quantity | parts in the batch | every process |
| Material price | per kg | every process |
| Machine rate | per hour | every process |
| Setup | per batch | every process |
| Removal rate | cm³/min | machining |
| Stock allowance | mm per side | machining |
| Build rate | cm³/h | additive |
| Cycle time | s per part | molding, sheet metal |
| Tooling (a mold, a die) | per batch | molding, sheet metal |

The estimate per part is the sum of these lines, each a row that shows its working:

- **Material:** the mass bought times the price per kg. Machining buys the stock, the part's box with the allowance on each side at the part's density; the other processes buy the part's own mass. The part needs a material so its density is known.
- **Machine:** the machine time times the machine rate. Machining removes the stock that isn't part at the removal rate, additive builds the part's volume at the build rate, and molding and sheet metal take the cycle time.
- **Setup** and **tooling:** each divided over the quantity.

For example, a 100 × 10 × 10 mm aluminum bar (27 g) machined from its own box, at 10 per kg, 60 per hour and 50 per batch of 10, costs 0.27 for material, nothing for machining and 5.00 for setup: 5.27 per part. A rate the process needs and that is left empty is named in the message.

## Thickness

Thickness measures the wall thickness at every point of every face's mesh: the distance from the point into the body, along the inward surface normal, to the next face. The canvas colors the part by it, from red at the thinnest to blue at the thickest, with the range in the legend. Where the display mesh is coarse, such as a flat face drawn as two triangles, its triangles are split for the analysis so the colors follow the measured values, and each vertex is placed on the exact surface.

Set a target thickness and choose which regions to show:

- **Thin regions** flags each face with a point thinner than the target, such as a 1.5 mm wall in a part designed to 2 mm.
- **Thick regions** flags each face with a point thicker than the target, where a molded part would sink or need a core.

Each face is a row with its thinnest and thickest point, in mm. On a 100 × 20 × 10 mm block, the top and bottom faces read 10 mm through, and the ends 100 mm.

## Curvature

Curvature colors every face by its curvature at each point of its mesh, from the exact surface's principal curvatures. Convex regions are red, concave regions blue and flat faces neutral, on a scale that is symmetric about zero. Choose what to show:

| Show | Value | Unit |
|---|---|---|
| Largest curvature | the principal curvature of larger size, with its sign | 1/mm |
| Mean curvature | the average of the two principal curvatures | 1/mm |
| Gaussian curvature | the product of the two principal curvatures | 1/mm² |

Each face is a row with its least and greatest value and its smallest radius, the inverse of its largest principal curvature, or `Flat`. The summary names the smallest radius on the part and the face it is on. A 5 mm round bar's side reads 0.2 1/mm (convex) and a 5 mm hole's wall −0.2 1/mm (concave).

## Zebra stripes

Zebra stripes shows black and white bands, banded by the angle between each point's exact surface normal and a light direction: a plane, axis or face you pick, or the isometric direction (1, 1, 1), which is square to no origin plane, so a block's flat faces take different bands and a hole's wall shows stripes. Where two faces meet tangentially, the bands run on across the edge; where they meet at a crease, the bands break. Set the number of stripes over the half turn from facing the light to facing away; 20 is the default. The faces are split for the analysis until a triangle's normals turn by no more than a third of a band, so the bands keep their shape on curved faces.

Each edge between two faces is a row with its continuity, measured by the kernel at the edge's midpoint:

| Continuity | Meaning |
|---|---|
| G0 | the faces meet, at an angle (a crease) |
| G1 | the faces are tangent |
| G2 | the faces are tangent and their curvature matches |

Creases come first. Selecting a row selects the edge in the canvas.

## Deviation

Deviation measures the angle between the two faces along edges: at each point of the edge, the angle between the faces' exact surface normals. A tangent edge, such as one along a fillet, measures 0°; an edge of a box measures 90°. Pick edges, or leave the pick empty to measure every edge between two faces. Each edge is a row with its least, greatest and mean angle, sharpest first, and edges whose greatest angle is over the tangent tolerance (0.5° by default) are marked. Use it to find an edge meant to be tangent that isn't, such as a blend that leaves a 2° step.

All four commands run on the part's active configuration. The agent runs them with `design.propose_edit` and the commands `evaluate.thickness`, `evaluate.curvature`, `evaluate.zebra` and `evaluate.deviation`; the result's `data` lists each face's or edge's values: thicknesses and radii in metres, curvatures in the unit shown, and angles in degrees. The field's values per vertex are in the unit the legend shows.
