---
description: Evaluate a RunMat part or assembly. Mass, center of gravity and inertia from materials, interference and clearance between parts, weldment cut lists, and design-for-manufacture checks for machining, sheet metal, molding and additive.
keywords: [mass properties, center of gravity, moment of inertia, interference detection, clearance check, cut list, weldment, DFM, design for manufacturing, wall thickness, draft analysis]
---
# Evaluate

Evaluate measures a part or an assembly in its active configuration. The Desktop lists each result under the canvas; the agent reads the same results through `design.inspect`, and checks can gate on them.

| Command | Works on | Result |
|---|---|---|
| `evaluate.mass-properties` | a part or an assembly | mass, center of gravity and inertia about it |
| `evaluate.interference` | an assembly | pairs of occurrences whose solids overlap, with the overlap volume |
| `evaluate.clearance` | an assembly | pairs closer than a gap you set |
| `weldment.cut-list` | a weldment part | its structural members, grouped |

## Mass properties

A part weighs its output bodies at their density: the output's `density` property when it has one, else its material's density from a table of common alloys, plastics and composites matched by the words in the material's name (`material.aluminum_7075`, `steel_300m`, `carbon_prepreg`, `nylon`, `fr4`). An assembly adds every placed part occurrence at its world placement: its center of gravity is mass-weighted, and its inertia is taken about that center by the parallel-axis theorem,

$$
I_{cg} = \sum_i \left( I_i + m_i \left( |d_i|^2 E - d_i d_i^{T} \right) \right),
$$

where $m_i$ and $I_i$ are each occurrence's mass and inertia about its own center, $d_i$ is the offset of that center from the assembly's, and $E$ is the identity. The result lists parts with no density, so a missing material never hides as a light assembly.

## Interference and clearance

Interference finds pairs of occurrences whose solids share volume. Clearance finds pairs whose surfaces come closer than a gap, such as 0.5 mm between moving parts. Each pair names both occurrences by path (`arm_front_left/motor`), so a pick in the result selects them in the canvas.

## Cut lists

A weldment's cut list groups identical structural members: item, quantity, profile, material, cut length, start and end angles, and mass each. The same list can be placed on a drawing as a `cut_list` table; see [Drawing views and annotations](/docs/mcad/drawing/views-and-annotations#tables).

## Design for manufacture

A part's manufacturing features (bodies, holes, pockets, fillets, sheets, flanges and ribs, each with measured attributes) are checked by rule packs for machining, sheet metal, molding and additive: hole depth over diameter, edge distance, sharp internal corners in pockets, bend radius over thickness, wall thickness, draft and undercuts. A company pack can tighten or extend the rules. A failed check is a `cad.dfm.check_failed` diagnostic on the feature.

## Surface and shape analysis

Section views through the part, wall-thickness maps, draft and undercut maps for a pull direction, curvature and zebra stripes, and deviation from a reference body [not yet in the current release](pending:design.mcad.evaluate.surface-analysis).
