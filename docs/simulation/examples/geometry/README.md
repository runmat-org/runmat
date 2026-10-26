# Geometry fixtures

`two-box-assembly.step` is a repository-generated Open CASCADE STEP assembly with two occurrences
of one exact box definition. The assembly example exercises assembly-aware exact
import and material coverage for more than one solid.

`cantilever.step` is an Open CASCADE 7.8 STEP export of an exact box constructed with
`BRepPrimAPI_MakeBox(1000.0, 100.0, 100.0)` in millimeters. The linear-static guide
imports it with `units = mm`, producing a 1 m beam with a 100 mm square section.
The file names the part `Beam` and its end faces `Fixed end` (x = 0) and `Loaded end` (x = 1 m).

`bar.step` and `duct.step` are the same box with names for the thermal-bar and acoustic-duct
examples: `bar.step` names the part `Bar`, the end faces `End A` (x = 0) and `End B` (x = 1 m),
and the side faces `Side A` (y = 0), `Side B` (y = 100 mm), `Side C` (z = 0) and `Side D`
(z = 100 mm); `duct.step` names the part `Duct` and the end faces `Near end` (x = 0) and
`Far end` (x = 1 m). They were written from `cantilever.step` with `name-step-faces.cc`.

`cavity-slab.step`, `plane-channel.step` and `pipe.step` come from `generate-flow-fixtures.cc`,
which names the part and every face (`Lid`, `Inlet`, `Outlet`, `Lower wall`, `Wall`, ...).
`portal-frame.step` names its members (`Girder`, `Left column`, `Right column`, `Ground`) and
joints (`Left base`, `Right base`, `Left top`, `Right top`); `square-plate.step` names its
edges `South edge`, `East edge`, `North edge` and `West edge`. Both were named with
`name-step-faces.cc`, which also names edges and vertices.

`box.brep` is a repository-generated exact test solid used by the modal, transient, and
random-response examples. `tetrahedron.stl` is the faceted geometry used by meshing examples.
