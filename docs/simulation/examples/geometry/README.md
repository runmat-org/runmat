# Geometry fixtures

`two-box-assembly.step` is a repository-generated Open CASCADE STEP assembly with two occurrences
of one exact box definition. The assembly example exercises assembly-aware exact
import and material coverage for more than one solid.

`cantilever.step` is an Open CASCADE 7.8 STEP export of an exact box constructed with
`BRepPrimAPI_MakeBox(1000.0, 100.0, 100.0)` in millimeters. The linear-static guide
imports it with `units = mm`, producing a 1 m beam with a 100 mm square section.
The included model fixes `face_000001` at x = 0 and loads `face_000002` at x = 1 m.

`box.brep` is a repository-generated exact test solid used by the modal, transient, and
random-response examples. `tetrahedron.stl` is the faceted geometry used by meshing examples.
