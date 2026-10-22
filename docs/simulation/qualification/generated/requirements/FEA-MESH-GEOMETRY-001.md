# FEA-MESH-GEOMETRY-001: Exact CAD import of real parts and assemblies

0 PASS, 4 PARTIAL, 0 FAIL, 14 OPEN, 0 TAGGED. Generated; do not edit.

## exact_formats

Evidence kind: IndependentReference. Criterion: Exact STEP, IGES and BREP import keeps authoritative B-rep topology and mass properties, or fails typed; tessellated formats (STL, OBJ, PLY, glTF) are rejected for solver meshing with a typed diagnostic. A governed corpus of at least the verification/meshing/corpus.json fixtures passes with recorded hashes, topology counts and mass properties.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-io[test:external_cad_corpus]::public_industrial_step_imports_exact_topology_and_mass_properties` (crates/runmat-geometry/io/tests/external_cad_corpus.rs:132) - A public NIST ACIS STEP part imports with one body, solid, shell, six faces, twelve edges and eight vertices and kernel mass-property volume equal to the refere... |
| shared_infrastructure | mesh_independent | native_process | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## multibody_assembly_import

Evidence kind: SemanticRealization. Criterion: Assemblies with touching, imprinted or nested bodies import as named parts with explicit interfaces and without failing on shared faces: a gear, an inductor and the surrounding air; a core, primary and air; a plate and its coolant volume. Each part keeps its own solid, shell and face identities and the assembly keeps the interface lineage the study selects against.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-io[test:shared_interfaces]::separate_parts_stay_independent_unless_interfaces_are_shared` (crates/runmat-geometry/io/tests/shared_interfaces.rs:72) - A part-in-air assembly imports as two named solids; with shared interfaces it yields six interface faces, keeps the names Part, Air, Outer boundary and the part...<br>`runmat-geometry-io[test:shared_interfaces]::a_general_fuse_compound_imports_with_its_shared_faces_and_solid_names` (crates/runmat-geometry/io/tests/shared_interfaces.rs:136) - A General Fuse compound imports with its shared faces and solid names. |
| shared_infrastructure | mesh_independent | native_process | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## authored_names_survive_to_run

Evidence kind: SemanticRealization. Criterion: Face, edge, solid and assembly names authored in the STEP file (for example Beam, Fixed end, Loaded end, Heat sink, Mounting face, Coil terminal in, Coil terminal out, Outer boundary, Inlet, Outlet, Walls) are read from the file and resolve with `select name` at check and at run to the same entities. Unnamed or ambiguous selections fail with a located diagnostic; no run silently selects zero or th...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-io[test:shared_interfaces]::authored_face_and_body_names_reach_the_exact_model` (crates/runmat-geometry/io/tests/shared_interfaces.rs:54) - Authored face names (Fixed end, Loaded end) and a solid name (Bar) reach the exact model and are stable under file rename and reimport; resolution at check and ...<br>`runmat-geometry-io[test:step_face_names]::authored_step_face_names_become_face_region_names` (crates/runmat-geometry/io/tests/step_face_names.rs:16) - Authored STEP face names become face region names in the imported model. |
| shared_infrastructure | mesh_independent | native_process | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## selectors_on_constructed_and_cell_regions

Evidence kind: SemanticRealization. Criterion: `select id` on constructed geometry, `select result` and `select name` for fluid `cell` regions on CAD, and every other selector form the language admits resolve identically at check and at run; a selector that check accepts cannot fail at run for lack of an entity.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | native_process | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## curved_import_validation_budget

Evidence kind: ResourceAdmission. Criterion: Exact import validation of curved parts (coaxial pipes with chamber and baffle, a 1.5 m hemisphere, spherical sinuses) completes within a stated allocation and time budget and either passes or fails with a typed diagnostic naming the entity; `EXACT_VALIDATION_BUDGET_EXCEEDED` on a valid part is a defect, and the budget accounts allocation consumption in every validation stage.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | native_process | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## geometry_load_units_option

Evidence kind: RejectionInvariant. Criterion: The `geometry.load` units option and the `units` declaration of a geometry block scale imported coordinates consistently and reject mixed or missing units with a typed diagnostic before meshing.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-io[test:external_cad_corpus]::declared_step_units_preserve_physical_scale_across_requested_coordinate_units` (crates/runmat-geometry/io/tests/external_cad_corpus.rs:58) - Declared STEP units preserve physical scale across requested coordinate units; mixed or missing unit rejection is not asserted here. |
| shared_infrastructure | mesh_independent | native_process | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

