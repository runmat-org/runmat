# FEA-MESH-GEOMETRY-001: Exact CAD import of real parts and assemblies

0 PASS, 4 PARTIAL, 0 FAIL, 14 OPEN, 0 TAGGED. Generated; do not edit.

## exact_formats

Evidence kind: IndependentReference. Criterion: Exact STEP, IGES and BREP import keeps authoritative B-rep topology and mass properties, or fails typed; tessellated formats (STL, OBJ, PLY, glTF) are rejected for solver meshing with a typed diagnostic. A governed corpus of at least the verification/meshing/corpus.json fixtures passes with recorded hashes, topology counts and mass properties.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-io[test:external_cad_corpus]::public_industrial_step_imports_exact_topology_and_mass_properties` (crates/runmat-geometry/io/tests/external_cad_corpus.rs:381) - A public NIST ACIS STEP part imports with one body, solid, shell, six faces, twelve edges and eight vertices and kernel mass-property volume equal to the refere... |
| shared_infrastructure | mesh_independent | native_process | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## multibody_assembly_import

Evidence kind: SemanticRealization. Criterion: Assemblies with touching, imprinted or nested bodies import as named parts with explicit interfaces and without failing on shared faces: a gear, an inductor and the surrounding air; a core, primary and air; a plate and its coolant volume. Each part keeps its own solid, shell and face identities and the assembly keeps the interface lineage the study selects against.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-io[test:shared_interfaces]::separate_parts_stay_independent_unless_interfaces_are_shared` (crates/runmat-geometry/io/tests/shared_interfaces.rs:92) - A part-in-air assembly imports as two named solids; with shared interfaces it yields six interface faces, keeps the names Part, Air, Outer boundary and the part...<br>`runmat-geometry-io[test:shared_interfaces]::a_general_fuse_compound_imports_with_its_shared_faces_and_solid_names` (crates/runmat-geometry/io/tests/shared_interfaces.rs:183) - A General Fuse compound imports with its shared faces and solid names.<br>`runmat-geometry-io[test:shared_interfaces]::imprinted_coincident_faces_become_one_shared_face_with_one_source_ordinal` (crates/runmat-geometry/io/tests/shared_interfaces.rs:250) - Imprinted coincident faces of an assembly become one shared face with one source ordinal. |
| shared_infrastructure | mesh_independent | native_process | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## authored_names_survive_to_run

Evidence kind: SemanticRealization. Criterion: Face, edge, solid and assembly names authored in the STEP file (for example Beam, Fixed end, Loaded end, Heat sink, Mounting face, Coil terminal in, Coil terminal out, Outer boundary, Inlet, Outlet, Walls) are read from the file and resolve with `select name` at check and at run to the same entities. Unnamed or ambiguous selections fail with a located diagnostic; no run silently selects zero or th...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-io[test:shared_interfaces]::authored_face_and_body_names_reach_the_exact_model` (crates/runmat-geometry/io/tests/shared_interfaces.rs:65) - Authored face names (Fixed end, Loaded end) and a solid name (Bar) reach the exact model and are stable under file rename and reimport; resolution at check and ...<br>`runmat-geometry-io[test:step_edge_vertex_names]::authored_step_edge_and_vertex_names_select_their_entities` (crates/runmat-geometry/io/tests/step_edge_vertex_names.rs:93) - Authored STEP edge and vertex names select exactly the authored edges and vertices by persistent identity; face and solid names are unchanged; names stay outsid...<br>`runmat-geometry-io[test:step_edge_vertex_names]::edge_and_vertex_names_survive_placement_and_the_shared_interface_join` (crates/runmat-geometry/io/tests/step_edge_vertex_names.rs:147) - Edge and vertex names of an assembly's parts survive part placement and the shared-interface join, including a named vertex inside the shared contact patch.<br>`runmat-geometry-io[test:step_edge_vertex_names]::an_unnamed_edge_or_vertex_selection_reports_the_carried_names` (crates/runmat-geometry/io/tests/step_edge_vertex_names.rs:179) - Selecting an edge or vertex name the source does not carry fails with a message that names the requested name and lists the names the source carries for that ro...<br>`runmat-geometry-io[test:step_edge_vertex_names]::documentation_frame_and_plate_edge_and_vertex_names_resolve` (crates/runmat-geometry/io/tests/step_edge_vertex_names.rs:214) - The documentation portal-frame and square-plate sheets resolve every edge and vertex name their studies select to the edge or vertex at the authored position.<br>`runmat-geometry-io[test:step_face_names]::authored_step_face_names_become_face_region_names` (crates/runmat-geometry/io/tests/step_face_names.rs:16) - Authored STEP face names become face region names in the imported model. |
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
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:geometry_features]::imported_geometry_without_units_names_the_missing_units` (crates/runmat-analysis/language/tests/geometry_features.rs:108) - A B-rep geometry without authored units fails closed at check with a located diagnostic naming the missing units.<br>`runmat-analysis-language[test:geometry_features]::imported_geometry_without_units_uses_the_file_declared_unit` (crates/runmat-analysis/language/tests/geometry_features.rs:145) - Omitted units resolve to the single unit the STEP file declares, without diagnostics.<br>`runmat-analysis-language[test:geometry_features]::select_name_resolves_only_names_authored_in_the_cad_file` (crates/runmat-analysis/language/tests/geometry_features.rs:192) - Authored `units = m` over a STEP file declaring millimetres resolves to metres and reports a located units-override warning; the check stays valid.<br>`runmat[test:check_run_parity]::only_names_authored_in_the_cad_file_resolve` (crates/runmat-cli/tests/check_run_parity.rs:249) - `select name` resolves only names authored in the STEP file; a selections file under .runmat/geometry is never read.<br>`runmat-geometry-io[test:external_cad_corpus]::declared_step_units_preserve_physical_scale_across_requested_coordinate_units` (crates/runmat-geometry/io/tests/external_cad_corpus.rs:58) - Declared STEP units preserve physical scale across requested coordinate units; mixed or missing unit rejection is not asserted here.<br>`runmat-geometry-io[test:external_cad_corpus]::authored_step_units_override_a_different_declaration_and_record_it` (crates/runmat-geometry/io/tests/external_cad_corpus.rs:124) - An authored metre unit over a STEP file declaring millimetres reads the coordinates in metres, records the declared unit as provenance in the geometry document,...<br>`runmat-geometry-io[test:external_cad_corpus]::authored_iges_units_override_a_different_declaration_and_record_it` (crates/runmat-geometry/io/tests/external_cad_corpus.rs:179) - An authored metre unit over an IGES file whose global section declares millimetres reads the coordinates in metres and records the declared unit; without author...<br>`runmat-geometry-io[test:external_cad_corpus]::centimetre_and_foot_step_declarations_resolve_and_accept_authored_overrides` (crates/runmat-geometry/io/tests/external_cad_corpus.rs:259) - STEP files declaring centimetres (SI prefix) or feet (conversion-based unit with a 304.8 mm factor) import at their declared scale, and an authored metre unit o...<br>`runmat-geometry-io[test:external_cad_corpus]::authored_units_refuse_step_files_without_one_declared_unit` (crates/runmat-geometry/io/tests/external_cad_corpus.rs:319) - Authored units over a STEP file whose global contexts declare several different length units, or no length unit, are refused with GEOMETRY_UNITS_UNRESOLVED inst... |
| shared_infrastructure | mesh_independent | native_process | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

