# FEA-MESH-GEOMETRY-001: Exact CAD import of real parts and assemblies

0 PASS, 0 FAIL, 18 OPEN, 0 TAGGED. Generated; do not edit.

## exact_formats

Evidence kind: IndependentReference. Criterion: Exact STEP, IGES and BREP import keeps authoritative B-rep topology and mass properties, or fails typed; tessellated formats (STL, OBJ, PLY, glTF) are rejected for solver meshing with a typed diagnostic. A governed corpus of at least the verification/meshing/corpus.json fixtures passes with recorded hashes, topology counts and mass properties.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | native_process | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## multibody_assembly_import

Evidence kind: SemanticRealization. Criterion: Assemblies with touching, imprinted or nested bodies import as named parts with explicit interfaces and without failing on shared faces: a gear, an inductor and the surrounding air; a core, primary and air; a plate and its coolant volume. Each part keeps its own solid, shell and face identities and the assembly keeps the interface lineage the study selects against.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | native_process | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

## authored_names_survive_to_run

Evidence kind: SemanticRealization. Criterion: Face, edge, solid and assembly names authored in the STEP file (for example Beam, Fixed end, Loaded end, Heat sink, Mounting face, Coil terminal in, Coil terminal out, Outer boundary, Inlet, Outlet, Walls) are read from the file and resolve with `select name` at check and at run to the same entities. Unnamed or ambiguous selections fail with a located diagnostic; no run silently selects zero or th...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
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
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | mesh_independent | native_process | OPEN |  |
| shared_infrastructure | mesh_independent | public_study | OPEN |  |

