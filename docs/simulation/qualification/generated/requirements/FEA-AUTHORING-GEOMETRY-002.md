# FEA-AUTHORING-GEOMETRY-002: Exact geometry, persistent topology and engineering exchange

1 PASS, 0 FAIL, 46 OPEN, 0 TAGGED. Generated; do not edit.

## geometry_source_forms

Evidence kind: SemanticRealization. Criterion: A geometry declaration has exactly one closed source form: imported file, declarative construction, immutable derivation, or materialized RunMat generator. All forms produce the same typed canonical geometry asset and exact-revision boundary consumed by region resolution and meshing.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## imported_geometry

Evidence kind: SemanticRealization. Criterion: The file form admits one inspectable exact-geometry source: STEP is the primary documentation and production example format. Other admitted exact-CAD and canonical geometry sources remain registry-governed rather than accepted as arbitrary extensions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## bounded_feature_graph

Evidence kind: SemanticRealization. Criterion: The construction form is a typed feature graph, not an imperative CAD macro: The initial production feature set should remain deliberately bounded: dimensioned primitives, rigid transforms and instances, extrude/revolve, governed booleans, holes/pockets, and qualified fillet/chamfer operations. Fully constrained sketching, mating, advanced surface construction, healing, and arbitrary kernel comman...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## immutable_derivation

Evidence kind: SemanticRealization. Criterion: Geometry is never changed in place. A derived declaration creates a new revision with explicit lineage: Source edits may feel like CAD mutation, but semantic evaluation is an immutable dependency graph. The base geometry remains addressable and cacheable. The derived revision records the source revision, operation graph, inputs, kernel capability cohort, persistent-entity history, validation, and ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## generated_geometry

Evidence kind: SemanticRealization. Criterion: General procedural construction remains in `.m` and materializes through the same bounded, source-resolvable generator boundary as signals: The imported function must return exactly one immutable `geometry.Generated` value. It builds that value with the ordinary typed RunMat geometry API (`geometry.box`, `geometry.cylinder`, `geometry.sphere`, transforms/instances, Boolean operations, `geometry.fi...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## persistent_topology

Evidence kind: SemanticRealization. Criterion: Feature-derived selection must solve the topological naming problem explicitly. Stable feature identity, complete provider-proven source-to-result lineage, semantic feature roles, persistent entity digests, cardinality, and ambiguity diagnostics are part of geometry realization evidence. Face or edge vector position is never persistent identity. A feature parameter change that splits one selected ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## feature_lineage_inventory

Evidence kind: SemanticRealization. Criterion: Every feature result publishes the complete final entity inventory grouped by topology role, the entities introduced by that operation, and a canonical typed lineage record for every source entity. A lineage record maps its source to all provider-proven descendants in that operation's final result; an empty descendant set means that no source entity survived. Unchanged, modified, split, merged, ge...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## exact_geometry_closure

Evidence kind: SemanticRealization. Criterion: Imported, constructed, derived, and generated sources all end at the same immutable exact-geometry closure. Constructed graphs have no hidden source. Derived graphs receive exactly their declared source revision. Generated calls receive already-resolved typed arguments and execute once as a bounded materialization task; downstream feature execution and meshing consume the published closure rather ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## geometry_owned_regions

Evidence kind: SemanticRealization. Criterion: A region is a typed, geometry-owned selection, not a free-standing FEA label. Import first creates an exact geometry asset with an identity, revision, persistent entities, and imported semantic metadata. Each nested region selector is then evaluated against that exact asset. The resolved region retains the geometry identity/revision, selector and selection digest, matched persistent entity ids, to...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## mesh_source_alternatives

Evidence kind: SemanticRealization. Criterion: A mesh declaration either references an immutable canonical mesh artifact or describes a meshing request against exact geometry. These are closed alternatives. Quality and partition controls are optional and validated only when present.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## inspected_selector_semantics

Evidence kind: SemanticRealization. Criterion: The initial selector surface maps directly to imported geometry metadata and persistent ids: `name` and `tag` refer to metadata actually imported from the source—such as STEP/XCAF product, body, component, assembly, layer, color, or material ownership—or to explicit annotations in a supported geometry source. `id` selects the role-scoped source alias reported by geometry inspection, such as `face_...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## selector_inspection

Evidence kind: SemanticRealization. Criterion: The inspection result and interactive geometry view expose selectable ids, names, tags, topology roles, and grouping provenance. If the CAD file contains no meaningful boundary grouping, RunMat may still expose exact face ids, but it cannot invent names such as `Fixed_Base`. The durable choices are to add semantic names/layers/groups in CAD, author a separately persisted geometry annotation revisi...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## region_role_rejection

Evidence kind: SemanticRealization. Criterion: Consumers cannot reinterpret a declared role. A face cannot become a solid material domain merely because an assignment asks for one. A selector that matches the wrong kind, no entities, or an unexpected cardinality fails during checking. Editing/reimporting CAD creates a new geometry revision; a stale resolved region or incompatible mesh cannot be reused silently.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## ap242_exchange

Evidence kind: SemanticRealization. Criterion: RunMat should treat STEP AP242 as more than a source of boundary representation when the exchange file actually contains richer product data. ISO 10303-242:2025 includes parametric and constrained geometry, sketches, manufacturing features, part properties, composite definitions, and material specifications/properties in its scope. That scope does not guarantee that a particular CAD exporter wrote...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## cad_parameter_admission

Evidence kind: SemanticRealization. Criterion: Imported parameters have stable source identity, declared physical type, units, value, bounds or constraint facts where present, ownership path, edition/profile, and read/write capability: Textual names and external identifiers are selectors at the file boundary; successful resolution produces a strongly typed parameter reference. Duplicate names, missing units, incompatible dimensions, ambiguous ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PASS | `runmat-analysis-language[test:ap242_exchange]::imported_cad_parameters_reject_wrong_dimensions_and_unqualified_writes` (crates/runmat-analysis/language/tests/ap242_exchange.rs:74) - Fixture host parameter metadata: wrong dimensions/unqualified write rejection only; no CAD kernel regeneration. |

## qualified_parameter_regeneration

Evidence kind: SemanticRealization. Criterion: Changing an imported parameter is an immutable derivation, not in-place mutation: This checks only when admission proves that the selected parameter is writable and that RunMat has a qualified evaluator for the complete dependent parametric/constraint representation. A flattened B-rep with a nominal dimension annotation remains readable metadata; it is not promoted to a writable parameter. Unsuppo...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## explicit_material_adoption

Evidence kind: SemanticRealization. Criterion: AP242 material and product properties are imported as typed, provenance-bearing candidates. They do not silently become FEA constitutive laws, because a CAD material may be only a name, a visual material, a density, a specification reference, a composite constituent, or a property qualified by temperature or another operating environment. Adoption is explicit:

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## physical_material_graph

Evidence kind: SemanticRealization. Criterion: The primary physical-material graph follows the CAx-IF recommended practice: stock material is a `PRODUCT` in a recognized material `PRODUCT_RELATED_PRODUCT_CATEGORY`; material properties point to its `PRODUCT_DEFINITION`; a part-to-material callout is a `MAKE_FROM_USAGE_OPTION`; applicable specifications are `APPLIED_DOCUMENT_REFERENCE` relationships; and each dependent environment is resolved th...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## adopted_property_provenance

Evidence kind: SemanticRealization. Criterion: Every adopted property retains its AP242 entity/persistent id, source association, units, specification/document reference, environment, uncertainty/validation facts where available, and conversion evidence. Missing required capabilities reject before analysis. Visual materials are never accepted as physical materials. Geometry-to-material associations may create selectable material domains, but t...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## truthful_exchange_export

Evidence kind: SemanticRealization. Criterion: For CAD round trips, RunMat retains the immutable admitted source and emits a typed change set plus an export-conformance report. A qualified AP242 writer may update supported parameters, properties, associations, persistent ids, and validation properties. Unsupported or lossy content must be reported; RunMat must not claim a lossless round trip merely because it can write another STEP file. Sidec...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## typed_property_assignments

Evidence kind: SemanticRealization. Criterion: Material, section, layup, orientation, and other physical-property assignments are distinct typed forms. Bulk materials normally target solids or cells; shells and beams additionally require their relevant section/layup/orientation contracts. Boundary faces do not own a solid's bulk material. Beam-section, shell-layup, and orientation assignments use the same geometry-qualified shape but require r...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## section_layup_orientation_contract

Evidence kind: SemanticRealization. Criterion: Sections initially admit general, rectangular, circular, and tubular beams plus homogeneous shells. Layups require an ordered, uniquely named ply sequence; every ply names its material, thickness, and angle. Ply order is `along_surface_normal` or `opposite_surface_normal`. A layup reference surface is its mid-surface, first-ply surface, last-ply surface, or an explicit length offset from the mid-s...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## complete_property_coverage

Evidence kind: SemanticRealization. Criterion: Every active material-bearing entity must have exactly one compatible owner. Gaps, overlaps, wrong topology roles, stale geometry revisions, and missing material capabilities fail before assembly.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

