# FEA-AUTHORING-GEOMETRY-002: Exact geometry, persistent topology and engineering exchange

0 PASS, 21 PARTIAL, 0 FAIL, 26 OPEN, 0 TAGGED. Generated; do not edit.

## geometry_source_forms

Evidence kind: SemanticRealization. Criterion: A geometry declaration has exactly one closed source form: imported file, declarative construction, immutable derivation, or materialized RunMat generator. All forms produce the same typed canonical geometry asset and exact-revision boundary consumed by region resolution and meshing.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-ops[lib]::feature::tests::derived_geometry_requires_the_exact_declared_source` (crates/runmat-geometry/ops/src/feature/tests.rs:435) - A derived geometry requires the exact declared source revision; the closed-form check that imported, constructed and generated forms are mutually exclusive is n... |

## imported_geometry

Evidence kind: SemanticRealization. Criterion: The file form admits one inspectable exact-geometry source: STEP is the primary documentation and production example format. Other admitted exact-CAD and canonical geometry sources remain registry-governed rather than accepted as arbitrary extensions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-core[lib]::model::authoring_tests::imported_geometry_requires_exact_identity_and_persistent_topology` (crates/runmat-geometry/core/src/model/authoring_tests.rs:15) - Imported geometry requires exact identity and persistent topology at the model level; file inspection and unit handling are covered in the language tests. |

## bounded_feature_graph

Evidence kind: SemanticRealization. Criterion: The construction form is a typed feature graph, not an imperative CAD macro: The initial production feature set should remain deliberately bounded: dimensioned primitives, rigid transforms and instances, extrude/revolve, governed booleans, holes/pockets, and qualified fillet/chamfer operations. Fully constrained sketching, mating, advanced surface construction, healing, and arbitrary kernel comman...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-ops[lib]::feature::tests::unsupported_operations_and_provider_bounds_fail_before_execution` (crates/runmat-geometry/ops/src/feature/tests.rs:164) - Unsupported operations and provider bounds fail before execution; the full bounded operation set is not enumerated. |
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
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:programs]::generated_geometry_freezes_function_provenance_and_typed_arguments` (crates/runmat-analysis/language/tests/programs.rs:666) - Generated geometry freezes imported-function provenance and typed arguments at check.<br>`runmat-geometry-ops[lib]::feature::tests::generated_geometry_receives_resolved_arguments` (crates/runmat-geometry/ops/src/feature/tests.rs:490) - Generated geometry receives resolved arguments at the provider boundary; materialization of the function result is covered by the OCCT tests. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## persistent_topology

Evidence kind: SemanticRealization. Criterion: Feature-derived selection must solve the topological naming problem explicitly. Stable feature identity, complete provider-proven source-to-result lineage, semantic feature roles, persistent entity digests, cardinality, and ambiguity diagnostics are part of geometry realization evidence. Face or edge vector position is never persistent identity. A feature parameter change that splits one selected ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-io[lib]::occt::evaluator::tests::persistent_entity_names_reject_ambiguous_coincident_topology` (crates/runmat-geometry/io/src/occt/evaluator/tests.rs:574) - Persistent entity names reject ambiguous coincident topology on the OCCT evaluator; feature-derived naming across booleans is covered by the feature tests. |

## feature_lineage_inventory

Evidence kind: SemanticRealization. Criterion: Every feature result publishes the complete final entity inventory grouped by topology role, the entities introduced by that operation, and a canonical typed lineage record for every source entity. A lineage record maps its source to all provider-proven descendants in that operation's final result; an empty descendant set means that no source entity survived. Unchanged, modified, split, merged, ge...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-io[lib]::occt::feature_tests::boolean_history_projects_authored_regions_to_the_final_exact_topology` (crates/runmat-geometry/io/src/occt/feature_tests.rs:223) - Boolean history projects authored regions onto the final exact topology.<br>`runmat-geometry-io[lib]::occt::feature_tests::feature_lineage_merges_shared_ancestors_across_boolean_branches` (crates/runmat-geometry/io/src/occt/feature_tests.rs:299) - Feature lineage merges shared ancestors across boolean branches; the full per-operation entity inventory and canonical order are not asserted. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## exact_geometry_closure

Evidence kind: SemanticRealization. Criterion: Imported, constructed, derived, and generated sources all end at the same immutable exact-geometry closure. Constructed graphs have no hidden source. Derived graphs receive exactly their declared source revision. Generated calls receive already-resolved typed arguments and execute once as a bounded materialization task; downstream feature execution and meshing consume the published closure rather ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-io[lib]::occt::feature_tests::constructed_materialization_and_result_regions_share_the_exact_artifact` (crates/runmat-geometry/io/src/occt/feature_tests.rs:173) - Constructed materialization and result regions share one exact artifact; derived and generated closure are covered by sibling tests. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## geometry_owned_regions

Evidence kind: SemanticRealization. Criterion: A region is a typed, geometry-owned selection, not a free-standing FEA label. Import first creates an exact geometry asset with an identity, revision, persistent entities, and imported semantic metadata. Each nested region selector is then evaluated against that exact asset. The resolved region retains the geometry identity/revision, selector and selection digest, matched persistent entity ids, to...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-core[lib]::model::authoring_tests::imported_geometry_requires_exact_identity_and_persistent_topology` (crates/runmat-geometry/core/src/model/authoring_tests.rs:15) - Regions belong to an exact geometry asset with identity and revision; role checks on consumption are other tests. |

## mesh_source_alternatives

Evidence kind: SemanticRealization. Criterion: A mesh declaration either references an immutable canonical mesh artifact or describes a meshing request against exact geometry. These are closed alternatives. Quality and partition controls are optional and validated only when present.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:programs]::mesh_request_binds_geometry_and_lowers_units_into_canonical_settings` (crates/runmat-analysis/language/tests/programs.rs:285) - A mesh request binds geometry and lowers its units into canonical settings; the reference-an-existing-mesh alternative is not asserted. |

## inspected_selector_semantics

Evidence kind: SemanticRealization. Criterion: The initial selector surface maps directly to imported geometry metadata and persistent ids: `name` and `tag` refer to metadata actually imported from the source—such as STEP/XCAF product, body, component, assembly, layer, color, or material ownership—or to explicit annotations in a supported geometry source. `id` selects the role-scoped source alias reported by geometry inspection, such as `face_...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:programs]::name_selectors_must_match_names_the_source_carries` (crates/runmat-analysis/language/tests/programs.rs:462) - Name selectors must match names the inspected source carries; tag and id selector forms are covered elsewhere.<br>`runmat-analysis-language[test:programs]::edge_and_vertex_name_selectors_must_match_names_the_source_carries` (crates/runmat-analysis/language/tests/programs.rs:506) - Edge and vertex name selectors are checked against the edge and vertex names the inspected source carries, like face and solid names. |

## selector_inspection

Evidence kind: SemanticRealization. Criterion: The inspection result and interactive geometry view expose selectable ids, names, tags, topology roles, and grouping provenance. If the CAD file contains no meaningful boundary grouping, RunMat may still expose exact face ids, but it cannot invent names such as `Fixed_Base`. The durable choices are to add semantic names/layers/groups in CAD, author a separately persisted geometry annotation revisi...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:programs]::name_selectors_must_match_names_the_source_carries` (crates/runmat-analysis/language/tests/programs.rs:462) - Selection by name resolves only against imported metadata; the interactive view exposure is not covered. |

## region_role_rejection

Evidence kind: SemanticRealization. Criterion: Consumers cannot reinterpret a declared role. A face cannot become a solid material domain merely because an assignment asks for one. A selector that matches the wrong kind, no entities, or an unexpected cardinality fails during checking. Editing/reimporting CAD creates a new geometry revision; a stale resolved region or incompatible mesh cannot be reused silently.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-core[lib]::model::authoring_tests::feature_region_selectors_must_resolve_in_the_authored_graph` (crates/runmat-geometry/core/src/model/authoring_tests.rs:117) - Feature region selectors must resolve in the authored graph; consumer reinterpretation of roles is checked in the property-assignment tests. |

## ap242_exchange

Evidence kind: SemanticRealization. Criterion: RunMat should treat STEP AP242 as more than a source of boundary representation when the exchange file actually contains richer product data. ISO 10303-242:2025 includes parametric and constrained geometry, sketches, manufacturing features, part properties, composite definitions, and material specifications/properties in its scope. That scope does not guarantee that a particular CAD exporter wrote...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:ap242_exchange]::imported_cad_parameters_resolve_exact_typed_source_facts` (crates/runmat-analysis/language/tests/ap242_exchange.rs:17) - Imported CAD parameters resolve to exact typed source facts; other AP242 content (constraints, associations) is covered by the profile tests. |

## cad_parameter_admission

Evidence kind: SemanticRealization. Criterion: Imported parameters have stable source identity, declared physical type, units, value, bounds or constraint facts where present, ownership path, edition/profile, and read/write capability: Textual names and external identifiers are selectors at the file boundary; successful resolution produces a strongly typed parameter reference. Duplicate names, missing units, incompatible dimensions, ambiguous ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:ap242_exchange]::imported_cad_parameters_reject_wrong_dimensions_and_unqualified_writes` (crates/runmat-analysis/language/tests/ap242_exchange.rs:80) - Fixture host parameter metadata: wrong dimensions/unqualified write rejection only; no CAD kernel regeneration. |

## qualified_parameter_regeneration

Evidence kind: SemanticRealization. Criterion: Changing an imported parameter is an immutable derivation, not in-place mutation: This checks only when admission proves that the selected parameter is writable and that RunMat has a qualified evaluator for the complete dependent parametric/constraint representation. A flattened B-rep with a nominal dimension annotation remains readable metadata; it is not promoted to a writable parameter. Unsuppo...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:ap242_exchange]::source_regeneration_fails_during_checking_without_qualified_write_capability` (crates/runmat-analysis/language/tests/ap242_exchange.rs:116) - Source regeneration fails at check without a qualified write capability; the qualified success path is not exercised. |

## explicit_material_adoption

Evidence kind: SemanticRealization. Criterion: AP242 material and product properties are imported as typed, provenance-bearing candidates. They do not silently become FEA constitutive laws, because a CAD material may be only a name, a visual material, a density, a specification reference, a composite constituent, or a property qualified by temperature or another operating environment. Adoption is explicit:

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:ap242_exchange]::cad_material_adoption_is_explicit_typed_and_provenance_bearing` (crates/runmat-analysis/language/tests/ap242_exchange.rs:200) - CAD material adoption is explicit, typed and provenance-bearing; mapping to every constitutive law is not covered. |

## physical_material_graph

Evidence kind: SemanticRealization. Criterion: The primary physical-material graph follows the CAx-IF recommended practice: stock material is a `PRODUCT` in a recognized material `PRODUCT_RELATED_PRODUCT_CATEGORY`; material properties point to its `PRODUCT_DEFINITION`; a part-to-material callout is a `MAKE_FROM_USAGE_OPTION`; applicable specifications are `APPLIED_DOCUMENT_REFERENCE` relationships; and each dependent environment is resolved th...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-io[lib]::ap242::profile::tests::retains_the_standard_legacy_material_designation_as_an_explicit_alternate` (crates/runmat-geometry/io/src/ap242/profile.rs:461) - The standard legacy material designation is retained as an explicit alternate to the recommended-practice graph; the primary graph walk is covered by the resolv... |

## adopted_property_provenance

Evidence kind: SemanticRealization. Criterion: Every adopted property retains its AP242 entity/persistent id, source association, units, specification/document reference, environment, uncertainty/validation facts where available, and conversion evidence. Missing required capabilities reject before analysis. Visual materials are never accepted as physical materials. Geometry-to-material associations may create selectable material domains, but t...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:ap242_exchange]::cad_material_adoption_is_explicit_typed_and_provenance_bearing` (crates/runmat-analysis/language/tests/ap242_exchange.rs:200) - Adopted properties retain provenance; entity ids, document references and uncertainty are not all asserted. |

## truthful_exchange_export

Evidence kind: SemanticRealization. Criterion: For CAD round trips, RunMat retains the immutable admitted source and emits a typed change set plus an export-conformance report. A qualified AP242 writer may update supported parameters, properties, associations, persistent ids, and validation properties. Unsupported or lossy content must be reported; RunMat must not claim a lossless round trip merely because it can write another STEP file. Sidec...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-geometry-io[lib]::ap242::change_set::tests::reports_validated_changes_as_sidecar_only_without_claiming_export` (crates/runmat-geometry/io/src/ap242/change_set.rs:126) - Validated changes are reported as sidecar-only without claiming a CAD export. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## typed_property_assignments

Evidence kind: SemanticRealization. Criterion: Material, section, layup, orientation, and other physical-property assignments are distinct typed forms. Bulk materials normally target solids or cells; shells and beams additionally require their relevant section/layup/orientation contracts. Boundary faces do not own a solid's bulk material. Beam-section, shell-layup, and orientation assignments use the same geometry-qualified shape but require r...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:models]::parameterized_model_assignments_remain_strongly_typed` (crates/runmat-analysis/language/tests/models.rs:84) - Parameterized model assignments remain strongly typed; section, layup and orientation forms are not covered. |

## section_layup_orientation_contract

Evidence kind: SemanticRealization. Criterion: Sections initially admit general, rectangular, circular, and tubular beams plus homogeneous shells. Layups require an ordered, uniquely named ply sequence; every ply names its material, thickness, and angle. Ply order is `along_surface_normal` or `opposite_surface_normal`. A layup reference surface is its mid-surface, first-ply surface, last-ply surface, or an explicit length offset from the mid-s...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## complete_property_coverage

Evidence kind: SemanticRealization. Criterion: Every active material-bearing entity must have exactly one compatible owner. Gaps, overlaps, wrong topology roles, stale geometry revisions, and missing material capabilities fail before assembly.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:models]::property_roles_and_composite_ownership_fail_closed` (crates/runmat-analysis/language/tests/models.rs:267) - Property roles and composite ownership fail closed; gaps, overlaps and stale revisions across a whole mesh are checked in the properties requirement. |

