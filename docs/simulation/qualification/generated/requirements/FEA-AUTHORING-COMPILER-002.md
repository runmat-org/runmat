# FEA-AUTHORING-COMPILER-002: Compiler passes and host ownership

0 PASS, 11 PARTIAL, 0 FAIL, 47 OPEN, 0 TAGGED. Generated; do not edit.

## crate_boundary

Evidence kind: SemanticRealization. Criterion: 1. Add one narrow `runmat-analysis-language` crate. A real dependency boundary requires it: source syntax must depend toward canonical analysis contracts, while `runmat-analysis-core` and every numerical consumer must remain independent of lexer/parser code. Putting this compiler in core, runtime, parser, or execution would reverse or confuse a domain boundary. 2. Add a separate FEA frontend insid...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## dependency_edges

Evidence kind: SemanticRealization. Criterion: A -> B means A depends on B. runmat-parser -> runmat-lexer -> runmat-types runmat-builtins -> runmat-types + runmat-value runmat-geometry-ops -> runmat-geometry-core runmat-geometry-io -> runmat-geometry-ops + runmat-geometry-core runmat-analysis-core -> runmat-types + runmat-geometry-core runmat-analysis-language -> runmat-parser + runmat-analysis-core + runmat-types + runmat-geometry-core runmat...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## typed_pipeline

Evidence kind: SemanticRealization. Criterion: .fea bytes │ ├─ 1. FEA lexing │ FeaTokenStream + trivia + byte spans │ ├─ 2. Error-tolerant parsing │ FeaSyntaxModule + syntax diagnostics │ ├─ 3. Module graph and symbol binding │ BoundFeaModuleGraph + explicit import provenance │ ├─ 4. Type, unit, reference, and operation resolution │ TypedFeaProgram + semantic diagnostics │ ├─ 5. Domain lowering │ typed geometry declarations │ typed authored mo...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## product_ownership

Evidence kind: SemanticRealization. Criterion: | Product | Owner | May contain | Must not contain | | --- | --- | --- | --- | | `FeaTokenStream` | `runmat-lexer::fea` | FEA tokens, trivia, byte spans | MATLAB tokens, semantic types | | `FeaSyntaxModule` | `runmat-parser::fea` | lossless declarations/expressions and parse errors | geometry assets, materials, filesystem access | | `BoundFeaModuleGraph` | `runmat-analysis-language::binding` | mod...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## lexer_ownership

Evidence kind: SemanticRealization. Criterion: crates/runmat-lexer/src/ lib.rs ... existing MATLAB lexer ... fea/ mod.rs lexer.rs token.rs trivia.rs diagnostic.rs tests.rs `fea/token.rs` owns only lexical categories. Keywords that are genuinely structural are tokens; operation names such as `max`, `resultant`, or `box` remain identifiers resolved by the operation catalog. This prevents grammar edits for every new operation overload.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## parser_ownership

Evidence kind: SemanticRealization. Criterion: crates/runmat-parser/src/ lib.rs ... existing MATLAB parser ... fea/ mod.rs span.rs diagnostic.rs syntax/ mod.rs module.rs import.rs declaration.rs geometry.rs material.rs model.rs study.rs sweep.rs execution.rs expression.rs type_syntax.rs parser/ mod.rs cursor.rs module.rs import.rs geometry.rs model.rs study.rs sweep.rs execution.rs expression.rs recovery.rs format/ mod.rs document.rs expressio...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## neutral_operation_catalog

Evidence kind: SemanticRealization. Criterion: First extract only the dependency-neutral catalog vocabulary shared by builtin registration and analysis typing: crates/runmat-types/src/operation/ mod.rs identity.rs frontend.rs type_pattern.rs signature.rs quantity.rs topology.rs effect.rs distribution.rs lowering_id.rs extension.rs These are immutable facts and stable identifiers. They contain no runtime function pointer, `Value`, device buffer...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## geometry_contracts

Evidence kind: SemanticRealization. Criterion: crates/runmat-geometry/core/src/ lib.rs model/ ... existing canonical geometry ... authoring/ mod.rs definition.rs source.rs parameter.rs region.rs feature/ mod.rs graph.rs operation.rs output.rs identity.rs history.rs capability.rs realization.rs evidence.rs exchange/ mod.rs capability.rs persistent_id.rs parameter.rs engineering_property.rs material_candidate.rs association.rs change_set.rs expo...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## pure_geometry_planning

Evidence kind: SemanticRealization. Criterion: crates/runmat-geometry/ops/src/ lib.rs ... existing query/stat/quality operations ... feature/ mod.rs types.rs resolve.rs plan.rs provider.rs materialize.rs regions.rs tests.rs This crate validates and plans provider-neutral geometry operations. `feature/provider.rs` defines the exact-kernel capability and execution interface. It never imports OCCT and never interprets an FEA assignment or study.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## geometry_provider_boundary

Evidence kind: SemanticRealization. Criterion: crates/runmat-geometry/io/src/ lib.rs feature/ mod.rs import/ ... existing format adapters ... ap242/ mod.rs detect.rs profile.rs capability.rs persistent_ids.rs parameters.rs constraints.rs features.rs materials.rs properties.rs associations.rs validation.rs change_set.rs export.rs occt/ ... existing evaluation/projection ... feature.rs feature_tests.rs exact_feature.cc tests/ ap242/ capabilities...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## authored_contracts

Evidence kind: SemanticRealization. Criterion: Existing `problem/` and `study/` remain the canonical authored domain and receive only genuinely missing typed contracts proven by the surface ledger. Core keeps no source syntax, module graph, compiler coordinator, filesystem trait, or operation registry. Its tests validate domain objects without parsing `.fea` source.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## compiler_host_interfaces

Evidence kind: SemanticRealization. Criterion: Add one narrow source-to-domain adapter crate. Its modules are compiler passes, not alternate model definitions: crates/runmat-analysis/language/ Cargo.toml src/ lib.rs compiler.rs source.rs diagnostic.rs module_graph/ mod.rs load.rs cycle.rs provenance.rs binding/ mod.rs scope.rs symbol.rs import.rs reference.rs semantic/ mod.rs types.rs units.rs expression.rs requiredness.rs operations/ mod.rs e...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## numerical_lowering

Evidence kind: SemanticRealization. Criterion: The numerical crate receives no parser modules. It gains only the lowering targets required to execute newly typed observation operations and geometry-derived properties: crates/runmat-analysis/fea/src/ results/ observation/ ... existing evaluation ... operation/ mod.rs projection.rs spatial.rs series.rs attributed.rs compatibility.rs properties/ ... canonical property compilation ... imported_cap...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## execution_adapter

Evidence kind: SemanticRealization. Criterion: crates/runmat-analysis/execution/src/ ... existing analysis tasks ... source_materialization/ mod.rs geometry.rs function.rs input.rs observation_operation/ mod.rs partition.rs reduction.rs publication.rs This crate converts already resolved analysis/geometry materialization needs into generic task and artifact contracts. It does not resolve source imports or interpret language expressions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## runtime_host_adapter

Evidence kind: SemanticRealization. Criterion: Do not implement the compiler here. Runtime supplies host services and adapts final values: crates/runmat-runtime/src/analysis/ mod.rs # declarations/re-exports only after extraction source/ mod.rs compile.rs # thin compiler invocation module_resolver.rs function_resolver.rs geometry_resolver.rs execution_profile_resolver.rs diagnostic_adapter.rs public/ load.rs check.rs plan.rs run.rs result.rs r...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## lsp_services

Evidence kind: SemanticRealization. Criterion: crates/runmat-lsp/src/core/ language/ mod.rs dispatch.rs service.rs matlab.rs fea/ mod.rs document.rs analysis.rs symbols.rs diagnostics.rs completion.rs hover.rs signature.rs definition.rs references.rs rename.rs formatting.rs semantic_tokens.rs code_actions.rs project/ symbol_graph.rs cross_language.rs ... existing transport/workspace ... `language/service.rs` defines operations common to a lang...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-lsp[lib]::core::fea::tests::checked_editor_product_drives_diagnostics_navigation_and_signature_help` (crates/runmat-lsp/src/core/fea/tests.rs:76) - The checked editor product drives diagnostics, navigation and signature help across an imported material file; completion, rename and formatting services are no... |

## cli_and_native_adapters

Evidence kind: SemanticRealization. Criterion: crates/runmat-cli/src/commands/fea/ mod.rs source.rs check.rs format.rs inspect.rs plan.rs run.rs placement.rs callables.rs input_materialization.rs remote/ ... existing remote preparation ... crates/runmat-execution-runner-native/src/ geometry/ mod.rs admission.rs materialization.rs feature_provider.rs ap242.rs publication.rs ... existing meshing/analysis execution ... Split the current `commands...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## desktop_and_agent_adapters

Evidence kind: SemanticRealization. Criterion: desktop/src/ runtime/domain/fea/ source-diagnostics.ts study-summary.ts plan-summary.ts execution-state.ts result-view-model.ts app/components/fea/ fea-workspace.tsx fea-editor-pane.tsx fea-diagnostics-pane.tsx fea-geometry-pane.tsx fea-study-preview.tsx fea-study-outline.tsx fea-study-detail.tsx fea-overlay-toolbar.tsx fea-execution-pane.tsx fea-results-pane.tsx presentation/ overlay-model.ts ove...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | desktop_integration | OPEN |  |

## manifest_and_facade

Evidence kind: SemanticRealization. Criterion: The dependency edits are deliberately small and auditable: | Manifest | Change | | --- | --- | | workspace `Cargo.toml` | Add `crates/runmat-analysis/language` as a member and workspace dependency. | | `runmat-analysis/language/Cargo.toml` | Depend on parser, analysis-core, geometry-core, `runmat-types`, `serde`, and `thiserror`; never builtins, value, runtime, execution, OCCT, CLI, or LSP. | | `r...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## syntax_pass

Evidence kind: SemanticRealization. Criterion: Input is one source file and a source id. Output is a lossless syntax module plus diagnostics. Parsing never loads another file. Syntax validity and formatter stability are independently testable.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-fea-syntax[lib]::tests::recovers_at_the_next_top_level_declaration` (crates/runmat-fea-syntax/src/tests.rs:145) - Malformed top-level token then valid material declaration; syntax recovery only, not complete editor services. |

## module_graph_pass

Evidence kind: SemanticRealization. Criterion: The compiler requests imported symbols through `FeaModuleSource`, canonicalizes logical paths, checks source-root confinement, detects cycles, and retains content digests and source spans. Imports select one declared symbol; there are no wildcard or ambient-path imports.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:modules]::compiler_rejects_imports_outside_the_declared_source_root` (crates/runmat-analysis/language/tests/modules.rs:109) - An import that escapes the declared source root is rejected with fea.module.invalid_path; cycle and digest retention are covered by sibling tests. |

## binding_pass

Evidence kind: SemanticRealization. Criterion: Binding creates distinct namespaces for declaration kinds and lexical scopes for model/study members. It detects duplicates, ambiguity, invalid qualification, inaccessible symbols, and composition conflicts. It does not yet evaluate geometry or lower physics.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:model_physics]::model_physics_rejects_wrong_roles_values_and_composition_conflicts` (crates/runmat-analysis/language/tests/model_physics.rs:11) - Wrong region roles, wrong values and composition conflicts between included models are rejected at binding; namespace and shadowing rules are not enumerated. |

## type_and_unit_pass

Evidence kind: SemanticRealization. Criterion: Semantic checking resolves input types, dimensioned literals, immutable expressions, declaration references, region roles, field/history types, and operation overloads. Requiredness is evaluated after the selected procedure/model/geometry is known; optional syntax does not create empty default objects.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:programs]::declaration_calls_require_every_typed_model_input` (crates/runmat-analysis/language/tests/programs.rs:156) - Declaration calls must bind every typed model input with the right type; dimensioned-literal and operation-overload checks are covered by other tests. |

## geometry_lowering

Evidence kind: SemanticRealization. Criterion: Each geometry declaration becomes exactly one `GeometryDefinition` variant. File imports retain logical source identity. Feature graphs retain stable feature paths. Derivations bind source revision plus changes. Generated declarations bind frozen-function requirements, never a live callable. Geometry inspection resolves source parameters/properties/regions into typed candidates. It does not execut...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:programs]::constructed_and_derived_geometry_lower_parameterized_feature_graphs` (crates/runmat-analysis/language/tests/programs.rs:588) - Constructed and derived geometry lower parameterized feature graphs.<br>`runmat-analysis-language[test:programs]::generated_geometry_freezes_function_provenance_and_typed_arguments` (crates/runmat-analysis/language/tests/programs.rs:666) - Generated geometry freezes imported-function provenance and typed arguments; constructed, derived and file-import lowering are other cases. |

## physical_lowering

Evidence kind: SemanticRealization. Criterion: Materials, assignments, constraints, boundaries, connectors, interfaces, signals, excitations, initial conditions, and domains lower into existing analysis-core contracts. Imported engineering properties must pass through one explicit candidate-to-material conversion. Model composition uses stable typed identity and emits no override order.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:programs]::material_lowering_rejects_unknown_members_instead_of_dropping_them` (crates/runmat-analysis/language/tests/programs.rs:255) - Material lowering rejects unknown members instead of dropping them; loads, constraints, interfaces and signals lowering are not covered here. |

## study_and_sweep_lowering

Evidence kind: SemanticRealization. Criterion: The compiler binds model parameters and typed regions to one geometry, verifies mesh lineage, constructs the semantic procedure/state graph, validates accuracy and retained state, then lowers sweep axes/cases over declared study inputs. Scheduler tasks are not constructed here.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:studies]::study_rejects_mesh_lineage_mismatch_and_out_of_bounds_model_constants` (crates/runmat-analysis/language/tests/studies.rs:296) - Mesh-lineage mismatch and out-of-bounds model constants are rejected when a study binds geometry and mesh; sweep lowering is covered elsewhere. |

## output_lowering

Evidence kind: SemanticRealization. Criterion: Operation resolution has already selected a stable overload id. Lowering converts it to typed observation requests and expression nodes. `max(field, over = region)` produces an attributed spatial-extremum contract; no FEA evaluator sees the text `"max"`. Cost and distributability facts enter planning before solve.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:studies]::study_outputs_lower_fields_projections_and_attributed_spatial_extrema` (crates/runmat-analysis/language/tests/studies.rs:142) - Outputs lower to fields, projections and attributed spatial extrema; catalog overload identity is covered in the operations tests. |

## policy_merge

Evidence kind: SemanticRealization. Criterion: The compiler validates only source-authored policy. The runtime later merges it with project and CLI policy through a typed precedence function. The merged policy may change placement/resources, not physical meaning or accuracy. Its logical and attempt identities remain separate from the study-definition identity.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## check_plan_run

Evidence kind: SemanticRealization. Criterion: check parse + bind + type/unit check + import/callable signature resolution + available geometry/AP242 capability inspection - no user function execution - no solver or cluster allocation plan check + canonical materialization/mesh/study/observation/resource graphs + provider/capability admission - no claim that generated values or numerical results exist run plan + bounded materialization + geome...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat[test:fea_seeded_faults]::seeded_setup_faults_are_rejected_before_meshing` (crates/runmat-cli/tests/fea_seeded_faults.rs:123) - Seeded setup faults in the cantilever example are rejected by the run command with the expected diagnostic before any mesh artifact is produced. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## layered_identity

Evidence kind: SemanticRealization. Criterion: - Source identity: logical module path plus exact source digest. - Authored declaration identity: bound symbol path plus canonical typed content. - Generator identity: imported function source graph, captures, arguments, and policy. - Geometry-definition identity: source/feature graph/derivation and typed inputs. - Geometry-realization identity: definition plus kernel/profile cohort and exact resu...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## structured_diagnostics

Evidence kind: SemanticRealization. Criterion: One diagnostic envelope crosses lexer, parser, semantic, geometry, and host boundaries: - stable code and severity; - primary source span and source id; - typed related spans across imported `.fea` or `.m` files; - declaration/symbol/operation identity when known; - concise message and structured facts; - optional safe fix/code action; - phase and completeness (`syntax`, `binding`, `semantic`, `in...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## unit_verification

Evidence kind: SemanticRealization. Criterion: - lexer tokens, trivia, invalid bytes, and recovery; - parser declarations, precedence, incomplete files, diagnostics, and format idempotence; - module resolution, cycle/traversal rejection, symbol namespaces, and composition; - type/unit inference and invalid operation overloads; - every output operation descriptor, result type, lowering, tie rule, and docs entry; - geometry feature-graph validat...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## cross_layer_verification

Evidence kind: SemanticRealization. Criterion: Use small fixtures that cross one real seam rather than giant all-purpose fixtures: tests/fea_language/ syntax/ modules/ geometry/ ap242/ materials/ studies/ sweeps/ outputs/ execution/ diagnostics/ migration/ Each fixture has source, expected diagnostics or resolved summary, and only when applicable a canonical result/evidence expectation. During development, internal parity fixtures compare the ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## lsp_verification

Evidence kind: SemanticRealization. Criterion: Verify completion, hover, signature help, definition, references, rename, formatting, semantic tokens, and diagnostics for complete and incomplete `.fea` files. Cross-language tests navigate from `import function impactPulse from "./impactPulse.m"` to the `.m` definition and report a signature/effect mismatch at the import/call site.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-lsp[lib]::core::fea::tests::checked_editor_product_drives_diagnostics_navigation_and_signature_help` (crates/runmat-lsp/src/core/fea/tests.rs:76) - Diagnostics, definition and signature help over a complete multi-file source; incomplete-file recovery and cross-language navigation are not covered here. |

## execution_release_verification

Evidence kind: SemanticRealization. Criterion: - one minimal local study with no sweep/execution declaration; - one parameterized constructed-geometry study; - one immutable derived-geometry sweep; - one `.m`-generated geometry study; - AP242 read-only parameter/material import fixture; - AP242 qualified regeneration fixture where the supported kernel/profile exists; - local versus encrypted remote identity/result equivalence; - independent fa...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## architecture_enforcement

Evidence kind: SemanticRealization. Criterion: Add source-level architecture checks for: - no `serde_yaml` or YAML document types in the final runtime FEA load path; - no FEA parser/AST imports in numerical, execution-worker, or geometry-kernel modules; - no runtime/CLI/LSP dependencies from analysis core or geometry crates; - no `runmat-builtins` or `runmat-value` dependency from analysis core or analysis-language; - no AP242 raw entity types...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## coherent_slice_boundaries

Evidence kind: SemanticRealization. Criterion: The slices are deliberately large enough to deliver coherent vertical behavior, but each ends at a real ownership or evidence boundary.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## surface_reconciliation

Evidence kind: SemanticRealization. Criterion: - Create the code-derived current/target surface ledger. - Extract dependency-neutral operation facts into `runmat-types`; add source-host traits, the read-only analysis catalog view, and deterministic catalog composition/fingerprint checks. - Establish architecture checks preventing additions to the known concentration points. - Surface-ledger and catalog-boundary work preserves admitted loader s...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## frontend_completion

Evidence kind: SemanticRealization. Criterion: - Implement FEA lexer, lossless syntax tree, parser recovery, and formatter. - Add syntax diagnostics and parser/formatter fixtures for the complete grammar. - The syntax frontend owns neither domain lowering nor independent public-loader authority.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## semantic_compiler_completion

Evidence kind: SemanticRealization. Criterion: - Implement module graph, explicit imports, symbols, types, units, requiredness, and diagnostics. - Lower materials/models/studies/sequences/sweeps/execution into existing typed contracts. - Implement direct `.m` symbol facts without executing functions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## output_catalog_completion

Evidence kind: SemanticRealization. Criterion: - Extend the canonical catalog and register FEA overloads. - Implement attributed result types and lower existing observation/reduction contracts. - Complete LSP/catalog/reference integration and distributed reduction evidence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## geometry_realization_completion

Evidence kind: SemanticRealization. Criterion: - Add normalized geometry source/feature/lineage contracts and provider-neutral planning. - Implement the bounded OCCT feature provider and persistent history. - The qualified bounded native profile must cover primitives, affine transforms/instances, extrusion, revolution, governed Booleans, holes, pockets, fillets, and chamfers. Removal operations reject zero-volume overlap; edge treatments resol...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## ap242_completion

Evidence kind: SemanticRealization. Criterion: - Inventory current XCAF translation and raw AP242 surface. - Add parameters, properties, materials, associations, persistent ids, capabilities, and fixtures. - Add explicit typed material adoption and read-only parameter use. - Add qualified regeneration/change-set/export paths only for proven profiles. Required ownership: geometry I/O owns a bounded Part 21 decoder, SI/derived-unit normalization...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## product_cutover

Evidence kind: SemanticRealization. Criterion: - Route runtime, CLI, LSP, and Desktop through the new compiler. - Migrate examples and compare resolved semantics/results across implemented procedures. - Publish language/reference/guides and make docs executable in release. - Delete YAML loading, duplicate schema/default/template logic, and dead compatibility code. - Split touched runtime/LSP/Desktop concentration points and verify no net repla...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## compiler_acceptance

Evidence kind: SemanticRealization. Criterion: - Every current field and behavior has a reviewed surface-ledger disposition. - Lexer/parser own syntax; analysis-language owns compiler passes; analysis-core owns authored contracts; geometry core/ops/io own normalized contracts, pure planning and provider translation; execution adapts generic tasks; runtime/LSP/CLI/Desktop own their host and presentation roles. - One parser/lowering path produce...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## focused_modules

Evidence kind: SemanticRealization. Criterion: Compiler modules own one named responsibility and direction of translation. Production modules normally remain below roughly 600 lines and orchestration facades below roughly 250; review must split a concentration of responsibilities rather than arbitrarily fragment one owner. Existing runtime analysis coordinator, LSP analysis coordinator, document loader, figure renderer and Desktop study surfac...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

