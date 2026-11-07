# FEA-AUTHORING-LANGUAGE-002: Declarative source semantics and product tooling

0 PASS, 12 PARTIAL, 1 FAIL, 19 OPEN, 0 TAGGED. Generated; do not edit.

## language_scope

Evidence kind: SemanticRealization. Criterion: RunMat `.fea` files are declarative source files for finite-element and coupled-physics work. The language describes physical models, studies, parameter spaces, requested results, and optional execution policy. It resolves to the same strongly typed canonical analysis contracts used by the RunMat API, native workers, cluster workers, persistence, and result tooling. The language is deliberately sm...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## declarative_semantics

Evidence kind: SemanticRealization. Criterion: 1. **Physical meaning is visible.** Materials, assignments, constraints, loads, procedures, transitions, and output requests are explicit. The compiler may derive bookkeeping, but may not invent physical content. 2. **Requiredness is semantic and minimal.** A declaration requires only the information needed to make that declaration unambiguous. A local one-off study does not declare a sweep, execu...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:studies]::modal_reduction_enrichment_lowers_into_typed_procedure_settings` (crates/runmat-analysis/language/tests/studies.rs:411) - Modal residual vectors and modal-reduced static correction lower into the typed procedure settings; malformed values are rejected with a procedure diagnostic.<br>`runmat-analysis-language[test:studies]::procedure_requiredness_rejects_hidden_physical_defaults` (crates/runmat-analysis/language/tests/studies.rs:493) - A transient procedure with no start_time is rejected with a located procedure diagnostic and no study is produced, so physical defaults are not invented; covers... |

## imports_and_scopes

Evidence kind: SemanticRealization. Criterion: A file is a module and may contain any number of peer top-level declarations. There is no artificial one-study or one-model limit. Imports have an explicit symbol kind, local name, and source file. A function import names a function file directly; a package folder is not required. The imported `.m` file must define a compatible named function entrypoint. Scripts are rejected because they have no s...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:studies]::study_preserves_the_defining_module_of_imported_declarations` (crates/runmat-analysis/language/tests/studies.rs:259) - Imported geometry, mesh and model keep their defining module identity through study resolution; function and package import forms and scope shadowing rules are ... |

## declaration_requiredness

Evidence kind: SemanticRealization. Criterion: | Declaration | Purpose | Required members | Optional members | | --- | --- | --- | --- | | `geometry` | Admit CAD or canonical geometry | exactly one source | authored units, admission assertions | | `mesh` | Name an existing mesh or a meshing request | geometry plus source/request | quality goals, partition hints | | `material` | Reusable capability aggregate | name or at least one capability | ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:studies]::procedure_requiredness_rejects_hidden_physical_defaults` (crates/runmat-analysis/language/tests/studies.rs:493) - One required procedure member (start_time) is enforced for structural transient; the full required/optional member table for all declaration kinds is not exerci... |

## formulation_admission

Evidence kind: SemanticRealization. Criterion: A minimal standalone study implies neither remote execution nor a parameter sweep. No empty `execution`, `sweep`, `outputs`, or scaling block is needed. Formulation admission is physics/numerics admission, not placement policy. A study has no formulation setting: the registry admits exactly the realization the authored model and mesh select, and that exact realization survives in realized study id...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:studies]::study_has_no_formulation_setting` (crates/runmat-analysis/language/tests/studies.rs:43) - A study without any formulation setting checks and realizes; the presence of no setting is asserted, remote and sweep non-implication are not. |

## model_composition

Evidence kind: SemanticRealization. Criterion: A `model` owns reusable physical intent: typed inputs (including typed region inputs), materials or material references, property assignments, constraints, natural boundaries, connectors, interfaces, signals, excitations, load cases, initial conditions, spectra, and family-domain declarations. Geometry declarations own the selectors that produce regions; a model either refers to those regions expl...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:models]::model_composition_and_material_assignments_retain_qualified_sources` (crates/runmat-analysis/language/tests/models.rs:11) - Included models and material assignments retain their qualified source modules across imports; the full list of model members is not exercised.<br>`runmat-analysis-language[test:models]::model_composition_rejects_cycles_and_conflicting_assignments` (crates/runmat-analysis/language/tests/models.rs:117) - Composition cycles and conflicting assignments are rejected. |

## spectral_inputs

Evidence kind: SemanticRealization. Criterion: Spectral environments are named model-owned physical inputs. They are not untyped procedure options and are not placement policy. A procedure refers to them through the bound model, so the reference is navigable and the compiler can prove that the selected model actually owns it. A response spectrum declares exactly one monotonically increasing abscissa (`frequencies` or `periods`), ordinates whos...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:spectral_inputs]::response_spectrum_is_typed_and_retains_complete_solver_policy` (crates/runmat-analysis/language/tests/spectral_inputs.rs:25) - A response spectrum is a typed model-owned input retaining the complete solver policy (combination rules, mass truncation, rigid response); procedure references... |

## typed_expressions

Evidence kind: SemanticRealization. Criterion: Inputs are immutable, strongly typed, and scoped to the declaration that consumes them: An input without a default must be bound by its caller. A study may expose model inputs, and a sweep may bind exposed study inputs. CLI or RunMat-language invocation may bind study inputs using the same typed contract. Bounds and allowed sets are validation constraints, not clipping rules. `let` introduces an i...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:programs]::compiler_checks_inputs_defaults_locals_and_outputs_as_one_program` (crates/runmat-analysis/language/tests/programs.rs:11) - Constructed geometry source with local/default quantity typing and one node-displacement output; source checking only.<br>`runmat-analysis-language[test:programs]::compiler_rejects_wrong_input_defaults_and_duplicate_outputs` (crates/runmat-analysis/language/tests/programs.rs:99) - Wrong dimension and duplicate-output diagnostic fixtures, not all invalid contracts. |

## semantic_state_flow

Evidence kind: SemanticRealization. Criterion: A study binds one model instance to geometry and mesh, declares an ordered semantic procedure graph, and names requested outputs. It is the normal runnable root. Analyses own procedure-specific options, active model content, accuracy obligations, and named state retention. A dependency such as `after analysis.modes` is semantic state flow, not merely a scheduler edge. The procedure kind determines...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:studies]::study_lowers_typed_modal_state_flow_into_the_canonical_sequence` (crates/runmat-analysis/language/tests/studies.rs:357) - A modal analysis retaining a basis feeds a modal transient through a named state reference lowered into the canonical sequence; other state kinds and the study-... |

## frozen_function_inputs

Evidence kind: SemanticRealization. Criterion: An imported `.m` function is a source-resolvable generator, not a live solver callback: The semantic checker resolves the imported symbol through the ordinary RunMat project graph, checks its typed inputs/outputs and effects, and records source/capture provenance. Before a solve, the function is frozen and evaluated in a bounded preprocessing task. Its result is validated and published as a canoni...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:signals]::generated_signal_binds_imported_source_and_typed_sampling` (crates/runmat-analysis/language/tests/signals.rs:84) - An imported .m generator binds source and typed sampling (range, rate, interpolation, extrapolation, tolerance) with typed arguments; capture provenance and eff... |

## initial_conditions

Evidence kind: SemanticRealization. Criterion: Initial conditions describe physical state at the beginning of a procedure. They belong to a reusable model, while the consuming analysis selects which named conditions apply before that step. A model therefore does not embed a study-step name: The closed initial-condition kinds are structural displacement, velocity, and angular velocity; thermal temperature; electric potential; magnetic vector po...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:initial_conditions]::studies_reject_unknown_or_physics_incompatible_initial_state` (crates/runmat-analysis/language/tests/initial_conditions.rs:247) - Unknown or physics-incompatible initial state is rejected at study check; the closed initial-condition kind list is not enumerated. |

## sweep_identity

Evidence kind: SemanticRealization. Criterion: A sweep is optional and wraps a parameterized study without changing the study's physical definition: Axes bind declared study inputs and have named values. Names give cases stable identities that do not change when source order changes. `product` forms the Cartesian product and `zip` pairs named positions. Correlated or sparse combinations use explicit cases: Every case is independently identifia...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:sweeps]::axis_sweeps_expand_product_and_zip_with_stable_case_identities` (crates/runmat-analysis/language/tests/sweeps.rs:54) - Axis sweeps expand product and zip with stable named case identities; independence of identity from source order and the placement envelopes are not covered. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## execution_precedence

Evidence kind: SemanticRealization. Criterion: Execution policy is optional. With no policy in the source, CLI/project defaults choose local or remote placement and automatic backend selection. The study remains runnable and portable. `fanout` controls independent schedulable work such as cases, frequencies, directions, or independent recovery partitions. `cooperative` requests tightly coupled distributed numerical work with shared operator ow...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:execution_policies]::execution_policy_is_typed_and_inherited_by_a_sweep` (crates/runmat-analysis/language/tests/execution_policies.rs:34) - A typed execution policy is inherited by a sweep; CLI and project default precedence is not covered.<br>`runmat-analysis-language[test:execution_policies]::execution_policy_rejects_contradictions_and_invalid_limits` (crates/runmat-analysis/language/tests/execution_policies.rs:144) - Contradictory or out-of-range policy members are rejected.<br>`runmat-analysis-language[test:execution_policies]::memory_per_worker_shares_the_wire_integer_bound` (crates/runmat-analysis/language/tests/execution_policies.rs:172) - memory_per_worker is a positive whole byte count no larger than the shared FEA wire-integer bound 2^53 - 1; 2^53 and above, zero and fractional bytes are reject... |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## grammar_and_recovery

Evidence kind: SemanticRealization. Criterion: The concrete grammar will be frozen with parser implementation, but its structural contract is: module := import* declaration* import := "import" import_kind identifier "from" string declaration := geometry | mesh | material | model | study | sweep | execution geometry := "geometry" identifier ("from" geometry_ref)? geometry_block geometry_source := file_source | construct_block | derive_block | g...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | FAIL | `runmat-fea-syntax[lib]::tests::parses_representative_language_surface` (crates/runmat-fea-syntax/src/tests.rs:66) - Representative declarations, spans and imports only; no complete grammar or semantics proof.<br>`runmat-fea-syntax[lib]::tests::formatter_is_canonical_and_idempotent` (crates/runmat-fea-syntax/src/tests.rs:272) - One formatter fixture and second formatting equality; no complete recovery or adversarial grammar credit. |

## compiler_and_editor_ownership

Evidence kind: SemanticRealization. Criterion: The implementation reuses RunMat infrastructure but keeps the FEA language as its own frontend: .fea text -> FEA lexer tokens -> lossless/syntax AST with source spans -> module and symbol resolution -> typed semantic FEA model -> authored-contract validation -> geometry and callable resolution -> canonical analysis contracts -> planning, execution, persistence, and evidence - `runmat-lexer` may ho...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## study_presentation

Evidence kind: SemanticRealization. Criterion: The compiler/runtime publishes one typed study-presentation projection from the resolved study and canonical geometry. It contains no source syntax decisions and no rendering instructions. Each presentable declaration carries its stable authored identity, source span, activation, exact geometry/topology targets, role, physical quantity, units, frame, direction or normal semantics, distribution, sc...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | desktop_integration | OPEN |  |

## atomic_surface_cutover

Evidence kind: SemanticRealization. Criterion: The DSL must be designed from the true implementation surface, not only from old documentation or example YAML. Before replacing authoring, build a machine-checkable inventory of: - every currently accepted document field and variant; - every `runmat-analysis-core` authored type and validation rule; - every runtime/CLI/Desktop study construction field; - every value that survives into canonical id...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## executable_documentation

Evidence kind: SemanticRealization. Criterion: The public FEA documentation gains this structure: The language documentation contains a landing page, conceptual module/type/geometry/state/input/output/execution/function/diagnostic pages, complete grammar/declaration/procedure/constraint/operation/geometry references, and runnable static/parameterized/transient/modal/sweep/scaling/large-result guides. Reference tables are generated from, or mec...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## generated_agent_context

Evidence kind: SemanticRealization. Criterion: FEA-mode agent context is derived from the standing language specification, checked examples, and the canonical procedure and operation registries. It supplies a bounded language summary, relevant project symbols and importable `.m` signatures, and the exact edit/check/plan/run workflow. The primary action is editing the `.fea` source and responding to structured checker diagnostics. Broad typed t...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## language_acceptance

Evidence kind: SemanticRealization. Criterion: The language is ready to replace the current authoring surface only when: - the current/target surface inventory has no unexplained fields or gaps; - grammar, formatter, parser recovery, semantic typing, units, imports, and cross-language symbol resolution have focused and adversarial tests; - imported, constructed, derived, and `.m`-generated geometry share one canonical artifact, immutable revis...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

