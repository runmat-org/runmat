# FEA-AUTHORING-LANGUAGE-002: Declarative source semantics and product tooling

2 PASS, 0 FAIL, 30 OPEN, 0 TAGGED. Generated; do not edit.

## language_scope

Evidence kind: SemanticRealization. Criterion: RunMat `.fea` files are declarative source files for finite-element and coupled-physics work. The language describes physical models, studies, parameter spaces, requested results, and optional execution policy. It resolves to the same strongly typed canonical analysis contracts used by the RunMat API, native workers, cluster workers, persistence, and result tooling. The language is deliberately sm...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## declarative_semantics

Evidence kind: SemanticRealization. Criterion: 1. **Physical meaning is visible.** Materials, assignments, constraints, loads, procedures, transitions, and output requests are explicit. The compiler may derive bookkeeping, but may not invent physical content. 2. **Requiredness is semantic and minimal.** A declaration requires only the information needed to make that declaration unambiguous. A local one-off study does not declare a sweep, execu...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## imports_and_scopes

Evidence kind: SemanticRealization. Criterion: A file is a module and may contain any number of peer top-level declarations. There is no artificial one-study or one-model limit. Imports have an explicit symbol kind, local name, and source file. A function import names a function file directly; a package folder is not required. The imported `.m` file must define a compatible named function entrypoint. Scripts are rejected because they have no s...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## declaration_requiredness

Evidence kind: SemanticRealization. Criterion: | Declaration | Purpose | Required members | Optional members | | --- | --- | --- | --- | | `geometry` | Admit CAD or canonical geometry | exactly one source | authored units, admission assertions | | `mesh` | Name an existing mesh or a meshing request | geometry plus source/request | quality goals, partition hints | | `material` | Reusable capability aggregate | name or at least one capability | ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## formulation_admission

Evidence kind: SemanticRealization. Criterion: A minimal standalone study implies neither remote execution nor a parameter sweep. No empty `execution`, `sweep`, `outputs`, or scaling block is needed. Formulation admission is physics/numerics admission, not placement policy. A study has no formulation setting: the registry admits exactly the realization the authored model and mesh select, and that exact realization survives in realized study id...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## model_composition

Evidence kind: SemanticRealization. Criterion: A `model` owns reusable physical intent: typed inputs (including typed region inputs), materials or material references, property assignments, constraints, natural boundaries, connectors, interfaces, signals, excitations, load cases, initial conditions, spectra, and family-domain declarations. Geometry declarations own the selectors that produce regions; a model either refers to those regions expl...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## spectral_inputs

Evidence kind: SemanticRealization. Criterion: Spectral environments are named model-owned physical inputs. They are not untyped procedure options and are not placement policy. A procedure refers to them through the bound model, so the reference is navigable and the compiler can prove that the selected model actually owns it. A response spectrum declares exactly one monotonically increasing abscissa (`frequencies` or `periods`), ordinates whos...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## typed_expressions

Evidence kind: SemanticRealization. Criterion: Inputs are immutable, strongly typed, and scoped to the declaration that consumes them: An input without a default must be bound by its caller. A study may expose model inputs, and a sweep may bind exposed study inputs. CLI or RunMat-language invocation may bind study inputs using the same typed contract. Bounds and allowed sets are validation constraints, not clipping rules. `let` introduces an i...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PASS | `runmat-analysis-language[test:programs]::compiler_checks_inputs_defaults_locals_and_outputs_as_one_program` (crates/runmat-analysis/language/tests/programs.rs:11) - Constructed geometry source with local/default quantity typing and one node-displacement output; source checking only.<br>`runmat-analysis-language[test:programs]::compiler_rejects_wrong_input_defaults_and_duplicate_outputs` (crates/runmat-analysis/language/tests/programs.rs:93) - Wrong dimension and duplicate-output diagnostic fixtures, not all invalid contracts. |

## semantic_state_flow

Evidence kind: SemanticRealization. Criterion: A study binds one model instance to geometry and mesh, declares an ordered semantic procedure graph, and names requested outputs. It is the normal runnable root. Analyses own procedure-specific options, active model content, accuracy obligations, and named state retention. A dependency such as `after analysis.modes` is semantic state flow, not merely a scheduler edge. The procedure kind determines...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## frozen_function_inputs

Evidence kind: SemanticRealization. Criterion: An imported `.m` function is a source-resolvable generator, not a live solver callback: The semantic checker resolves the imported symbol through the ordinary RunMat project graph, checks its typed inputs/outputs and effects, and records source/capture provenance. Before a solve, the function is frozen and evaluated in a bounded preprocessing task. Its result is validated and published as a canoni...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## initial_conditions

Evidence kind: SemanticRealization. Criterion: Initial conditions describe physical state at the beginning of a procedure. They belong to a reusable model, while the consuming analysis selects which named conditions apply before that step. A model therefore does not embed a study-step name: The closed initial-condition kinds are structural displacement, velocity, and angular velocity; thermal temperature; electric potential; magnetic vector po...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## sweep_identity

Evidence kind: SemanticRealization. Criterion: A sweep is optional and wraps a parameterized study without changing the study's physical definition: Axes bind declared study inputs and have named values. Names give cases stable identities that do not change when source order changes. `product` forms the Cartesian product and `zip` pairs named positions. Correlated or sparse combinations use explicit cases: Every case is independently identifia...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## execution_precedence

Evidence kind: SemanticRealization. Criterion: Execution policy is optional. With no policy in the source, CLI/project defaults choose local or remote placement and automatic backend selection. The study remains runnable and portable. `fanout` controls independent schedulable work such as cases, frequencies, directions, or independent recovery partitions. `cooperative` requests tightly coupled distributed numerical work with shared operator ow...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## grammar_and_recovery

Evidence kind: SemanticRealization. Criterion: The concrete grammar will be frozen with parser implementation, but its structural contract is: module := import* declaration* import := "import" import_kind identifier "from" string declaration := geometry | mesh | material | model | study | sweep | execution geometry := "geometry" identifier ("from" geometry_ref)? geometry_block geometry_source := file_source | construct_block | derive_block | g...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PASS | `runmat-parser[lib]::fea::tests::parses_representative_language_surface` (crates/runmat-parser/src/fea/tests.rs:66) - Representative declarations, spans and imports only; no complete grammar or semantics proof.<br>`runmat-parser[lib]::fea::tests::formatter_is_canonical_and_idempotent` (crates/runmat-parser/src/fea/tests.rs:272) - One formatter fixture and second formatting equality; no complete recovery or adversarial grammar credit. |

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

