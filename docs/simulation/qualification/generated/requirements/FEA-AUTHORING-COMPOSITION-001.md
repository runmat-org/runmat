# FEA-AUTHORING-COMPOSITION-001: Typed declarative authoring and explicit composition

0 PASS, 1 PARTIAL, 0 FAIL, 14 OPEN, 0 TAGGED. Generated; do not edit.

## declaration_ownership

Evidence kind: SemanticRealization. Criterion: The target authoring surface is the purpose-built declarative language specified in the canonical typed declarative language contract, not YAML deserialization into execution contracts. The `.fea` extension selects the language. Files have no repeated version/kind preamble and may declare multiple peer models, studies, sweeps, geometries, meshes, materials, and execution policies. Nesting follows ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## explicit_imports

Evidence kind: SemanticRealization. Criterion: Composition uses explicit source-resolvable imports and named declarations, for example `import model bracket from "./models/bracket.fea"`. Equal definitions deduplicate by typed stable identity; conflicts fail with both source locations. There is no last-writer-wins override order, hidden template/default expansion, or Rust-side builtin that manufactures physical objects. Geometry-bound selectors...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | PARTIAL | `runmat-analysis-language[test:modules]::compiler_loads_transitive_modules_and_binds_explicit_symbols` (crates/runmat-analysis/language/tests/modules.rs:11) - Mock host checks three-module transitive imports, explicit model/function binding and symbol-use provenance; no geometry execution, numerical result or remote e...<br>`runmat-analysis-language[test:modules]::compiler_reports_cycles_duplicates_and_missing_exports` (crates/runmat-analysis/language/tests/modules.rs:66) - Mock-host binding diagnostics identify cyclic imports, duplicate symbols and missing exports; no complete semantic or numerical qualification. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

## clean_cutover

Evidence kind: SemanticRealization. Criterion: The migration is a clean cut with one FEA lexer/parser/semantic lowering path. There is no YAML conversion command or compatibility tooling: the product has not shipped this authoring format, so the old implementation is only an internal inventory/parity source while the replacement is built. No dual YAML/DSL authority, compatibility switch, duplicate canonicalization, or dead schema remains. Veri...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | mesh_independent | source_check | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |

