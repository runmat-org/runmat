use runmat_hir::{
    CallKind, CallSyntax, EnvironmentEffect as HirEnvironmentEffect, FunctionHandleTarget,
    HirCallableRef, HirDiagnostic, HirDiagnosticSeverity, HirError, LoweringContext,
    LoweringResult, Span,
};
use runmat_mir::{analysis::AnalysisStore, MirAssembly, MirStmtKind};
use runmat_parser::{CompatMode, ParserOptions};
use runmat_vm::CompileError;
use serde::{Deserialize, Serialize};

pub const DIAGNOSTIC_UNRESOLVED_FUNCTION: &str = "RM-RES0001";
pub const DIAGNOSTIC_RUNTIME_DEPENDENT_RESOLUTION: &str = "RM-RES0002";
pub const DIAGNOSTIC_RUNTIME_METHOD_DISPATCH: &str = "RM-RES0003";
pub const DIAGNOSTIC_SOURCE_CATALOG: &str = "RM-CAT0001";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionState {
    Resolved,
    Unresolved,
    RuntimeDependent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolutionEvidence {
    pub name: String,
    pub span: Span,
    pub state: ResolutionState,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub definition: Option<runmat_package::ProjectSymbolDefinition>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisCompleteness {
    Complete,
    Partial,
    RuntimeDependent,
    Unavailable,
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisDomains {
    pub syntax: AnalysisCompleteness,
    pub name_resolution: AnalysisCompleteness,
    pub definite_assignment: AnalysisCompleteness,
    pub effects: AnalysisCompleteness,
    pub types: AnalysisCompleteness,
    pub shapes: AnalysisCompleteness,
    pub async_safety: AnalysisCompleteness,
}

impl AnalysisDomains {
    fn unavailable_after_syntax() -> Self {
        Self {
            syntax: AnalysisCompleteness::Complete,
            name_resolution: AnalysisCompleteness::Unavailable,
            definite_assignment: AnalysisCompleteness::Unavailable,
            effects: AnalysisCompleteness::Unavailable,
            types: AnalysisCompleteness::Unavailable,
            shapes: AnalysisCompleteness::Unavailable,
            async_safety: AnalysisCompleteness::Unavailable,
        }
    }

    fn unavailable_after_hir() -> Self {
        Self {
            syntax: AnalysisCompleteness::Complete,
            name_resolution: AnalysisCompleteness::Partial,
            definite_assignment: AnalysisCompleteness::Unavailable,
            effects: AnalysisCompleteness::Unavailable,
            types: AnalysisCompleteness::Unavailable,
            shapes: AnalysisCompleteness::Unavailable,
            async_safety: AnalysisCompleteness::Unavailable,
        }
    }

    fn completed_frontend() -> Self {
        Self {
            syntax: AnalysisCompleteness::Complete,
            name_resolution: AnalysisCompleteness::Complete,
            definite_assignment: AnalysisCompleteness::Complete,
            effects: AnalysisCompleteness::Complete,
            // The facts are useful today, but the comprehensive builtin and
            // interprocedural propagation audit is tracked separately.
            types: AnalysisCompleteness::Partial,
            shapes: AnalysisCompleteness::Partial,
            async_safety: AnalysisCompleteness::Complete,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseFailure {
    pub message: String,
    pub position: usize,
    pub found_token: Option<String>,
    pub expected: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FrontendAnalysis {
    pub lowering: Option<LoweringResult>,
    pub mir: Option<MirAssembly>,
    pub facts: Option<AnalysisStore>,
    pub semantic_facts: Option<crate::semantic::SemanticDocumentFacts>,
    pub diagnostics: Vec<HirDiagnostic>,
    pub parse_failure: Option<ParseFailure>,
    pub lowering_failure: Option<HirError>,
    pub compile_failure: Option<CompileError>,
    pub bytecode: Option<runmat_vm::Bytecode>,
    pub resolution: Vec<ResolutionEvidence>,
    pub project_revision: Option<runmat_package::ProjectRevision>,
    pub domains: AnalysisDomains,
}

impl FrontendAnalysis {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == HirDiagnosticSeverity::Error)
    }

    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == HirDiagnosticSeverity::Warning)
            .count()
    }
}

/// Run the canonical source-local frontend. Project and host resolution inputs
/// are supplied through `LoweringContext`; CLI, LSP, and runtime adapters are
/// responsible for constructing that context from the same source catalog.
pub fn analyze_source(
    source: &str,
    compat: CompatMode,
    lowering_context: &LoweringContext<'_>,
) -> FrontendAnalysis {
    analyze_source_with_catalog(source, compat, lowering_context, None)
}

pub fn analyze_source_with_catalog(
    source: &str,
    compat: CompatMode,
    lowering_context: &LoweringContext<'_>,
    source_catalog: Option<&runmat_package::DiscoveredSourceSymbols>,
) -> FrontendAnalysis {
    let ast = match runmat_parser::parse_with_options(source, ParserOptions::new(compat)) {
        Ok(ast) => ast,
        Err(error) => {
            let failure = ParseFailure {
                message: error.message,
                position: error.position,
                found_token: error.found_token,
                expected: error.expected,
            };
            let message = parse_failure_message(&failure);
            let span = source_position_span(source, failure.position);
            return FrontendAnalysis {
                lowering: None,
                mir: None,
                facts: None,
                semantic_facts: None,
                diagnostics: vec![HirDiagnostic::new(
                    "RunMat:ParseError",
                    HirDiagnosticSeverity::Error,
                    message,
                    span,
                )
                .with_primary_label("syntax error occurs here")
                .with_category("syntax")],
                parse_failure: Some(failure),
                lowering_failure: None,
                compile_failure: None,
                bytecode: None,
                resolution: Vec::new(),
                project_revision: source_catalog.and_then(|catalog| catalog.project_revision()),
                domains: AnalysisDomains::unavailable_after_syntax(),
            };
        }
    };

    analyze_program_with_catalog(&ast, lowering_context, source_catalog)
}

/// Run the canonical frontend from a parsed program. Runtime compilation uses
/// this entry point after it has appended companion sources to the primary AST;
/// CLI and LSP use [`analyze_source_with_catalog`] so parsing still happens in
/// this crate. Every consumer therefore shares lowering, MIR analysis,
/// resolution auditing, linting, and bytecode compilation.
pub fn analyze_program_with_catalog(
    ast: &runmat_parser::Program,
    lowering_context: &LoweringContext<'_>,
    source_catalog: Option<&runmat_package::DiscoveredSourceSymbols>,
) -> FrontendAnalysis {
    let lowering = match runmat_hir::lower(ast, lowering_context) {
        Ok(lowering) => lowering,
        Err(error) => {
            let span = error.span.unwrap_or(Span { start: 0, end: 0 });
            let code = error
                .identifier
                .clone()
                .unwrap_or_else(|| "RunMat:LoweringError".to_string());
            return FrontendAnalysis {
                lowering: None,
                mir: None,
                facts: None,
                semantic_facts: None,
                diagnostics: vec![HirDiagnostic::new(
                    code,
                    HirDiagnosticSeverity::Error,
                    error.message.clone(),
                    span,
                )
                .with_primary_label("semantic error occurs here")
                .with_category("semantic")],
                parse_failure: None,
                lowering_failure: Some(error),
                compile_failure: None,
                bytecode: None,
                resolution: Vec::new(),
                project_revision: source_catalog.and_then(|catalog| catalog.project_revision()),
                domains: AnalysisDomains::unavailable_after_hir(),
            };
        }
    };

    let mir = match runmat_mir::lowering::lower_assembly(&lowering.assembly) {
        Ok(mir) => mir,
        Err(error) => {
            let span = error.span.unwrap_or(Span { start: 0, end: 0 });
            let code = error
                .identifier
                .clone()
                .unwrap_or_else(|| "RunMat:MirLoweringError".to_string());
            return FrontendAnalysis {
                lowering: Some(lowering),
                mir: None,
                facts: None,
                semantic_facts: None,
                diagnostics: vec![HirDiagnostic::new(
                    code,
                    HirDiagnosticSeverity::Error,
                    error.message.clone(),
                    span,
                )
                .with_primary_label("MIR lowering failed here")
                .with_category("mir-lowering")],
                parse_failure: None,
                lowering_failure: Some(error),
                compile_failure: None,
                bytecode: None,
                resolution: Vec::new(),
                project_revision: source_catalog.and_then(|catalog| catalog.project_revision()),
                domains: AnalysisDomains::unavailable_after_hir(),
            };
        }
    };

    let facts = runmat_mir::analysis::analyze_assembly(&mir);
    let mut diagnostics = facts
        .diagnostics
        .iter()
        // Environment effects are facts. They become diagnostics below only
        // when they make a concrete subsequent call impossible to resolve.
        .filter(|diagnostic| diagnostic.code != "RM-MIR0009")
        .cloned()
        .collect::<Vec<_>>();
    let environment_effects = environment_effects(&mir);
    let resolution = call_resolution_diagnostics(&lowering, &environment_effects, source_catalog);
    let runtime_dependent = resolution.runtime_dependent;
    diagnostics.extend(resolution.diagnostics);

    let compiled = compile_lowering(&lowering, &mir);
    let compile_failure = compiled.as_ref().err().cloned();
    if let Some(error) = &compile_failure {
        diagnostics.push(
            HirDiagnostic::new(
                error
                    .identifier
                    .clone()
                    .unwrap_or_else(|| "RunMat:CompileError".to_string()),
                HirDiagnosticSeverity::Error,
                error.message.clone(),
                error.span.unwrap_or(Span { start: 0, end: 0 }),
            )
            .with_primary_label("compilation failed here")
            .with_category("compile"),
        );
    }

    diagnostics.sort_by_key(|diagnostic| {
        (
            diagnostic.primary.span.start,
            diagnostic.primary.span.end,
            diagnostic.code.clone(),
        )
    });
    diagnostics.dedup_by(|left, right| {
        left.code == right.code
            && left.primary.span == right.primary.span
            && left.message == right.message
    });

    let mut domains = AnalysisDomains::completed_frontend();
    if runtime_dependent {
        domains.name_resolution = AnalysisCompleteness::RuntimeDependent;
    }
    let semantic_facts = crate::semantic::project_document_facts(&lowering, &mir, &facts);
    FrontendAnalysis {
        lowering: Some(lowering),
        mir: Some(mir),
        facts: Some(facts),
        semantic_facts: Some(semantic_facts),
        diagnostics,
        parse_failure: None,
        lowering_failure: None,
        compile_failure,
        bytecode: compiled.ok(),
        resolution: resolution.evidence,
        project_revision: source_catalog.and_then(|catalog| catalog.project_revision()),
        domains,
    }
}

fn compile_lowering(
    lowering: &LoweringResult,
    mir: &MirAssembly,
) -> Result<runmat_vm::Bytecode, CompileError> {
    let Some(entrypoint) = lowering.assembly.entrypoints.first() else {
        let bound_functions =
            runmat_vm::compile_semantic_function_registry(&lowering.assembly, mir)?;
        let function_registry = runmat_vm::FunctionRegistry::new(bound_functions.clone());
        let mut bytecode = runmat_vm::Bytecode::empty();
        bytecode.bound_functions = bound_functions;
        bytecode.function_registry = function_registry;
        return Ok(bytecode);
    };
    runmat_vm::compile(&lowering.assembly, mir, entrypoint.id)
}

fn parse_failure_message(failure: &ParseFailure) -> String {
    let mut message = failure.message.clone();
    if let Some(expected) = &failure.expected {
        message.push_str(&format!("; expected {expected}"));
    }
    if let Some(found) = &failure.found_token {
        message.push_str(&format!("; found `{found}`"));
    }
    message
}

fn source_position_span(source: &str, position: usize) -> Span {
    let start = position.min(source.len());
    let end = source[start..]
        .chars()
        .next()
        .map(|character| start + character.len_utf8())
        .unwrap_or(start);
    Span { start, end }
}

#[derive(Debug, Clone)]
struct EnvironmentEffect {
    span: Span,
    effect: HirEnvironmentEffect,
}

fn environment_effects(mir: &MirAssembly) -> Vec<EnvironmentEffect> {
    let mut effects = Vec::new();
    for body in mir.bodies.values() {
        for block in &body.blocks {
            for statement in &block.statements {
                if let MirStmtKind::EnvironmentEffect(effect) = &statement.kind {
                    effects.push(EnvironmentEffect {
                        span: statement.span,
                        effect: effect.clone(),
                    });
                }
            }
        }
    }
    effects.sort_by_key(|effect| effect.span.start);
    effects
}

struct ResolutionDiagnostics {
    diagnostics: Vec<HirDiagnostic>,
    evidence: Vec<ResolutionEvidence>,
    runtime_dependent: bool,
}

fn call_resolution_diagnostics(
    lowering: &LoweringResult,
    environment_effects: &[EnvironmentEffect],
    source_catalog: Option<&runmat_package::DiscoveredSourceSymbols>,
) -> ResolutionDiagnostics {
    let mut diagnostics = Vec::new();
    let mut evidence = Vec::new();
    let mut runtime_dependent = false;
    for call in &lowering.hir_index.calls {
        let name = call
            .name
            .display_name()
            .unwrap_or_else(|| "<dynamic call>".to_string());
        if !matches!(call.callee, HirCallableRef::Unresolved(_)) {
            if let Some(definition) = catalog_definition(source_catalog, &name) {
                evidence.push(ResolutionEvidence {
                    name,
                    span: call.span,
                    state: ResolutionState::Resolved,
                    reason: "matched a statically indexed project source".to_string(),
                    definition: Some(definition.clone()),
                });
            }
            continue;
        }
        if !matches!(call.kind, CallKind::Dynamic)
            || !matches!(call.callee, HirCallableRef::Unresolved(_))
        {
            continue;
        }
        if matches!(call.syntax, CallSyntax::Method | CallSyntax::DottedInvoke) {
            runtime_dependent = true;
            evidence.push(ResolutionEvidence {
                name,
                span: call.span,
                state: ResolutionState::RuntimeDependent,
                reason: "method dispatch depends on the receiver's runtime class".to_string(),
                definition: None,
            });
            diagnostics.push(
                HirDiagnostic::new(
                    DIAGNOSTIC_RUNTIME_METHOD_DISPATCH,
                    HirDiagnosticSeverity::Warning,
                    "cannot determine which function is called here",
                    call.span,
                )
                .with_primary_label("the call target is selected at runtime")
                .with_note(
                    "method dispatch depends on the runtime class of the receiver expression",
                )
                .with_category("call-resolution"),
            );
            continue;
        }
        let causal_effect = environment_effects
            .iter()
            .rev()
            .find(|effect| effect.span.start < call.span.start);
        if let Some(effect) = causal_effect {
            runtime_dependent = true;
            evidence.push(ResolutionEvidence {
                name: name.clone(),
                span: call.span,
                state: ResolutionState::RuntimeDependent,
                reason: "a preceding statement changes the runtime function lookup environment"
                    .to_string(),
                definition: None,
            });
            let effect_label = match effect.effect {
                HirEnvironmentEffect::PathMutation => {
                    "this changes where subsequent functions are loaded from"
                }
                HirEnvironmentEffect::WorkingDirectoryMutation => {
                    "this changes the base directory used for subsequent lookup"
                }
                HirEnvironmentEffect::FunctionCacheInvalidation => {
                    "this invalidates cached function lookup"
                }
                HirEnvironmentEffect::DynamicLookupInvalidation => {
                    "this changes subsequent dynamic lookup behavior"
                }
            };
            diagnostics.push(
                HirDiagnostic::new(
                    DIAGNOSTIC_RUNTIME_DEPENDENT_RESOLUTION,
                    HirDiagnosticSeverity::Warning,
                    format!("cannot resolve `{name}` after a runtime environment change"),
                    call.span,
                )
                .with_primary_label(format!(
                    "RunMat cannot determine which `{name}.m` will be used"
                ))
                .with_secondary(effect.span, effect_label)
                .with_note(
                    "all statically known source roots were checked before classifying this call",
                )
                .with_category("call-resolution"),
            );
        } else {
            evidence.push(ResolutionEvidence {
                name: name.clone(),
                span: call.span,
                state: ResolutionState::Unresolved,
                reason: "no matching builtin, local, imported, or indexed project source was found"
                    .to_string(),
                definition: None,
            });
            diagnostics.push(
                HirDiagnostic::new(
                    DIAGNOSTIC_UNRESOLVED_FUNCTION,
                    HirDiagnosticSeverity::Warning,
                    format!("cannot find function `{name}`"),
                    call.span,
                )
                .with_primary_label("not defined in this file or project")
                .with_note(
                    "RunMat checked built-ins, local functions, imports, and every configured source root",
                )
                .with_help(format!(
                    "place `{name}.m` beside this source or in a source root configured by `runmat.toml`"
                ))
                .with_category("call-resolution"),
            );
        }
    }
    for handle in &lowering.hir_index.function_handles {
        let FunctionHandleTarget::DynamicName(_) = &handle.target else {
            continue;
        };
        let name = handle
            .name
            .display_name()
            .unwrap_or_else(|| "<dynamic function handle>".to_string());
        if let Some(effect) = environment_effects
            .iter()
            .rev()
            .find(|effect| effect.span.start < handle.span.start)
        {
            runtime_dependent = true;
            evidence.push(ResolutionEvidence {
                name: name.clone(),
                span: handle.span,
                state: ResolutionState::RuntimeDependent,
                reason: "a preceding statement changes the runtime function lookup environment"
                    .to_string(),
                definition: None,
            });
            diagnostics.push(
                HirDiagnostic::new(
                    DIAGNOSTIC_RUNTIME_DEPENDENT_RESOLUTION,
                    HirDiagnosticSeverity::Warning,
                    format!("cannot resolve function handle `@{name}` after a runtime environment change"),
                    handle.span,
                )
                .with_primary_label(format!(
                    "RunMat cannot determine which `{name}.m` this handle will reference"
                ))
                .with_secondary(
                    effect.span,
                    "this changes the function lookup environment used by the handle",
                )
                .with_category("call-resolution"),
            );
        } else {
            evidence.push(ResolutionEvidence {
                name: name.clone(),
                span: handle.span,
                state: ResolutionState::Unresolved,
                reason: "no matching builtin, local, imported, or indexed project source was found"
                    .to_string(),
                definition: None,
            });
            diagnostics.push(
                HirDiagnostic::new(
                    DIAGNOSTIC_UNRESOLVED_FUNCTION,
                    HirDiagnosticSeverity::Warning,
                    format!("cannot find function referenced by `@{name}`"),
                    handle.span,
                )
                .with_primary_label("the named function is not defined in this file or project")
                .with_note(
                    "RunMat checked built-ins, local functions, imports, and every configured source root",
                )
                .with_help(format!(
                    "place `{name}.m` beside this source or in a source root configured by `runmat.toml`"
                ))
                .with_category("call-resolution"),
            );
        }
    }
    ResolutionDiagnostics {
        diagnostics,
        evidence,
        runtime_dependent,
    }
}

fn catalog_definition<'a>(
    catalog: Option<&'a runmat_package::DiscoveredSourceSymbols>,
    name: &str,
) -> Option<&'a runmat_package::ProjectSymbolDefinition> {
    catalog?
        .definitions
        .iter()
        .find(|definition| definition.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::PathBuf;

    fn analyze(source: &str) -> FrontendAnalysis {
        analyze_source(source, CompatMode::default(), &LoweringContext::empty())
    }

    #[test]
    fn reports_parse_failures_as_structured_errors() {
        let analysis = analyze("x = ;");
        assert!(analysis.has_errors());
        assert_eq!(analysis.diagnostics[0].code, "RunMat:ParseError");
        assert_eq!(
            analysis.domains.name_resolution,
            AnalysisCompleteness::Unavailable
        );
    }

    #[test]
    fn reports_unresolved_bare_calls_without_rejecting_compilation() {
        let analysis = analyze("x = definitely_missing(42);");
        assert!(analysis.compile_failure.is_none());
        assert!(analysis
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_UNRESOLVED_FUNCTION));
    }

    #[test]
    fn resolves_known_project_functions() {
        let symbols = HashSet::from(["helper".to_string()]);
        let context = LoweringContext::empty().with_known_project_symbols(&symbols);
        let analysis = analyze_source("x = helper(42);", CompatMode::default(), &context);
        assert!(!analysis
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_UNRESOLVED_FUNCTION));
    }

    #[test]
    fn preserves_the_project_definition_that_justified_resolution() {
        let symbols = HashSet::from(["helper".to_string()]);
        let package_instance = runmat_package::ContentDigest::sha256("package");
        let source_id = runmat_package::StableSourceId {
            package_instance: package_instance.clone(),
            relative_path: runmat_package::NormalizedRelativePath::new("src/helper.m").unwrap(),
            content_digest: runmat_package::ContentDigest::sha256("function y = helper();"),
        };
        let graph_digest = runmat_package::ContentDigest::sha256("graph");
        let source_revision = runmat_package::ContentDigest::sha256("sources");
        let definition = runmat_package::ProjectSymbolDefinition {
            name: "helper".to_string(),
            qualified_name: "helper".to_string(),
            source_path: PathBuf::from("src/helper.m"),
            package_name: "demo".to_string(),
            dependency_alias: None,
            package_instance: Some(package_instance),
            source_id: Some(source_id),
            is_private: false,
        };
        let catalog = runmat_package::DiscoveredSourceSymbols {
            manifest_path: Some(PathBuf::from("runmat.toml")),
            project_root: PathBuf::from("."),
            graph_digest: Some(graph_digest.clone()),
            source_revision: Some(source_revision.clone()),
            symbols: symbols.clone(),
            definitions: vec![definition.clone()],
        };
        let context = LoweringContext::empty().with_known_project_symbols(&symbols);
        let analysis = analyze_source_with_catalog(
            "x = helper(42);",
            CompatMode::default(),
            &context,
            Some(&catalog),
        );
        assert_eq!(
            analysis.resolution,
            vec![ResolutionEvidence {
                name: "helper".to_string(),
                span: analysis.resolution[0].span,
                state: ResolutionState::Resolved,
                reason: "matched a statically indexed project source".to_string(),
                definition: Some(definition),
            }]
        );
        assert_eq!(
            analysis.project_revision,
            Some(runmat_package::ProjectRevision {
                graph_digest,
                source_revision,
            })
        );
    }

    #[test]
    fn includes_canonical_mir_fact_diagnostics() {
        let analysis = analyze("a = ones(2,3); b = ones(4,2); c = a * b;");
        assert!(analysis
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-TYPE-MATMUL"));
    }

    #[test]
    fn distinguishes_runtime_path_mutation_from_an_unresolved_static_call() {
        let analysis = analyze("addpath('plugins'); value = selected_at_runtime(1);");
        assert!(analysis
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_RUNTIME_DEPENDENT_RESOLUTION));
        assert_eq!(
            analysis.domains.name_resolution,
            AnalysisCompleteness::RuntimeDependent
        );
    }

    #[test]
    fn read_only_which_does_not_replace_the_causal_path_mutation() {
        let analysis = analyze(
            "addpath('plugins'); located = which('selected_at_runtime'); value = selected_at_runtime(1);",
        );
        let diagnostic = analysis
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code == DIAGNOSTIC_RUNTIME_DEPENDENT_RESOLUTION)
            .expect("runtime-dependent resolution diagnostic");
        assert!(diagnostic.secondary.iter().any(|secondary| {
            secondary
                .label
                .as_deref()
                .is_some_and(|label| label.contains("where subsequent functions are loaded"))
        }));
    }

    #[test]
    fn a_later_path_mutation_does_not_explain_an_earlier_missing_call() {
        let analysis = analyze("value = still_missing(1); addpath('plugins');");
        assert!(analysis
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_UNRESOLVED_FUNCTION));
        assert!(!analysis
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_RUNTIME_DEPENDENT_RESOLUTION));
    }

    #[test]
    fn unresolved_method_dispatch_is_reported_as_runtime_dependent() {
        let analysis = analyze("receiver = struct(); value = receiver.compute(1);");
        assert!(analysis
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_RUNTIME_METHOD_DISPATCH));
    }

    #[test]
    fn builtins_do_not_produce_missing_function_diagnostics() {
        let analysis = analyze("value = sin(1);");
        assert!(!analysis
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.category.as_deref() == Some("call-resolution")));
    }

    #[test]
    fn semantic_projection_answers_reassignment_at_the_requested_source_position() {
        let source = "x = 1; before = x; x = 'changed'; after = x;";
        let analysis = analyze(source);
        let facts = analysis.semantic_facts.as_ref().expect("semantic facts");

        let before_offset = source.find("before = x").unwrap() + "before = ".len();
        let after_offset = source.find("after = x").unwrap() + "after = ".len();
        let before = facts
            .quick_information("x", before_offset)
            .and_then(|information| information.observation)
            .and_then(|observation| observation.fact)
            .expect("fact before reassignment");
        let after = facts
            .quick_information("x", after_offset)
            .and_then(|information| information.observation)
            .and_then(|observation| observation.fact)
            .expect("fact after reassignment");

        assert!(matches!(
            before.kind,
            runmat_types::ValueKindFact::Numeric(_)
        ));
        assert_eq!(after.kind, runmat_types::ValueKindFact::Character);
    }

    #[test]
    fn semantic_projection_exposes_the_safe_control_flow_join() {
        let source = "c = true; x = 1; if c; x = false; end; y = x;";
        let analysis = analyze(source);
        let facts = analysis.semantic_facts.as_ref().expect("semantic facts");
        let offset = source.find("y = x").unwrap() + "y = ".len();
        let joined = facts
            .quick_information("x", offset)
            .and_then(|information| information.observation)
            .and_then(|observation| observation.fact)
            .expect("joined fact");

        assert_eq!(joined.kind, runmat_types::ValueKindFact::Unknown);
        assert_eq!(joined.shape, runmat_types::ShapeFact::Scalar);
        assert_eq!(
            joined.certainty,
            runmat_types::CertaintyFact::Dynamic(
                runmat_types::DynamicReason::ConflictingControlFlow
            )
        );
    }

    #[test]
    fn semantic_projection_is_deterministic_strict_and_round_trips() {
        let analysis = analyze("x = linspace(-1, 1, 5); y = sin(x);");
        let facts = analysis.semantic_facts.as_ref().expect("semantic facts");
        let first = serde_json::to_string(facts).expect("serialize facts");
        let second = serde_json::to_string(facts).expect("serialize facts again");
        assert_eq!(first, second);
        let round_trip: crate::semantic::SemanticDocumentFacts =
            serde_json::from_str(&first).expect("round-trip facts");
        assert_eq!(&round_trip, facts);
        round_trip.validate_current().expect("current revision");

        let mut incompatible_revision = round_trip.clone();
        incompatible_revision.revision.fact_schema_minor += 1;
        assert!(matches!(
            incompatible_revision.validate_current(),
            Err(runmat_mir::analysis::AnalysisRevisionMismatch::FactSchema { .. })
        ));

        let mut incompatible = serde_json::to_value(facts).expect("facts as JSON");
        incompatible
            .as_object_mut()
            .expect("document object")
            .insert("unknown_future_field".to_string(), serde_json::json!(true));
        assert!(
            serde_json::from_value::<crate::semantic::SemanticDocumentFacts>(incompatible).is_err()
        );
    }

    #[test]
    fn unresolved_function_handles_use_the_same_resolution_policy() {
        let missing = analyze("handle = @definitely_missing;");
        assert!(missing
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_UNRESOLVED_FUNCTION));

        let symbols = HashSet::from(["project.helper".to_string()]);
        let context = LoweringContext::empty().with_known_project_symbols(&symbols);
        let resolved = analyze_source("handle = @project.helper;", CompatMode::default(), &context);
        assert!(!resolved
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_UNRESOLVED_FUNCTION));
    }
}
