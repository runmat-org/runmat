use crate::core::docs;
use crate::core::position::{offset_to_position, position_to_offset};
use crate::core::project::ProjectContext;
use crate::core::semantic::model::{
    AnalysisModel, FunctionSemantic, TextRange, VariableKind, VariableSymbol,
};
use lsp_types::{
    CompletionItem, Diagnostic, DiagnosticSeverity, DocumentSymbol, Hover, Location, Position,
    Range, SignatureHelp, Url,
};
use runmat_builtins::{self, BuiltinCompletionPolicy, BuiltinFunction, Type};
use runmat_hir::{
    CallKind, DefPath, HirDiagnostic, HirDiagnosticSeverity, HirError, LoweringContext,
    LoweringResult,
};
use runmat_lexer::{tokenize_detailed, SpannedToken, Token};
pub use runmat_parser::CompatMode;
use runmat_vm::CompileError;
use std::fmt::Write;
use std::path::PathBuf;

#[derive(Clone)]
pub struct DocumentAnalysis {
    pub tokens: Vec<SpannedToken>,
    pub syntax_error: Option<SyntaxErrorInfo>,
    pub lowering_error: Option<HirError>,
    pub compile_error: Option<CompileError>,
    pub lowering: Option<LoweringResult>,
    pub semantic: Option<AnalysisModel>,
    /// Diagnostics produced by the canonical frontend, including failures
    /// before a semantic model can be constructed.
    pub diagnostics: Vec<HirDiagnostic>,
    /// Machine-readable name-resolution decisions, including the indexed
    /// project source that justified a successful resolution.
    pub resolution: Vec<runmat_static_analysis::frontend::ResolutionEvidence>,
    /// Immutable graph/source identity used for this analysis, when the
    /// document belongs to a package project.
    #[cfg_attr(not(test), allow(dead_code))]
    pub project_revision: Option<runmat_package::ProjectRevision>,
}

impl DocumentAnalysis {
    pub fn status_message(&self) -> String {
        if let Some(err) = &self.lowering_error {
            return format!("Lowering failed: {}", err.message);
        }
        if let Some(se) = &self.syntax_error {
            return format!("Syntax error: {}", se.message);
        }
        if let Some(err) = &self.compile_error {
            return format!("Compile error: {}", err.message);
        }
        if let Some(sem) = &self.semantic {
            return sem.status_message.clone();
        }
        "ok".to_string()
    }
}

#[derive(Clone)]
pub struct SyntaxErrorInfo {
    pub message: String,
    pub position: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn analyze_document_with_compat(text: &str, compat: CompatMode) -> DocumentAnalysis {
    analyze_document_with_compat_and_source(text, compat, None)
}

pub fn analyze_document_with_compat_and_source(
    text: &str,
    compat: CompatMode,
    source_name: Option<&str>,
) -> DocumentAnalysis {
    let tokens = tokenize_detailed(text);
    let (source_catalog, catalog_diagnostic) = discover_source_context(source_name);
    let known_project_symbols = source_catalog
        .as_ref()
        .map(|catalog| &catalog.symbols)
        .cloned()
        .unwrap_or_default();
    let mut lowering_context =
        LoweringContext::empty().with_runmat_extensions_enabled(compat.allows_runmat_extensions());
    if !known_project_symbols.is_empty() {
        lowering_context = lowering_context.with_known_project_symbols(&known_project_symbols);
    }
    let mut frontend = runmat_static_analysis::frontend::analyze_source_with_catalog(
        text,
        compat,
        &lowering_context,
        source_catalog.as_ref(),
    );
    if let Some(diagnostic) = catalog_diagnostic {
        frontend.diagnostics.push(diagnostic);
    }
    document_analysis_from_frontend(text, tokens, frontend)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub async fn analyze_document_with_compat_and_source_async(
    text: &str,
    compat: CompatMode,
    source_name: Option<&str>,
) -> DocumentAnalysis {
    let tokens = tokenize_detailed(text);
    let (source_catalog, catalog_diagnostic) = discover_source_context_async(source_name).await;
    let known_project_symbols = source_catalog
        .as_ref()
        .map(|catalog| &catalog.symbols)
        .cloned()
        .unwrap_or_default();
    let mut lowering_context =
        LoweringContext::empty().with_runmat_extensions_enabled(compat.allows_runmat_extensions());
    if !known_project_symbols.is_empty() {
        lowering_context = lowering_context.with_known_project_symbols(&known_project_symbols);
    }
    let mut frontend = runmat_static_analysis::frontend::analyze_source_with_catalog(
        text,
        compat,
        &lowering_context,
        source_catalog.as_ref(),
    );
    if let Some(diagnostic) = catalog_diagnostic {
        frontend.diagnostics.push(diagnostic);
    }
    document_analysis_from_frontend(text, tokens, frontend)
}

fn document_analysis_from_frontend(
    text: &str,
    tokens: Vec<SpannedToken>,
    frontend: runmat_static_analysis::frontend::FrontendAnalysis,
) -> DocumentAnalysis {
    let syntax_error = frontend
        .parse_failure
        .as_ref()
        .map(|failure| SyntaxErrorInfo {
            message: frontend
                .diagnostics
                .first()
                .map(|diagnostic| diagnostic.message.clone())
                .unwrap_or_else(|| failure.message.clone()),
            position: failure.position,
        });
    let semantic = frontend
        .lowering
        .as_ref()
        .map(|_| crate::core::semantic::model::build(&frontend, &tokens, text));
    let diagnostics = frontend.diagnostics.clone();
    let resolution = frontend.resolution.clone();
    let project_revision = frontend.project_revision.clone();
    DocumentAnalysis {
        tokens,
        syntax_error,
        lowering_error: frontend.lowering_failure,
        compile_error: frontend.compile_failure,
        lowering: frontend.lowering,
        semantic,
        diagnostics,
        resolution,
        project_revision,
    }
}

#[cfg(test)]
#[cfg(all(test, not(target_arch = "wasm32")))]
fn discover_known_project_symbols(source_name: Option<&str>) -> std::collections::HashSet<String> {
    discover_source_context(source_name)
        .0
        .map(|catalog| catalog.symbols)
        .unwrap_or_default()
}

fn discover_source_context(
    source_name: Option<&str>,
) -> (
    Option<runmat_package::DiscoveredSourceSymbols>,
    Option<HirDiagnostic>,
) {
    let cwd = source_name
        .and_then(|name| {
            let path = PathBuf::from(name);
            if path.is_absolute() {
                path.parent().map(|p| p.to_path_buf())
            } else {
                None
            }
        })
        .or_else(|| runmat_filesystem::current_dir().ok());

    let Some(cwd) = cwd else {
        return (None, None);
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        let Some(source_name) = source_name else {
            return (None, None);
        };
        match futures::executor::block_on(
            runmat_package::discover_source_symbols_from_source_name_async(source_name, &cwd),
        ) {
            Ok(Some(discovered)) => (Some(discovered), None),
            Ok(None) => (None, None),
            Err(error) => (None, Some(source_catalog_diagnostic(error.to_string()))),
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = source_name;
        let _ = cwd;
        (None, None)
    }
}

async fn discover_source_context_async(
    source_name: Option<&str>,
) -> (
    Option<runmat_package::DiscoveredSourceSymbols>,
    Option<HirDiagnostic>,
) {
    let cwd = source_name
        .and_then(|name| {
            let path = PathBuf::from(name);
            if path.is_absolute() {
                path.parent().map(|p| p.to_path_buf())
            } else {
                None
            }
        })
        .or_else(|| runmat_filesystem::current_dir().ok());

    let Some(cwd) = cwd else {
        return (None, None);
    };
    let Some(source_name) = source_name else {
        return (None, None);
    };
    match runmat_package::discover_source_symbols_from_source_name_async(source_name, &cwd).await {
        Ok(Some(discovered)) => (Some(discovered), None),
        Ok(None) => (None, None),
        Err(error) => (None, Some(source_catalog_diagnostic(error.to_string()))),
    }
}

fn source_catalog_diagnostic(message: String) -> HirDiagnostic {
    HirDiagnostic::new(
        runmat_static_analysis::frontend::DIAGNOSTIC_SOURCE_CATALOG,
        HirDiagnosticSeverity::Error,
        message,
        runmat_hir::Span { start: 0, end: 0 },
    )
    .with_primary_label("source lookup could not be constructed")
    .with_category("source-catalog")
}

pub fn diagnostics_for_document(text: &str, analysis: &DocumentAnalysis) -> Vec<Diagnostic> {
    if !analysis.diagnostics.is_empty() {
        return analysis
            .diagnostics
            .iter()
            .map(|diagnostic| {
                let mut rendered = diagnostic_for_hir_lint(diagnostic, text);
                rendered.data = analysis
                    .resolution
                    .iter()
                    .find(|evidence| evidence.span == diagnostic.primary.span)
                    .and_then(|evidence| serde_json::to_value(evidence).ok());
                rendered
            })
            .collect();
    }
    if let Some(syntax_err) = &analysis.syntax_error {
        return vec![Diagnostic {
            range: diagnostic_range_for_parse_error(syntax_err, &analysis.tokens, text),
            severity: Some(DiagnosticSeverity::ERROR),
            code: None,
            code_description: None,
            source: Some("runmat-parser".into()),
            message: syntax_err.message.clone(),
            related_information: None,
            tags: None,
            data: None,
        }];
    }
    if let Some(lowering_err) = &analysis.lowering_error {
        return vec![diagnostic_for_lowering_error(
            lowering_err,
            &analysis.tokens,
            text,
        )];
    }
    if let Some(compile_err) = &analysis.compile_error {
        return vec![diagnostic_for_compile_error(compile_err, text)];
    }
    if let Some(semantic) = &analysis.semantic {
        let mut diags: Vec<Diagnostic> = semantic
            .diagnostics
            .iter()
            .map(|diag| diagnostic_for_hir_lint(diag, text))
            .collect();
        if !semantic.status_message.is_empty() {
            diags.push(Diagnostic {
                range: Range {
                    start: Position::new(0, 0),
                    end: Position::new(0, 1),
                },
                severity: Some(DiagnosticSeverity::INFORMATION),
                code: None,
                code_description: None,
                source: Some("runmat-semantic".into()),
                message: semantic.status_message.clone(),
                related_information: None,
                tags: None,
                data: None,
            });
        }
        return diags;
    }
    Vec::new()
}

pub fn completion_at(
    text: &str,
    analysis: &DocumentAnalysis,
    position: &Position,
) -> Vec<CompletionItem> {
    if let Some(semantic) = &analysis.semantic {
        completion_from_semantic(semantic, position_to_offset(text, position))
    } else {
        Vec::new()
    }
}

pub fn hover_at(text: &str, analysis: &DocumentAnalysis, position: &Position) -> Option<Hover> {
    let offset = position_to_offset(text, position);
    let token = token_at_offset(&analysis.tokens, offset)?;
    let ident = token.lexeme.clone();

    if let Some(semantic) = analysis.semantic.as_ref() {
        if let Some(func) = semantic.function_at_offset(offset) {
            if let Some(var) = func.variables.get(&ident) {
                let fact = semantic
                    .facts
                    .as_ref()
                    .and_then(|facts| facts.fact_at(var.binding, offset))
                    .and_then(|observation| observation.fact.as_ref());
                return Some(Hover {
                    contents: lsp_types::HoverContents::Markup(lsp_types::MarkupContent {
                        kind: lsp_types::MarkupKind::Markdown,
                        value: crate::core::semantic::presentation::format_variable_hover(
                            &ident,
                            var.kind.as_label(),
                            fact,
                            false,
                        ),
                    }),
                    range: Some(
                        TextRange {
                            start: token.start,
                            end: token.end,
                        }
                        .to_lsp_range(text),
                    ),
                });
            }
        }

        if let Some(glob) = semantic.globals.get(&ident) {
            let fact = semantic
                .facts
                .as_ref()
                .and_then(|facts| facts.fact_at(glob.binding, offset))
                .and_then(|observation| observation.fact.as_ref());
            return Some(Hover {
                contents: lsp_types::HoverContents::Markup(lsp_types::MarkupContent {
                    kind: lsp_types::MarkupKind::Markdown,
                    value: crate::core::semantic::presentation::format_variable_hover(
                        &ident,
                        glob.kind.as_label(),
                        fact,
                        true,
                    ),
                }),
                range: Some(
                    TextRange {
                        start: token.start,
                        end: token.end,
                    }
                    .to_lsp_range(text),
                ),
            });
        }
    }

    // Canonical catalog entries are executor-neutral and remain available even
    // when no runtime implementation is linked into the language server.
    let call_site = super::signatures::CallSite::at(analysis, token.start)
        .filter(|site| site.callee_range.contains(&token.start));
    let builtin_name = call_site
        .as_ref()
        .map_or(ident.as_str(), |site| site.name.as_str());
    if let Some(entry) = runmat_builtins::public_builtin_catalog_entry_by_name(builtin_name) {
        let selected = call_site
            .as_ref()
            .map(|site| site.best(entry.descriptor.signatures))
            .unwrap_or_default();
        return Some(Hover {
            contents: lsp_types::HoverContents::Markup(lsp_types::MarkupContent {
                kind: lsp_types::MarkupKind::Markdown,
                value: docs::build_catalog_hover_with_signatures(entry, &selected),
            }),
            range: Some(
                TextRange {
                    start: token.start,
                    end: token.end,
                }
                .to_lsp_range(text),
            ),
        });
    }

    // Legacy built-in functions (rich docs)
    if let Some(func) = runmat_builtins::builtin_functions()
        .into_iter()
        .find(|f| f.name.eq_ignore_ascii_case(builtin_name))
    {
        return Some(Hover {
            contents: lsp_types::HoverContents::Markup(lsp_types::MarkupContent {
                kind: lsp_types::MarkupKind::Markdown,
                value: docs::build_builtin_hover(func),
            }),
            range: Some(
                TextRange {
                    start: token.start,
                    end: token.end,
                }
                .to_lsp_range(text),
            ),
        });
    }

    // Built-in constants
    if let Some(constant) = runmat_builtins::constants()
        .into_iter()
        .find(|c| c.name.eq_ignore_ascii_case(&ident))
    {
        let mut buf = String::new();
        let _ = writeln!(buf, "```runmat\nconst {name}\n```", name = constant.name);
        let _ = writeln!(buf, "\nValue: `{val:?}`", val = constant.value);
        return Some(Hover {
            contents: lsp_types::HoverContents::Markup(lsp_types::MarkupContent {
                kind: lsp_types::MarkupKind::Markdown,
                value: buf,
            }),
            range: Some(
                TextRange {
                    start: token.start,
                    end: token.end,
                }
                .to_lsp_range(text),
            ),
        });
    }

    None
}

/// Presentation-neutral semantic information at a source position. This is
/// the structured counterpart to hover and is shared by native and WASM
/// transports.
pub fn quick_information_at(
    text: &str,
    analysis: &DocumentAnalysis,
    position: &Position,
) -> Option<runmat_static_analysis::semantic::SemanticQuickInformation> {
    let offset = position_to_offset(text, position);
    let token = token_at_offset(&analysis.tokens, offset)?;
    analysis
        .semantic
        .as_ref()?
        .facts
        .as_ref()?
        .quick_information(&token.lexeme, offset)
}

pub fn semantic_document_facts(
    analysis: &DocumentAnalysis,
) -> Option<&runmat_static_analysis::semantic::SemanticDocumentFacts> {
    analysis.semantic.as_ref()?.facts.as_ref()
}

pub fn definition_at(
    _text: &str,
    _analysis: &DocumentAnalysis,
    _position: &Position,
) -> Vec<Range> {
    let semantic = match &_analysis.semantic {
        Some(s) => s,
        None => return vec![],
    };
    let offset = position_to_offset(_text, _position);
    let Some(tok) = token_at_offset(&_analysis.tokens, offset) else {
        return vec![];
    };
    let mut ranges = Vec::new();

    if let Some(funcs) = semantic.function_lookup.get(&tok.lexeme) {
        for idx in funcs {
            if let Some(f) = semantic.functions.get(*idx) {
                ranges.push(f.signature.name_range.to_lsp_range(_text));
            }
        }
    }

    if let Some(func) = semantic.function_at_offset(offset) {
        if let Some(var) = func.variables.get(&tok.lexeme) {
            ranges.push(
                find_symbol_range(&_analysis.tokens, &tok.lexeme, Some(&func.range))
                    .unwrap_or(TextRange {
                        start: tok.start,
                        end: tok.end,
                    })
                    .to_lsp_range(_text),
            );
            if matches!(var.kind, VariableKind::Parameter | VariableKind::Output) {
                ranges.push(func.signature.name_range.to_lsp_range(_text));
            }
        }
    }

    if semantic.globals.contains_key(&tok.lexeme) {
        let range = find_symbol_range(&_analysis.tokens, &tok.lexeme, None).unwrap_or(TextRange {
            start: tok.start,
            end: tok.end,
        });
        ranges.push(range.to_lsp_range(_text));
    }

    ranges
}

pub fn definition_locations_at(
    text: &str,
    analysis: &DocumentAnalysis,
    position: &Position,
    uri: &Url,
) -> Vec<Location> {
    let mut locations = definition_at(text, analysis, position)
        .into_iter()
        .map(|range| Location {
            uri: uri.clone(),
            range,
        })
        .collect::<Vec<_>>();

    if let Some(symbol) = symbol_name_at(text, analysis, position) {
        if !symbol.is_local {
            if let Some(source_name) = uri_file_source_name(uri) {
                if let Some(project) = ProjectContext::discover_from_source_name(Some(&source_name))
                {
                    locations.extend(project_function_definitions(
                        &project,
                        &symbol.name,
                        CompatMode::RunMat,
                    ));
                }
            }
        }
    }
    dedupe_locations(&mut locations);
    locations
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub async fn definition_locations_at_async(
    text: &str,
    analysis: &DocumentAnalysis,
    position: &Position,
    uri: &Url,
) -> Vec<Location> {
    let mut locations = definition_at(text, analysis, position)
        .into_iter()
        .map(|range| Location {
            uri: uri.clone(),
            range,
        })
        .collect::<Vec<_>>();
    if let Some(symbol) = symbol_name_at(text, analysis, position) {
        if !symbol.is_local {
            if let Some(source_name) = uri_file_source_name(uri) {
                if let Some(project) =
                    ProjectContext::discover_from_source_name_async(Some(&source_name)).await
                {
                    locations.extend(
                        project_function_definitions_async(
                            &project,
                            &symbol.name,
                            CompatMode::RunMat,
                        )
                        .await,
                    );
                }
            }
        }
    }
    dedupe_locations(&mut locations);
    locations
}

pub fn references_locations_at(
    text: &str,
    analysis: &DocumentAnalysis,
    position: &Position,
    uri: &Url,
) -> Vec<Location> {
    let Some(symbol) = symbol_name_at(text, analysis, position) else {
        return Vec::new();
    };
    if symbol.is_local {
        return local_symbol_references(text, analysis, position, uri, &symbol.name);
    }
    let mut locations = function_references_in_document(text, analysis, &symbol.name)
        .into_iter()
        .map(|range| Location {
            uri: uri.clone(),
            range,
        })
        .collect::<Vec<_>>();
    locations.extend(
        function_definitions_in_document(text, analysis, &symbol.name)
            .into_iter()
            .map(|range| Location {
                uri: uri.clone(),
                range,
            }),
    );
    if let Some(source_name) = uri_file_source_name(uri) {
        if let Some(project) = ProjectContext::discover_from_source_name(Some(&source_name)) {
            locations.extend(project_function_references(
                &project,
                &symbol.name,
                CompatMode::RunMat,
            ));
            locations.extend(project_function_definitions(
                &project,
                &symbol.name,
                CompatMode::RunMat,
            ));
        }
    }
    dedupe_locations(&mut locations);
    locations
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct RenameSubject {
    pub name: String,
    pub is_local: bool,
    pub parameter_owner: Option<String>,
    pub range: Range,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn rename_subject_at(
    text: &str,
    analysis: &DocumentAnalysis,
    position: &Position,
) -> Option<RenameSubject> {
    let symbol = symbol_name_at(text, analysis, position)?;
    let offset = position_to_offset(text, position);
    let token = token_at_offset(&analysis.tokens, offset)?;
    let parameter_owner = analysis.semantic.as_ref().and_then(|semantic| {
        let function = semantic.function_at_offset(offset)?;
        function
            .variables
            .get(&symbol.name)
            .is_some_and(|variable| variable.kind == VariableKind::Parameter)
            .then(|| function.name.clone())
    });
    Some(RenameSubject {
        name: symbol.name,
        is_local: symbol.is_local,
        parameter_owner,
        range: TextRange {
            start: token.start,
            end: token.end,
        }
        .to_lsp_range(text),
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn rename_subject_has_collision(
    text: &str,
    analysis: &DocumentAnalysis,
    position: &Position,
    new_name: &str,
) -> bool {
    let Some(subject) = symbol_name_at(text, analysis, position) else {
        return true;
    };
    if !subject.is_local || subject.name == new_name {
        return false;
    }
    let offset = position_to_offset(text, position);
    let Some(semantic) = analysis.semantic.as_ref() else {
        return true;
    };
    if let Some(function) = semantic.function_at_offset(offset) {
        return function.variables.contains_key(new_name);
    }
    semantic.globals.contains_key(new_name)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub async fn references_locations_at_async(
    text: &str,
    analysis: &DocumentAnalysis,
    position: &Position,
    uri: &Url,
) -> Vec<Location> {
    let Some(symbol) = symbol_name_at(text, analysis, position) else {
        return Vec::new();
    };
    if symbol.is_local {
        return local_symbol_references(text, analysis, position, uri, &symbol.name);
    }
    let mut locations = function_references_in_document(text, analysis, &symbol.name)
        .into_iter()
        .map(|range| Location {
            uri: uri.clone(),
            range,
        })
        .collect::<Vec<_>>();
    locations.extend(
        function_definitions_in_document(text, analysis, &symbol.name)
            .into_iter()
            .map(|range| Location {
                uri: uri.clone(),
                range,
            }),
    );
    if let Some(source_name) = uri_file_source_name(uri) {
        if let Some(project) =
            ProjectContext::discover_from_source_name_async(Some(&source_name)).await
        {
            locations.extend(
                project_function_references_async(&project, &symbol.name, CompatMode::RunMat).await,
            );
            locations.extend(
                project_function_definitions_async(&project, &symbol.name, CompatMode::RunMat)
                    .await,
            );
        }
    }
    dedupe_locations(&mut locations);
    locations
}

pub fn signature_help_at(
    _text: &str,
    _analysis: &DocumentAnalysis,
    _position: &Position,
) -> Option<SignatureHelp> {
    let offset = position_to_offset(_text, _position);
    let site = super::signatures::CallSite::at(_analysis, offset);
    let name = site
        .as_ref()
        .map(|site| site.name.clone())
        .or_else(|| qualified_call_name_at_offset(&_analysis.tokens, offset))?;
    if let Some(entry) = runmat_builtins::public_builtin_catalog_entry_by_name(&name) {
        if !entry.descriptor.signatures.is_empty() {
            return Some(super::signatures::help(
                entry.descriptor.signatures,
                site.as_ref(),
            ));
        }
    }
    if let Some(descriptor) = runmat_builtins::builtin_functions()
        .into_iter()
        .find(|builtin| builtin.name.eq_ignore_ascii_case(&name))
        .and_then(|builtin| builtin.descriptor)
        .filter(|descriptor| !descriptor.signatures.is_empty())
    {
        return Some(super::signatures::help(
            descriptor.signatures,
            site.as_ref(),
        ));
    }

    let semantic = _analysis.semantic.as_ref()?;
    let funcs = semantic.function_lookup.get(&name)?;

    let mut sigs = Vec::new();
    for idx in funcs {
        if let Some(f) = semantic.functions.get(*idx) {
            let documentation = semantic
                .facts
                .as_ref()
                .and_then(|facts| {
                    facts
                        .functions
                        .iter()
                        .find(|facts| facts.function == f.function)
                })
                .and_then(|facts| facts.analysis.as_ref())
                .map(|analysis| {
                    let parameters = analysis
                        .callable
                        .parameters
                        .iter()
                        .map(|fact| crate::core::semantic::presentation::format_fact(Some(fact)))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let outputs = analysis
                        .callable
                        .outputs
                        .iter()
                        .map(|fact| crate::core::semantic::presentation::format_fact(Some(fact)))
                        .collect::<Vec<_>>()
                        .join(", ");
                    lsp_types::Documentation::MarkupContent(lsp_types::MarkupContent {
                        kind: lsp_types::MarkupKind::Markdown,
                        value: format!("Parameters: `{parameters}`  \nOutputs: `{outputs}`"),
                    })
                });
            sigs.push(lsp_types::SignatureInformation {
                label: f.signature.display(),
                documentation,
                parameters: Some(
                    f.signature
                        .inputs
                        .iter()
                        .map(|p| lsp_types::ParameterInformation {
                            label: lsp_types::ParameterLabel::Simple(p.clone()),
                            documentation: None,
                        })
                        .collect(),
                ),
                active_parameter: None,
            });
        }
    }

    Some(SignatureHelp {
        signatures: sigs,
        active_signature: Some(0),
        active_parameter: site.as_ref().map(|site| site.active_argument as u32),
    })
}

/// The `.m` outline: `%%` sections, script variables with their inferred types, functions with
/// inputs and outputs, and classes ([`crate::core::semantic::outline`]).
pub fn document_symbols(text: &str, analysis: &DocumentAnalysis) -> Vec<DocumentSymbol> {
    crate::core::semantic::outline::document_outline(
        text,
        analysis.lowering.as_ref(),
        analysis.semantic.as_ref(),
    )
}

pub(crate) fn make_document_symbol(
    name: String,
    detail: Option<String>,
    kind: lsp_types::SymbolKind,
    range: Range,
    selection_range: Range,
) -> DocumentSymbol {
    // lsp-types 0.94 requires this deprecated wire field in the struct literal;
    // omitting it is not possible until the shared protocol dependency advances.
    #[allow(deprecated)]
    DocumentSymbol {
        name,
        detail,
        kind,
        tags: None,
        deprecated: None,
        range,
        selection_range,
        children: None,
    }
}

pub fn semantic_tokens_legend() -> lsp_types::SemanticTokensLegend {
    crate::core::semantic_tokens::legend()
}

pub fn semantic_tokens_full(
    text: &str,
    analysis: &DocumentAnalysis,
) -> Option<lsp_types::SemanticTokens> {
    let hints = analysis
        .semantic
        .as_ref()
        .map(|semantic| semantic.token_hints.as_slice())
        .unwrap_or(&[]);
    crate::core::semantic_tokens::full(text, &analysis.tokens, hints)
}

/// Build the complete lexical semantic-token baseline for a document.
///
/// This deliberately uses the same lexer, legend, and encoder as the enriched
/// semantic-token path. Callers can therefore render correct comments,
/// keywords, strings, numbers, operators, and identifiers before parsing/HIR
/// analysis has completed, without depending on an editor-side grammar.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub fn semantic_tokens_lexical(text: &str) -> Option<lsp_types::SemanticTokens> {
    let tokens = tokenize_detailed(text);
    crate::core::semantic_tokens::full(text, &tokens, &[])
}

pub fn formatting_edits(text: &str, _analysis: &DocumentAnalysis) -> Vec<lsp_types::TextEdit> {
    crate::core::formatting::formatting_edits(text)
}

fn diagnostic_range_for_parse_error(
    error: &SyntaxErrorInfo,
    tokens: &[SpannedToken],
    text: &str,
) -> Range {
    if let Some(token) = tokens.iter().find(|tok| tok.start == error.position) {
        TextRange {
            start: token.start,
            end: token.end,
        }
        .to_lsp_range(text)
    } else {
        Range {
            start: offset_to_position(text, error.position),
            end: offset_to_position(text, error.position + 1),
        }
    }
}

fn diagnostic_for_lowering_error(
    error: &HirError,
    tokens: &[SpannedToken],
    text: &str,
) -> Diagnostic {
    let message = error.message.clone();
    let range = if let Some(span) = error.span {
        let end = span.end.max(span.start + 1);
        TextRange {
            start: span.start,
            end,
        }
        .to_lsp_range(text)
    } else {
        let undefined_var = error
            .message
            .split(':')
            .next_back()
            .map(str::trim)
            .and_then(|s| s.split_whitespace().last())
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric() && c != '_'));
        if let Some(name) = undefined_var {
            find_symbol_range(tokens, name, None)
                .unwrap_or(TextRange { start: 0, end: 1 })
                .to_lsp_range(text)
        } else {
            Range {
                start: Position::new(0, 0),
                end: Position::new(0, 0),
            }
        }
    };

    Diagnostic {
        range,
        severity: Some(DiagnosticSeverity::ERROR),
        code: None,
        code_description: None,
        source: Some("runmat-hir".into()),
        message,
        related_information: None,
        tags: None,
        data: None,
    }
}

fn diagnostic_for_compile_error(error: &CompileError, text: &str) -> Diagnostic {
    let message = error.message.clone();
    let range = if let Some(span) = error.span {
        let end = span.end.max(span.start + 1);
        TextRange {
            start: span.start,
            end,
        }
        .to_lsp_range(text)
    } else {
        Range {
            start: Position::new(0, 0),
            end: Position::new(0, 0),
        }
    };

    Diagnostic {
        range,
        severity: Some(DiagnosticSeverity::ERROR),
        code: None,
        code_description: None,
        source: Some("runmat-vm".into()),
        message,
        related_information: None,
        tags: None,
        data: None,
    }
}

fn diagnostic_for_hir_lint(diag: &HirDiagnostic, text: &str) -> Diagnostic {
    let span = diag.primary.span;
    let end = span.end.max(span.start + 1);
    let range = TextRange {
        start: span.start,
        end,
    }
    .to_lsp_range(text);
    let severity = match diag.severity {
        HirDiagnosticSeverity::Error => DiagnosticSeverity::ERROR,
        HirDiagnosticSeverity::Warning => DiagnosticSeverity::WARNING,
        HirDiagnosticSeverity::Information => DiagnosticSeverity::INFORMATION,
        HirDiagnosticSeverity::Help => DiagnosticSeverity::HINT,
    };
    Diagnostic {
        range,
        severity: Some(severity),
        code: Some(lsp_types::NumberOrString::String(diag.code.to_string())),
        code_description: None,
        source: Some("runmat-hir".into()),
        message: diag.message.clone(),
        related_information: None,
        tags: None,
        data: None,
    }
}

fn token_at_offset(tokens: &[SpannedToken], offset: usize) -> Option<&SpannedToken> {
    if tokens.is_empty() {
        return None;
    }
    let mut left = 0usize;
    let mut right = tokens.len();
    while left < right {
        let mid = (left + right) / 2;
        let token = &tokens[mid];
        if offset < token.start {
            right = mid;
        } else if offset >= token.end {
            left = mid + 1;
        } else {
            return Some(token);
        }
    }
    None
}

fn token_index_at_offset(tokens: &[SpannedToken], offset: usize) -> Option<usize> {
    if tokens.is_empty() {
        return None;
    }
    let mut left = 0usize;
    let mut right = tokens.len();
    while left < right {
        let mid = (left + right) / 2;
        let token = &tokens[mid];
        if offset < token.start {
            right = mid;
        } else if offset >= token.end {
            left = mid + 1;
        } else {
            return Some(mid);
        }
    }
    None
}

fn is_identifier_lexeme(lexeme: &str) -> bool {
    let mut chars = lexeme.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn qualified_call_name_at_offset(tokens: &[SpannedToken], offset: usize) -> Option<String> {
    let mut idx = token_index_at_offset(tokens, offset)?;
    if tokens.get(idx)?.lexeme == "(" {
        idx = idx.checked_sub(1)?;
    }
    if !is_identifier_lexeme(&tokens.get(idx)?.lexeme) {
        return None;
    }

    let mut start = idx;
    while start >= 2
        && tokens.get(start - 1)?.lexeme == "."
        && is_identifier_lexeme(&tokens.get(start - 2)?.lexeme)
    {
        start -= 2;
    }

    let mut end = idx;
    while end + 2 < tokens.len()
        && tokens.get(end + 1)?.lexeme == "."
        && is_identifier_lexeme(&tokens.get(end + 2)?.lexeme)
    {
        end += 2;
    }

    let mut name = String::new();
    for token in &tokens[start..=end] {
        name.push_str(&token.lexeme);
    }
    Some(name)
}

#[derive(Clone, Debug)]
struct SymbolAtCursor {
    name: String,
    is_local: bool,
}

fn symbol_name_at(
    text: &str,
    analysis: &DocumentAnalysis,
    position: &Position,
) -> Option<SymbolAtCursor> {
    let offset = position_to_offset(text, position);
    let token = token_at_offset(&analysis.tokens, offset)?;
    let semantic = analysis.semantic.as_ref()?;
    if let Some(func) = semantic.function_at_offset(offset) {
        if func.variables.contains_key(&token.lexeme) {
            return Some(SymbolAtCursor {
                name: token.lexeme.clone(),
                is_local: true,
            });
        }
    }
    Some(SymbolAtCursor {
        name: token.lexeme.clone(),
        is_local: false,
    })
}

fn local_symbol_references(
    text: &str,
    analysis: &DocumentAnalysis,
    position: &Position,
    uri: &Url,
    symbol_name: &str,
) -> Vec<Location> {
    let offset = position_to_offset(text, position);
    let function_range = analysis
        .semantic
        .as_ref()
        .and_then(|semantic| semantic.function_at_offset(offset).map(|f| f.range));
    analysis
        .tokens
        .iter()
        .filter(|token| token.lexeme == symbol_name)
        .filter(|token| {
            if let Some(range) = function_range {
                return token.start >= range.start && token.end <= range.end;
            }
            true
        })
        .map(|token| Location {
            uri: uri.clone(),
            range: TextRange {
                start: token.start,
                end: token.end,
            }
            .to_lsp_range(text),
        })
        .collect()
}

fn project_function_definitions(
    project: &ProjectContext,
    symbol_name: &str,
    compat: CompatMode,
) -> Vec<Location> {
    let mut locations = Vec::new();
    for source_file in project.visible_source_files() {
        let Some(text) = read_file_text(source_file) else {
            continue;
        };
        let Some(source_name) = source_file.to_str() else {
            continue;
        };
        let analysis = analyze_document_with_compat_and_source(&text, compat, Some(source_name));
        for lookup_name in project.definition_lookup_names(source_file, symbol_name) {
            for range in function_definitions_in_document(&text, &analysis, lookup_name) {
                if let Some(uri) = file_path_to_url(source_file) {
                    locations.push(Location { uri, range });
                }
            }
        }
    }
    locations
}

fn project_function_references(
    project: &ProjectContext,
    symbol_name: &str,
    compat: CompatMode,
) -> Vec<Location> {
    let mut locations = Vec::new();
    for source_file in project.visible_source_files() {
        let Some(text) = read_file_text(source_file) else {
            continue;
        };
        let Some(source_name) = source_file.to_str() else {
            continue;
        };
        let analysis = analyze_document_with_compat_and_source(&text, compat, Some(source_name));
        for range in function_references_in_document(&text, &analysis, symbol_name) {
            if let Some(uri) = file_path_to_url(source_file) {
                locations.push(Location { uri, range });
            }
        }
    }
    locations
}

async fn project_function_definitions_async(
    project: &ProjectContext,
    symbol_name: &str,
    compat: CompatMode,
) -> Vec<Location> {
    let mut locations = Vec::new();
    for source_file in project.visible_source_files() {
        let Ok(text) = runmat_filesystem::read_to_string_async(source_file).await else {
            continue;
        };
        let Some(source_name) = source_file.to_str() else {
            continue;
        };
        let analysis =
            analyze_document_with_compat_and_source_async(&text, compat, Some(source_name)).await;
        for lookup_name in project.definition_lookup_names(source_file, symbol_name) {
            for range in function_definitions_in_document(&text, &analysis, lookup_name) {
                if let Some(uri) = file_path_to_url(source_file) {
                    locations.push(Location { uri, range });
                }
            }
        }
    }
    locations
}

async fn project_function_references_async(
    project: &ProjectContext,
    symbol_name: &str,
    compat: CompatMode,
) -> Vec<Location> {
    let mut locations = Vec::new();
    for source_file in project.visible_source_files() {
        let Ok(text) = runmat_filesystem::read_to_string_async(source_file).await else {
            continue;
        };
        let Some(source_name) = source_file.to_str() else {
            continue;
        };
        let analysis =
            analyze_document_with_compat_and_source_async(&text, compat, Some(source_name)).await;
        for range in function_references_in_document(&text, &analysis, symbol_name) {
            if let Some(uri) = file_path_to_url(source_file) {
                locations.push(Location { uri, range });
            }
        }
    }
    locations
}

fn read_file_text(path: &std::path::Path) -> Option<String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        futures::executor::block_on(runmat_filesystem::read_to_string_async(path)).ok()
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = path;
        None
    }
}

fn uri_file_source_name(uri: &Url) -> Option<String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        uri.to_file_path()
            .ok()
            .and_then(|path| path.to_str().map(str::to_owned))
    }
    #[cfg(target_arch = "wasm32")]
    {
        if uri.scheme() != "file" {
            return None;
        }
        let path = uri.path();
        if path.is_empty() {
            None
        } else {
            Some(path.to_string())
        }
    }
}

fn dedupe_locations(locations: &mut Vec<Location>) {
    let mut seen = std::collections::HashSet::new();
    locations.retain(|loc| {
        let key = format!(
            "{}:{}:{}:{}:{}",
            loc.uri,
            loc.range.start.line,
            loc.range.start.character,
            loc.range.end.line,
            loc.range.end.character
        );
        seen.insert(key)
    });
}

fn file_path_to_url(path: &std::path::Path) -> Option<Url> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Url::from_file_path(path).ok()
    }
    #[cfg(target_arch = "wasm32")]
    {
        let raw = path.to_str()?;
        let normalized = if raw.starts_with('/') {
            raw.to_string()
        } else {
            format!("/{raw}")
        };
        Url::parse(&format!("file://{normalized}")).ok()
    }
}

pub fn function_definitions_in_document(
    text: &str,
    analysis: &DocumentAnalysis,
    symbol_name: &str,
) -> Vec<Range> {
    let Some(semantic) = analysis.semantic.as_ref() else {
        return Vec::new();
    };
    semantic
        .functions
        .iter()
        .filter(|function| function.name == symbol_name)
        .map(|function| function.signature.name_range.to_lsp_range(text))
        .collect()
}

pub fn function_references_in_document(
    text: &str,
    analysis: &DocumentAnalysis,
    symbol_name: &str,
) -> Vec<Range> {
    let Some(lowering) = analysis.lowering.as_ref() else {
        return Vec::new();
    };
    lowering
        .hir_index
        .calls
        .iter()
        .filter(|call| call_resolution_matches_symbol(call, symbol_name))
        .filter_map(|call| span_to_text_range(call.span, text.len()))
        .map(|range| range.to_lsp_range(text))
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn function_rename_references_in_document(
    text: &str,
    analysis: &DocumentAnalysis,
    symbol_name: &str,
) -> Vec<Range> {
    let Some(lowering) = analysis.lowering.as_ref() else {
        return Vec::new();
    };
    let leaf = symbol_name.rsplit('.').next().unwrap_or(symbol_name);
    lowering
        .hir_index
        .calls
        .iter()
        .filter(|call| call_resolution_matches_symbol(call, symbol_name))
        .filter_map(|call| {
            analysis
                .tokens
                .iter()
                .take_while(|token| token.start < call.span.end)
                .filter(|token| call.span.start <= token.start)
                .take_while(|token| token.token != Token::LParen)
                .filter(|token| token.token == Token::Ident && token.lexeme == leaf)
                .last()
        })
        .map(|token| {
            TextRange {
                start: token.start,
                end: token.end,
            }
            .to_lsp_range(text)
        })
        .collect()
}

fn call_resolution_matches_symbol(call: &runmat_hir::CallResolution, symbol_name: &str) -> bool {
    let Some(display) = call.name.display_name() else {
        return false;
    };
    if display == symbol_name {
        return true;
    }
    if let CallKind::PackageFunction(path) = &call.kind {
        return def_path_symbol_variants(path)
            .iter()
            .any(|variant| variant == symbol_name);
    }
    false
}

fn find_symbol_range(
    tokens: &[SpannedToken],
    name: &str,
    scope: Option<&TextRange>,
) -> Option<TextRange> {
    tokens
        .iter()
        .filter(|tok| matches!(tok.token, Token::Ident))
        .filter(|tok| tok.lexeme == name)
        .map(|tok| TextRange {
            start: tok.start,
            end: tok.end,
        })
        .find(|range| scope.is_none_or(|scope| scope.contains(range.start)))
}

fn completion_from_semantic(semantic: &AnalysisModel, offset: usize) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    let mut builtin_names = std::collections::BTreeSet::new();
    for var in semantic.globals.values() {
        items.push(variable_completion(var, semantic, offset));
    }
    for func in &semantic.functions {
        for var in func.variables.values() {
            items.push(variable_completion(var, semantic, offset));
        }
        items.push(function_completion(func));
    }
    for entry in runmat_builtins::public_builtin_catalog_entries() {
        if entry.descriptor.completion_policy == BuiltinCompletionPolicy::HiddenInternal {
            continue;
        }
        builtin_names.insert(entry.identity.name.to_ascii_lowercase());
        items.push(catalog_completion(entry));
    }
    for alias in runmat_builtins::builtin_catalog_aliases() {
        let Some(entry) =
            runmat_builtins::public_builtin_catalog_entry_by_name(alias.canonical.name)
        else {
            continue;
        };
        if entry.descriptor.completion_policy == BuiltinCompletionPolicy::HiddenInternal
            || !builtin_names.insert(alias.alias.name.to_ascii_lowercase())
        {
            continue;
        }
        items.push(catalog_alias_completion(alias.alias.name, entry));
    }
    for func in runmat_builtins::builtin_functions() {
        if func.descriptor.is_some_and(|descriptor| {
            descriptor.completion_policy == BuiltinCompletionPolicy::HiddenInternal
        }) {
            continue;
        }
        if !builtin_names.insert(func.name.to_ascii_lowercase()) {
            continue;
        }
        items.push(builtin_completion(func));
    }
    for constant in runmat_builtins::builtin_constant_catalog_entries() {
        items.push(constant_completion(constant));
    }
    items
}

fn catalog_alias_completion(
    alias: &str,
    entry: &runmat_builtins::BuiltinCatalogEntry,
) -> CompletionItem {
    let mut completion = catalog_completion(entry);
    completion.label = alias.to_string();
    completion.detail = completion
        .detail
        .map(|detail| format!("{detail} (alias of {})", entry.identity.name));
    completion
}

fn catalog_completion(entry: &runmat_builtins::BuiltinCatalogEntry) -> CompletionItem {
    CompletionItem {
        label: entry.identity.name.to_string(),
        kind: Some(lsp_types::CompletionItemKind::FUNCTION),
        detail: docs::catalog_completion_signature_label(entry)
            .or_else(|| Some(format!("builtin: {}", entry.identity.name))),
        documentation: Some(lsp_types::Documentation::MarkupContent(
            lsp_types::MarkupContent {
                kind: lsp_types::MarkupKind::Markdown,
                value: docs::build_catalog_hover(entry),
            },
        )),
        ..Default::default()
    }
}

fn variable_completion(
    var: &VariableSymbol,
    semantic: &AnalysisModel,
    offset: usize,
) -> CompletionItem {
    let fact = semantic
        .facts
        .as_ref()
        .and_then(|facts| facts.fact_at(var.binding, offset))
        .and_then(|observation| observation.fact.as_ref());
    CompletionItem {
        label: var.name.clone(),
        kind: Some(lsp_types::CompletionItemKind::VARIABLE),
        detail: Some(format!(
            "{}: {}",
            var.kind.as_label(),
            crate::core::semantic::presentation::format_fact(fact)
        )),
        ..Default::default()
    }
}

fn function_completion(func: &FunctionSemantic) -> CompletionItem {
    CompletionItem {
        label: func.name.clone(),
        kind: Some(lsp_types::CompletionItemKind::FUNCTION),
        detail: Some(func.signature.display()),
        ..Default::default()
    }
}

fn builtin_completion(func: &BuiltinFunction) -> CompletionItem {
    let detail = if let Some(label) = docs::completion_signature_label(func) {
        label
    } else if !func.param_types.is_empty() {
        let params: Vec<String> = func.param_types.iter().map(format_type).collect();
        format!(
            "({}) -> {}",
            params.join(", "),
            format_type(&func.return_type)
        )
    } else {
        format!("builtin: {}", func.name)
    };

    let documentation = Some(lsp_types::Documentation::MarkupContent(
        lsp_types::MarkupContent {
            kind: lsp_types::MarkupKind::Markdown,
            value: docs::build_builtin_hover(func),
        },
    ));

    CompletionItem {
        label: func.name.to_string(),
        kind: Some(lsp_types::CompletionItemKind::FUNCTION),
        detail: Some(detail),
        documentation,
        ..Default::default()
    }
}

fn constant_completion(constant: &runmat_builtins::BuiltinConstantCatalogEntry) -> CompletionItem {
    CompletionItem {
        label: constant.name.to_string(),
        kind: Some(lsp_types::CompletionItemKind::CONSTANT),
        detail: Some("constant".into()),
        ..Default::default()
    }
}

fn format_type(ty: &Type) -> String {
    match ty {
        Type::Tensor { shape } => format!("Tensor {{ {} }}", format_shape(shape)),
        Type::Logical { shape } => format!("Logical {{ {} }}", format_shape(shape)),
        _ => format!("{ty:?}"),
    }
}

fn format_shape(shape: &Option<Vec<Option<usize>>>) -> String {
    let Some(shape) = shape else {
        return "unknown".to_string();
    };
    if shape.len() == 2 {
        let rows = format_dim(shape[0]);
        let cols = format_dim(shape[1]);
        return format!("{rows} x {cols}");
    }
    let dims: Vec<String> = shape.iter().map(|d| format_dim(*d)).collect();
    format!("shape: [{}]", dims.join(", "))
}

fn format_dim(dim: Option<usize>) -> String {
    match dim {
        Some(value) => value.to_string(),
        None => "unknown".to_string(),
    }
}

fn span_to_text_range(span: runmat_hir::Span, text_len: usize) -> Option<TextRange> {
    if span.end <= span.start || span.start >= text_len {
        return None;
    }
    Some(TextRange {
        start: span.start,
        end: span.end.min(text_len),
    })
}

fn def_path_symbol_variants(path: &DefPath) -> Vec<String> {
    let mut variants = Vec::new();
    let item = path.display_name();
    let module = path.module.display_name();
    if let Some(item_name) = item.as_ref() {
        variants.push(item_name.clone());
    }
    if let (Some(module_name), Some(item_name)) = (module.as_ref(), item.as_ref()) {
        variants.push(format!("{module_name}.{item_name}"));
        variants.push(format!("{}.{}.{}", path.package.0, module_name, item_name));
    }
    if variants.is_empty() {
        if let Some(item_name) = path.display_name() {
            variants.push(item_name);
        }
    }
    variants
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::core::workspace::workspace_symbols_with_project;
    #[cfg(not(target_arch = "wasm32"))]
    use futures::executor::block_on;
    #[cfg(not(target_arch = "wasm32"))]
    use lsp_types::Url;
    #[cfg(not(target_arch = "wasm32"))]
    use std::fs;
    #[cfg(not(target_arch = "wasm32"))]
    use std::path::PathBuf;
    #[cfg(not(target_arch = "wasm32"))]
    use std::time::{SystemTime, UNIX_EPOCH};

    // The catalog owns spellings and overloads; this consumer checks that each
    // source call projects the complete descriptor for the correct identity.
    fn assert_catalog_signature_projection(text: &str, identity: &str) {
        let entry = runmat_builtins::public_builtin_catalog_entry_by_name(identity)
            .unwrap_or_else(|| panic!("missing canonical identity {identity}"));
        let expected: Vec<&str> = entry
            .descriptor
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(!expected.is_empty(), "missing descriptor for {identity}");
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let actual = signature_help_at(text, &analysis, &position)
            .unwrap_or_else(|| panic!("missing signature projection for {text}"));
        let labels: Vec<&str> = actual
            .signatures
            .iter()
            .map(|signature| signature.label.as_str())
            .collect();
        assert_eq!(labels, expected, "wrong canonical descriptor for {text}");
        assert!(actual
            .active_signature
            .is_some_and(|index| (index as usize) < actual.signatures.len()));
    }

    #[test]
    fn hover_returns_builtin_docs() {
        let text = "plot(1, 2);";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        if let Some(err) = &analysis.syntax_error {
            panic!(
                "unexpected parse error at {}: {}",
                err.position, err.message
            );
        }
        if let Some(err) = &analysis.lowering_error {
            panic!("unexpected lowering error: {err}");
        }
        assert!(
            !analysis.tokens.is_empty(),
            "expected tokenize_detailed to produce tokens"
        );
        assert!(runmat_builtins::public_builtin_catalog_entry_by_name("plot").is_some());
        let position = lsp_types::Position::new(0, 0);
        let offset = position_to_offset(text, &position);
        let token = token_at_offset(&analysis.tokens, offset)
            .unwrap_or_else(|| panic!("no token found at offset {offset}"));
        assert_eq!(token.lexeme, "plot", "unexpected token at hover location");
        let hover = hover_at(text, &analysis, &position);
        let hover = hover.expect("expected hover for builtin function");
        match hover.contents {
            lsp_types::HoverContents::Markup(markup) => {
                assert_eq!(
                    markup.kind,
                    lsp_types::MarkupKind::Markdown,
                    "expected markdown hover"
                );
                assert!(
                    markup.value.contains("h = plot(X, Y)"),
                    "expected descriptor-backed plot signature header, got:\n{}",
                    markup.value
                );
                assert!(
                    markup
                        .value
                        .contains("Docs: https://runmat.com/docs/reference/builtins/"),
                    "expected docs link, got:\n{}",
                    markup.value
                );
            }
            other => panic!("expected Markup hover contents, got {other:?}"),
        }
    }

    #[test]
    fn hover_uses_descriptor_signature_header_when_available() {
        let text = "zeros(2);";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let hover = hover_at(text, &analysis, &position).expect("expected hover");
        let value = match hover.contents {
            lsp_types::HoverContents::Markup(markup) => markup.value,
            other => panic!("unexpected hover contents {other:?}"),
        };
        assert!(
            value.contains("A = zeros(n)"),
            "expected descriptor signature in hover header, got:\n{value}"
        );
    }

    #[test]
    fn catalog_backed_builtin_remains_visible_without_legacy_registration() {
        assert!(runmat_builtins::builtin_catalog_entry_by_name("full").is_some());
        assert!(runmat_builtins::builtin_function_by_name("full").is_none());

        let text = "full(1);";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let hover = hover_at(text, &analysis, &position).expect("catalog hover");
        let lsp_types::HoverContents::Markup(markup) = hover.contents else {
            panic!("catalog hover should use markdown")
        };
        assert!(markup.value.contains("A = full(S)"), "{}", markup.value);

        let signatures = signature_help_at(text, &analysis, &position)
            .expect("catalog signature help")
            .signatures;
        assert_eq!(signatures[0].label, "A = full(S)");

        let completions = completion_at(text, &analysis, &position);
        assert!(completions.iter().any(|item| item.label == "full"));
    }

    #[test]
    fn signature_help_uses_builtin_descriptor_signatures() {
        let text = "linspace(0, 1);";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let sig = signature_help_at(text, &analysis, &position).expect("signature help");
        let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
        assert!(
            labels.contains(&"x = linspace(start, stop)")
                && labels.contains(&"x = linspace(start, stop, n)"),
            "expected descriptor-backed linspace signatures, got {:?}",
            labels
        );
    }

    #[test]
    fn signature_help_uses_range_descriptor_signatures() {
        let text = "range([1 2 3]);";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let sig = signature_help_at(text, &analysis, &position).expect("signature help");
        let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
        assert!(
            labels.contains(&"y = range(X)")
                && labels.contains(&"y = range(X, dim_or_vecdim, nanflag)"),
            "expected descriptor-backed range signatures, got {:?}",
            labels
        );
    }

    #[test]
    fn signature_help_uses_rand_family_descriptor_signatures() {
        let cases = [
            ("rand(2);", "A = rand(n)"),
            ("randn(2);", "A = randn(n)"),
            ("randi(10);", "R = randi(imax)"),
            ("randperm(10, 3);", "p = randperm(n, k)"),
            ("exprnd(2);", "r = exprnd(mu)"),
            (
                "normrnd(0, 1, 3, 4);",
                "r = normrnd(mu, sigma, sz1, sz2, ...)",
            ),
            ("gamrnd(2, 3, 3, 4);", "r = gamrnd(a, b, sz1, sz2, ...)"),
            ("unifrnd(0, 1, [3 4]);", "r = unifrnd(a, b, sz)"),
            ("rng();", "s = rng()"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position)
                .unwrap_or_else(|| panic!("expected signature help for strings-core case: {text}"));
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_stats_summary_descriptors() {
        let cases = [
            ("mode([1 2 2 3]);", "M = mode(X)"),
            ("mode([1 2 2 3], 1);", "M = mode(X, dim_or_vecdim_or_all)"),
            (
                "mode([1 2 2 3], 'all');",
                "M = mode(X, dim_or_vecdim_or_all)",
            ),
            ("cov([1 2; 3 4]);", "C = cov(X)"),
            ("cov([1 2; 3 4], 0);", "C = cov(X, normalization)"),
            (
                "corrcoef([1 2; 3 4], 'rows', 'complete');",
                "___ = corrcoef(A, Name, Value)",
            ),
            ("corrcoef([1 2; 3 4], [5 6; 7 8]);", "R = corrcoef(A, B)"),
            ("corrcov([1 0.2; 0.2 1]);", "R = corrcov(C)"),
            (
                "cov2corr([1 0.2; 0.2 1]);",
                "[ExpSigma, ExpCorrC] = cov2corr(ExpCovariance)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).unwrap_or_else(|| {
                panic!(
                    "signature help missing for `{text}`; status={}",
                    analysis.status_message()
                )
            });
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_stats_hist_descriptors() {
        let cases = [
            ("histcounts([1 2 2 3]);", "N = histcounts(X)"),
            ("histcounts([1 2 2 3], 5);", "N = histcounts(X, bins)"),
            ("histcounts2([1 2 3], [4 5 6]);", "N = histcounts2(X, Y)"),
            (
                "histcounts2([1 2 3], [4 5 6], [1 2], [4 5]);",
                "N = histcounts2(X, Y, binsX, binsY)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).unwrap_or_else(|| {
                panic!(
                    "signature help missing for `{text}`; status={}",
                    analysis.status_message()
                )
            });
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_plotting_axis_label_descriptors() {
        let cases = [
            ("xlabel('Time');", "h = xlabel(txt)"),
            ("ylabel('Amplitude');", "h = ylabel(txt)"),
            ("zlabel('Depth');", "h = zlabel(txt)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_plotting_title_descriptors() {
        let cases = [
            ("title('Signal');", "h = title(txt)"),
            ("sgtitle('Overview');", "h = sgtitle(txt)"),
            ("suptitle('Overview');", "h = suptitle(txt)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_plotting_limit_descriptors() {
        let cases = [
            ("xlim([0 1]);", "xlim"),
            ("ylim([0 1]);", "ylim"),
            ("zlim([0 1]);", "zlim"),
            ("clim([0 1]);", "clim"),
            ("caxis([0 1]);", "caxis"),
        ];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_plotting_reference_line_descriptors() {
        let cases = [
            ("xline(1.5);", "h = xline(x)"),
            ("yline(2.0);", "h = yline(y)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_plotting_handle_core_descriptors() {
        let cases = [
            ("figure();", "fig = figure()"),
            ("gcf();", "fig = gcf()"),
            ("gca();", "ax = gca()"),
            ("isgraphics(1);", "tf = isgraphics(h)"),
            ("ishandle(1);", "tf = ishandle(h)"),
            ("get(1);", "value = get(h)"),
            (
                "set(1, 'Visible', true);",
                "status = set(h, property, value, ...)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_plotting_control_descriptors() {
        let cases = [("drawnow();", "drawnow"), ("hold('on');", "hold")];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_plotting_view_layout_descriptors() {
        let cases = [
            ("subplot(1, 2, 1);", "ax = subplot(rows, cols, position)"),
            ("view(45, 30);", "angles = view(az, el)"),
            ("legend('off');", "h = legend(mode)"),
            ("grid('on');", "enabled = grid(mode)"),
            (
                "axis([0 1 0 1]);",
                "axis([xmin xmax ymin ymax | ... cmin cmax])",
            ),
            ("cla();", "ok = cla()"),
            ("colormap('parula');", "colormap(name)"),
            ("shading('flat');", "ok = shading(mode)"),
            ("colorbar('off');", "enabled = colorbar(mode)"),
            ("semilogx([1 10 100]);", "h = semilogx(Y)"),
            ("semilogy([1 10 100]);", "h = semilogy(Y)"),
            ("loglog([1 10 100]);", "h = loglog(Y)"),
            (
                "regexp('abc123', '\\\\d+');",
                "out = regexp(subject, pattern)",
            ),
            ("regexpi('Alpha', 'a');", "out = regexpi(subject, pattern)"),
            (
                "regexprep('abc123', '\\\\d+', 'X');",
                "out = regexprep(subject, pattern, replacement)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_plotting_chart_descriptors() {
        let cases = [
            ("area([1 2 3]);", "h = area(Y)"),
            ("area([1 2 3], [3 2 1]);", "h = area(X, Y)"),
            ("bar([1 2 3]);", "h = bar(Y)"),
            ("bar([1 2 3], [3 2 1]);", "h = bar(X, Y)"),
            ("plot([1 2 3], [3 2 1]);", "h = plot(X, Y)"),
            ("plot3([1 2], [2 3], [3 4]);", "h = plot3(X, Y, Z)"),
            ("quiver([1 2], [2 1]);", "h = quiver(U, V)"),
            ("errorbar([1 2], [0.1 0.2]);", "h = errorbar(Y, E)"),
            (
                "fill3([0 1 0], [0 0 1], [0 1 0], 'r');",
                "h = fill3(X, Y, Z, C)",
            ),
            ("pie([1 2 3]);", "h = pie(X)"),
            ("heatmap([1 2; 3 4]);", "h = heatmap(CData)"),
            ("imshow([1 2; 3 4]);", "h = imshow(I)"),
            ("patch([0 1 0], [0 0 1], 'r');", "h = patch(X, Y, C)"),
            ("mesh([1 2; 3 4]);", "h = mesh(Z)"),
            ("meshc([1 2; 3 4]);", "h = meshc(Z)"),
            ("surf([1 2; 3 4]);", "h = surf(Z)"),
            ("surfc([1 2; 3 4]);", "h = surfc(Z)"),
            ("stairs([1 2 3]);", "h = stairs(Y)"),
            ("stem([1 2 3]);", "h = stem(Y)"),
            ("scatter([1 2 3], [3 2 1]);", "h = scatter(X, Y)"),
            (
                "scatter3([1 2 3], [3 2 1], [0 1 0]);",
                "h = scatter3(X, Y, Z)",
            ),
            ("image([1 2; 3 4]);", "h = image(C)"),
            ("imagesc([1 2; 3 4]);", "h = imagesc(C)"),
            ("contour([1 2; 3 4]);", "h = contour(Z)"),
            ("contourf([1 2; 3 4]);", "h = contourf(Z)"),
            ("contourf(1, [1 2; 3 4]);", "h = contourf(ax, ...)"),
            ("copyobj(1, 2);", "hnew = copyobj(h, parent)"),
            ("hist([1 2 3]);", "N = hist(X)"),
            ("histogram([1 2 3]);", "h = histogram(X)"),
            ("text(1, 2, 'pt');", "h = text(x, y, label)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_empty_and_magic_descriptors() {
        let cases = [("empty(0, 3);", "empty"), ("magic(5);", "magic")];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_colon_fill_meshgrid_peaks_descriptors() {
        let cases = [
            ("colon(1, 0.5, 3);", "x = colon(start, step, stop)"),
            ("fill(2, 3, 4);", "p = fill(X, Y, C, ..., Name, Value)"),
            ("meshgrid([1 2 3], [4 5]);", "[X,Y] = meshgrid(x, y)"),
            ("peaks(20);", "Z = peaks(n)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position)
                .unwrap_or_else(|| panic!("expected signature help for strings-core case: {text}"));
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_introspection_descriptors() {
        let cases = [
            ("isempty([]);", "tf = isempty(A)"),
            ("iscolumn([1; 2]);", "tf = iscolumn(A)"),
            ("ismatrix([1 2; 3 4]);", "tf = ismatrix(A)"),
            ("isrow([1 2]);", "tf = isrow(A)"),
            ("isscalar(1);", "tf = isscalar(A)"),
            ("isvector([1 2 3]);", "tf = isvector(A)"),
            ("length([1 2 3]);", "n = length(A)"),
            ("ndims(ones(2,2,2));", "n = ndims(A)"),
            ("height([1 2; 3 4]);", "n = height(A)"),
            ("width([1 2; 3 4]);", "n = width(A)"),
            ("numel([1 2; 3 4], 1, 2);", "n = numel(A, dim1, dim2, ...)"),
            ("size([1 2; 3 4], 1);", "sz = size(A, dim)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position)
                .unwrap_or_else(|| panic!("signature help missing for case: {text}"));
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_workspace_introspection_descriptors() {
        let cases = [
            ("class(1);", "name = class(A)"),
            ("isa(1, \"double\");", "tf = isa(A, type_name)"),
            ("ischar('abc');", "tf = ischar(A)"),
            ("isstring(\"abc\");", "tf = isstring(A)"),
            ("which(\"sin\");", "result = which(name)"),
            ("who();", "names = who()"),
            ("whos();", "vars = whos()"),
            ("clear(\"x\");", "clear(name, ...)"),
            (
                "clearvars(\"x\", \"-except\", \"y\");",
                "clearvars(name_or_option, ...)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position)
                .unwrap_or_else(|| panic!("expected signature help for diagnostics case: {text}"));
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_logical_test_descriptors() {
        let cases = [
            ("logical(1);", "tf = logical(A)"),
            ("isfinite([1 NaN]);", "tf = isfinite(A)"),
            ("isinf([1 Inf]);", "tf = isinf(A)"),
            ("isnan([1 NaN]);", "tf = isnan(A)"),
            ("islogical(true);", "tf = islogical(A)"),
            ("isnumeric(1);", "tf = isnumeric(A)"),
            ("isreal(1+0i);", "tf = isreal(A)"),
            ("isgpuarray(1);", "tf = isgpuarray(A)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position)
                .unwrap_or_else(|| panic!("expected signature help for diagnostics case: {text}"));
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_logical_rel_descriptors() {
        let cases = [
            ("eq(1, 1);", "tf = eq(A, B)"),
            ("ne(1, 2);", "tf = ne(A, B)"),
            ("lt(1, 2);", "tf = lt(A, B)"),
            ("le(1, 2);", "tf = le(A, B)"),
            ("gt(2, 1);", "tf = gt(A, B)"),
            ("ge(2, 1);", "tf = ge(A, B)"),
            ("isequal(1, 1, 1);", "tf = isequal(A, B, ...)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_logical_bit_descriptors() {
        let cases = [
            ("and(true, false);", "tf = and(A, B)"),
            ("or(true, false);", "tf = or(A, B)"),
            ("xor(true, false);", "tf = xor(A, B)"),
            ("not(true);", "tf = not(A)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_timing_descriptors() {
        let cases = [
            ("pause(0);", "out = pause(duration)"),
            ("tic();", "timerVal = tic()"),
            ("toc();", "elapsed = toc()"),
            ("timeit(@foo);", "t = timeit(f)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_gpu_acceleration_descriptors() {
        let cases = [
            ("arrayfun(\"sin\", 1);", "arrayfun"),
            ("gather(1);", "gather"),
            ("gpuArray(1);", "gpuArray"),
            ("gpuDevice();", "gpuDevice"),
            ("gpuDevice(1);", "gpuDevice"),
            ("gpuInfo();", "gpuInfo"),
            ("pagefun(\"mtimes\", 1, 1);", "pagefun"),
        ];

        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_array_indexing_descriptors() {
        let cases = [
            ("find([1 0 2]);", "find"),
            ("ind2sub([3 4], 7);", "ind2sub"),
            ("sub2ind([3 4], 2, 3);", "sub2ind"),
        ];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_array_shape_flip_descriptors() {
        let cases = [
            ("flip([1 2 3]);", "B = flip(A)"),
            ("flip([1 2 3], 1);", "B = flip(A, dim_or_direction)"),
            ("fliplr([1 2 3]);", "B = fliplr(A)"),
            ("flipud([1; 2; 3]);", "B = flipud(A)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_array_shape_transform_descriptors() {
        let cases = [
            ("permute([1 2 3], [1 2]);", "B = permute(A, order)"),
            ("ipermute([1 2 3], [1 2]);", "A = ipermute(B, order)"),
            ("reshape([1 2 3 4], [2 2]);", "B = reshape(A, sz)"),
            ("reshape([1 2 3 4], 2, 2);", "B = reshape(A, m, n)"),
            ("squeeze(ones(1,3,1));", "B = squeeze(A)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_array_shape_concat_descriptors() {
        let cases = [
            ("cat(1, [1], [2]);", "B = cat(dim, A1, A2, An...)"),
            (
                "cat(1, [1], [2], \"like\", [0]);",
                "B = cat(dim, A1, A2, An..., \"like\", prototype)",
            ),
            ("horzcat([1], [2]);", "B = horzcat(A1, An...)"),
            ("vertcat([1], [2]);", "B = vertcat(A1, An...)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_array_shape_diagonal_descriptors() {
        let cases = [
            ("diag([1 2 3]);", "D = diag(v)"),
            ("diag([1 2 3], 1);", "D = diag(v, k)"),
            ("diag([1 2 3], [3 4]);", "x = diag(A, k)"),
            ("diag([1 2 3], \"like\", [0]);", "D = diag(v, k)"),
            ("tril([1 2; 3 4]);", "B = tril(A)"),
            ("tril([1 2; 3 4], -1);", "B = tril(A, k)"),
            ("triu([1 2; 3 4]);", "B = triu(A)"),
            ("triu([1 2; 3 4], 1);", "B = triu(A, k)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_array_shape_rotation_product_descriptors() {
        let cases = [
            ("rot90([1 2; 3 4]);", "B = rot90(A)"),
            ("rot90([1 2; 3 4], -1);", "B = rot90(A, k_or_direction)"),
            ("kron([1 2], [3 4]);", "C = kron(A, B)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_array_shape_replication_shift_descriptors() {
        let cases = [
            ("repmat([1 2;3 4], [2 3]);", "B = repmat(A, r)"),
            ("repmat([1 2;3 4], 2, 3);", "B = repmat(A, m, n)"),
            ("repmat([1], 2, 3, 4);", "B = repmat(A, d1, d2, ...)"),
            ("repelem([1 2 3], 2);", "B = repelem(A, R)"),
            ("repelem([1 2;3 4], 2, 3);", "B = repelem(A, R1, R2, ...)"),
            ("circshift([1 2;3 4], 1);", "B = circshift(A, K)"),
            ("circshift([1 2;3 4], 1, 2);", "B = circshift(A, K, dim)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_array_sorting_sets_descriptors() {
        let cases = [
            ("sort([3 1 2]);", "B = sort(A)"),
            ("sort([3 1 2], 'descend');", "B = sort(A, arg1)"),
            ("sort([3 1 2], 1, 'descend');", "B = sort(A, arg1, arg2)"),
            (
                "sort([3 1 2], 'ComparisonMethod', 'abs');",
                "B = sort(A, ..., \"ComparisonMethod\", method)",
            ),
            ("argsort([3 1 2]);", "I = argsort(A)"),
            ("argsort([3 1 2], 'descend');", "I = argsort(A, arg1)"),
            ("issorted([1 2 3]);", "tf = issorted(A)"),
            ("issorted([1 2 3], 'ascend');", "tf = issorted(A, arg1)"),
            (
                "issorted([1 2 3], 'MissingPlacement', 'last');",
                "tf = issorted(A, ..., \"MissingPlacement\", placement)",
            ),
            ("unique([3 1 3 2]);", "C = unique(A)"),
            ("unique([3 1 3 2], 'stable');", "C = unique(A, option...)"),
            ("union([1 3], [2 3]);", "C = union(A, B)"),
            (
                "union([1 3], [2 3], 'stable');",
                "C = union(A, B, option...)",
            ),
            ("intersect([1 2 3], [2 4]);", "C = intersect(A, B)"),
            (
                "intersect([1 2 3], [2 4], 'stable');",
                "C = intersect(A, B, option...)",
            ),
            ("setdiff([1 2 3], [2]);", "C = setdiff(A, B)"),
            (
                "setdiff([1 2 3], [2], 'stable');",
                "C = setdiff(A, B, option...)",
            ),
            ("ismember([1 2 3], [2 4]);", "tf = ismember(A, B)"),
            (
                "ismember([1 2 3], [2 4], 'rows');",
                "tf = ismember(A, B, option...)",
            ),
            ("sortrows([3 1;2 4]);", "B = sortrows(A)"),
            (
                "sortrows([3 1;2 4], [1 -2], 'descend');",
                "B = sortrows(A, column, direction)",
            ),
            (
                "sortrows([3 1;2 4], 'MissingPlacement', 'first');",
                "B = sortrows(A, ..., \"MissingPlacement\", placement)",
            ),
            ("maxk([3 1 2], 2);", "B = maxk(A, k)"),
            (
                "maxk([3 1 2], 2, 'ComparisonMethod', 'abs');",
                "B = maxk(A, k, \"ComparisonMethod\", method)",
            ),
            ("mink([3 1 2], 2, 1);", "B = mink(A, k, dim)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_diagnostics_descriptors() {
        let cases = [
            ("assert(true);", "assert(condition)"),
            (
                "assert(false, \"id:test\", \"failed %d\", 1);",
                "assert(condition, message_id, message, A...)",
            ),
            ("error(\"failure\");", "error(msg)"),
            (
                "error(\"RunMat:demo:bad\", \"failed %d\", 1);",
                "error(errID, ___)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position)
                .unwrap_or_else(|| panic!("expected signature help for diagnostics case: {text}"));
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_io_filetext_stream_descriptors() {
        let cases = [
            ("fopen(\"demo.txt\");", "fid = fopen(filename)"),
            (
                "fopen(\"demo.txt\", \"w\", \"ieee-be\", \"latin1\");",
                "fid = fopen(filename, permission, machinefmt, encoding)",
            ),
            ("fopen(3);", "filename = fopen(fid)"),
            ("fopen(\"all\");", "fids = fopen(\"all\")"),
            (
                "fprintf(\"value=%d\", 7);",
                "count = fprintf(formatSpec, A...)",
            ),
            (
                "fprintf(1, \"value=%d\", 7);",
                "count = fprintf(fid_or_stream, formatSpec, A...)",
            ),
            ("fclose();", "status = fclose()"),
            ("fclose(3);", "status = fclose(fid)"),
            ("fclose(\"all\");", "status = fclose(\"all\")"),
            ("feof(3);", "tf = feof(fid)"),
            ("fileread(\"demo.txt\");", "text = fileread(filename)"),
            (
                "fileread(\"demo.txt\", \"Encoding\", \"utf-8\");",
                "text = fileread(filename, \"Encoding\", encoding)",
            ),
            (
                "filewrite(\"demo.txt\", \"abc\");",
                "count = filewrite(filename, data)",
            ),
            (
                "filewrite(\"demo.txt\", \"abc\", \"WriteMode\", \"append\");",
                "count = filewrite(filename, data, ..., \"WriteMode\", mode)",
            ),
            ("fread(3);", "data = fread(fid)"),
            (
                "fread(3, 10, \"uint8\", 1, \"ieee-le\");",
                "data = fread(fid, size, precision, skip, machinefmt)",
            ),
            (
                "fread(3, \"uint8\", \"ieee-be\");",
                "data = fread(fid, precision, machinefmt)",
            ),
            (
                "fread(3, 10, \"double\", \"like\", zeros(1,1));",
                "data = fread(fid, ..., \"like\", prototype)",
            ),
            ("fwrite(3, [1 2 3]);", "count = fwrite(fid, data)"),
            (
                "fwrite(3, [1 2 3], \"uint16\", 1, \"ieee-be\");",
                "count = fwrite(fid, data, precision, skip, machinefmt)",
            ),
            ("fgetl(3);", "tline = fgetl(fid)"),
            ("fgets(3);", "tline = fgets(fid)"),
            ("fgets(3, 10);", "tline = fgets(fid, nchar)"),
            ("frewind(3);", "frewind(fid)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_io_net_descriptors() {
        let cases = [
            ("read(1);", "data = read(client)"),
            ("read(1, 64);", "data = read(client, count)"),
            (
                "read(1, 64, \"uint16\");",
                "data = read(client, count, datatype)",
            ),
            ("write(1, [1 2 3]);", "count = write(client, data)"),
            (
                "write(1, [1 2 3], \"double\");",
                "count = write(client, data, datatype)",
            ),
            (
                "tcpclient(\"127.0.0.1\", 80);",
                "client = tcpclient(host, port)",
            ),
            (
                "tcpclient(\"127.0.0.1\", 80, \"Timeout\", 1);",
                "client = tcpclient(host, port, Name, Value, ...)",
            ),
            (
                "tcpserver(\"127.0.0.1\", 0);",
                "server = tcpserver(address, port)",
            ),
            (
                "tcpserver(\"127.0.0.1\", 0, \"Timeout\", 1);",
                "server = tcpserver(address, port, Name, Value, ...)",
            ),
            ("accept(1);", "client = accept(server)"),
            (
                "accept(1, \"Timeout\", 1);",
                "client = accept(server, \"Timeout\", timeout)",
            ),
            ("readline(1);", "line = readline(client)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_io_repl_fs_descriptors() {
        let cases = [
            ("addpath(\"toolbox\");", "addpath"),
            ("addpath(\"toolbox\", \"utils\", \"-end\");", "addpath"),
            ("addpath(\"toolbox\", \"-frozen\");", "addpath"),
            ("pwd();", "pwd"),
            ("cd();", "cd"),
            ("cd(\"tmp\");", "cd"),
            ("rmpath(\"toolbox\");", "rmpath"),
            ("rmpath(\"toolbox\", \"utils\");", "rmpath"),
            ("genpath();", "genpath"),
            ("genpath(\"toolbox\");", "genpath"),
            ("genpath(\"toolbox\", \"private\");", "genpath"),
            ("path();", "path"),
            ("path(\"a\");", "path"),
            ("path(\"a\", \"b\");", "path"),
            ("savepath();", "savepath"),
            ("savepath(\"pathdef.m\");", "savepath"),
            ("tempdir();", "tempdir"),
            ("tempname();", "tempname"),
            ("tempname(\"tmp\");", "tempname"),
            ("getenv();", "getenv"),
            ("getenv(\"PATH\");", "getenv"),
            ("setenv(\"A\", \"B\");", "setenv"),
            ("fullfile(\"a\", \"b\");", "fullfile"),
            ("exist(\"sin\");", "exist"),
            ("exist(\"sin\", \"builtin\");", "exist"),
            ("dir();", "dir"),
            ("dir(\".\");", "dir"),
            ("ls();", "ls"),
            ("ls(\".\");", "ls"),
            ("mkdir(\"tmp\");", "mkdir"),
            ("mkdir(\"parent\", \"child\");", "mkdir"),
            ("rmdir(\"tmp\");", "rmdir"),
            ("rmdir(\"tmp\", \"s\");", "rmdir"),
            ("copyfile(\"a\", \"b\");", "copyfile"),
            ("copyfile(\"a\", \"b\", \"f\");", "copyfile"),
            ("movefile(\"a\", \"b\");", "movefile"),
            ("movefile(\"a\", \"b\", \"f\");", "movefile"),
            ("delete(\"a.txt\");", "delete"),
            ("delete(\"a.txt\", \"b.txt\");", "delete"),
        ];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_io_tabular_descriptors() {
        let cases = [
            ("csvread(\"data.csv\");", "M = csvread(filename)"),
            (
                "csvread(\"data.csv\", 1, 2);",
                "M = csvread(filename, row, col)",
            ),
            (
                "csvread(\"data.csv\", 1, 2, \"B2:C4\");",
                "M = csvread(filename, row, col, range)",
            ),
            (
                "csvwrite(\"out.csv\", [1 2; 3 4]);",
                "csvwrite(filename, M)",
            ),
            (
                "csvwrite(\"out.csv\", [1 2; 3 4], 1, 2);",
                "csvwrite(filename, M, row, col)",
            ),
            (
                "writematrix([1 2; 3 4], \"out.csv\");",
                "writematrix(data, filename)",
            ),
            (
                "writematrix([1 2; 3 4], \"out.csv\", \"Delimiter\", \";\");",
                "writematrix(data, filename, name, optionValue)",
            ),
            ("readmatrix(\"data.csv\");", "M = readmatrix(filename)"),
            (
                "readmatrix(\"data.csv\", \"Range\", \"B2:C4\");",
                "M = readmatrix(filename, name, optionValue)",
            ),
            (
                "dlmwrite(\"out.csv\", [1 2; 3 4]);",
                "bytesWritten = dlmwrite(filename, M)",
            ),
            (
                "dlmwrite(\"out.csv\", [1 2; 3 4], \";\", 1, 2);",
                "bytesWritten = dlmwrite(filename, M, delimiter, row, col)",
            ),
            ("dlmread(\"data.csv\");", "M = dlmread(filename)"),
            (
                "dlmread(\"data.csv\", \",\", 1, 2);",
                "M = dlmread(filename, delimiter, row, col)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_io_json_descriptors() {
        let cases = [
            ("jsondecode(\"[]\");", "value = jsondecode(text)"),
            ("jsonencode(42);", "jsonText = jsonencode(value)"),
            (
                "jsonencode(42, struct('PrettyPrint', true));",
                "jsonText = jsonencode(value, options)",
            ),
            (
                "jsonencode(42, \"PrettyPrint\", true);",
                "jsonText = jsonencode(value, name, optionValue)",
            ),
            (
                "jsonencode(42, \"PrettyPrint\", true, \"ConvertInfAndNaN\", false);",
                "jsonText = jsonencode(value, nameValuePairs...)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_io_console_descriptors() {
        let cases = [
            ("disp(42);", "disp"),
            ("clc();", "clc"),
            ("format('long');", "format"),
            ("input('Value: ');", "input"),
            ("input('s', 'Name: ');", "input"),
        ];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_io_http_descriptors() {
        let cases = [
            ("weboptions();", "options = weboptions()"),
            (
                "weboptions(\"Timeout\", 5);",
                "options = weboptions(name, value, ...)",
            ),
            ("webread(\"https://example.com\");", "data = webread(url)"),
            (
                "webread(\"https://example.com\", \"Timeout\", 5);",
                "data = webread(url, name, value, ...)",
            ),
            (
                "webwrite(\"https://example.com\", \"hello\");",
                "response = webwrite(url, data)",
            ),
            (
                "webwrite(\"https://example.com\", \"hello\", \"Timeout\", 5);",
                "response = webwrite(url, data, name, value, ...)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_io_data_descriptors() {
        let cases = [
            (
                "data.create(\"tmp/demo.data\", struct());",
                "ds = data.create(path, schema)",
            ),
            ("data.open(\"tmp/demo.data\");", "ds = data.open(path)"),
            ("data.exists(\"tmp/demo.data\");", "tf = data.exists(path)"),
            ("data.inspect(\"tmp/demo.data\");", "S = data.inspect(path)"),
            ("Dataset.path(ds);", "path = Dataset.path(ds)"),
            (
                "Dataset.get_attr(ds, \"owner\", 0);",
                "value = Dataset.get_attr(ds, key, defaultValue)",
            ),
            ("DataArray.read(arr);", "X = DataArray.read(arr, sliceSpec)"),
            (
                "DataArray.write(arr, [1 2 3]);",
                "tf = DataArray.write(arr, values)",
            ),
            (
                "DataTransaction.write(tx, \"A\", [1 3], [1 2 3]);",
                "tf = DataTransaction.write(tx, arrayName, sliceSpec, values)",
            ),
            (
                "DataTransaction.commit(tx);",
                "tf = DataTransaction.commit(tx, options)",
            ),
            ("commit(tx);", "tf = commit(tx, options)"),
            (
                "DataTransaction.status(tx);",
                "status = DataTransaction.status(tx)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn completion_detail_uses_io_data_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in [
            "data.create",
            "data.open",
            "Dataset.path",
            "DataArray.read",
            "DataTransaction.commit",
            "commit",
        ] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            let call_head = format!("{builtin}(");
            assert!(
                details.iter().any(|detail| detail.contains(&call_head)),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn signature_help_uses_datetime_duration_descriptors() {
        let cases = [
            ("datetime();", "datetime"),
            ("datetime(2024, 4, 9, 13, 30, 0);", "datetime"),
            (
                "datetime(738000, \"ConvertFrom\", \"datenum\");",
                "datetime",
            ),
            ("year(t);", "year"),
            ("datetime.plus(t, 1);", "datetime.plus"),
            ("duration(1, 30, 45);", "duration"),
            ("duration.plus(d1, d2);", "duration.plus"),
        ];

        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn completion_detail_uses_datetime_duration_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in ["datetime", "year", "duration"] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            let call_head = format!("{builtin}(");
            assert!(
                details.iter().any(|detail| detail.contains(&call_head)),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn signature_help_uses_runtime_lib_descriptors() {
        let cases = [
            ("feval(@sin, 1);", "[varargout] = feval(f, varargin)"),
            ("str2func('@sin');", "fh = str2func(name)"),
            ("func2str(@sin);", "name = func2str(fh)"),
            ("prod([1 2 3], 1);", "B = prod(A, dim)"),
            ("any([1 0 1], 1);", "B = any(A, dim)"),
            ("all([1 1 1], 1);", "B = all(A, dim)"),
            ("warning('msg %d', 1);", "state = warning(message, A...)"),
            (
                "getmethod(classref('Point'), 'move');",
                "fh = getmethod(obj_or_class, name)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn completion_detail_uses_runtime_lib_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in [
            "feval",
            "str2func",
            "func2str",
            "prod",
            "any",
            "all",
            "warning",
            "getmethod",
        ] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            let call_head = format!("{builtin}(");
            assert!(
                details.iter().any(|detail| detail.contains(&call_head)),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn signature_help_uses_io_mat_descriptors() {
        let cases = [
            ("load();", "load"),
            ("load(\"vars.mat\");", "load"),
            ("load(\"vars.mat\", \"A\", \"B\");", "load"),
            ("load(\"vars.mat\", \"-regexp\", \"^A\");", "load"),
            ("save();", "save"),
            ("save(\"vars.mat\");", "save"),
            ("save(\"vars.mat\", \"A\", \"B\");", "save"),
            (
                "save(\"vars.mat\", \"-struct\", \"opts\", \"alpha\");",
                "save",
            ),
        ];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_control_descriptors() {
        let cases = [
            ("tf(1, [1, 1]);", "sys = tf(numerator, denominator)"),
            (
                "tf(1, [1, 1], 0.1);",
                "sys = tf(numerator, denominator, Ts)",
            ),
            ("step(tf(1, [1, 1]));", "y = step(sys)"),
            ("impulse(tf(1, [1, 1]));", "y = impulse(sys)"),
            ("db(10);", "yDb = db(y)"),
            ("db(10, \"power\");", "yDb = db(y, \"power\")"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_rounding_descriptors() {
        let cases = [
            ("fix(-3.7);", "Y = fix(X)"),
            ("round(3.14159, 2);", "Y = round(X, N)"),
            ("ceil(3.14159);", "Y = ceil(X)"),
            ("mod(17, 5);", "R = mod(A, B)"),
            ("rem(-7, 4);", "R = rem(A, B)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_reduction_cumulative_descriptors() {
        let cases = [
            ("sum([1,2,3]);", "sum"),
            ("sum([1,2,3], \"all\");", "sum"),
            ("mean([1,2,3]);", "mean"),
            ("mean([1,2,3], \"all\");", "mean"),
            ("max([1,2,3]);", "max"),
            ("max([1,2,3], [3,2,1]);", "max"),
            ("max([1,2,3], [], 1);", "max"),
            ("min([1,2,3]);", "min"),
            ("min([1,2,3], [3,2,1]);", "min"),
            ("min([1,2,3], [], 1);", "min"),
            ("median([1,2,3]);", "median"),
            ("median([1,2,3], 1);", "median"),
            ("median([1,2,3], \"omitnan\");", "median"),
            ("std([1,2,3]);", "std"),
            ("std([1,2,3], 0, 1);", "std"),
            ("var([1,2,3]);", "var"),
            ("var([1,2,3], 0, 1);", "var"),
            ("var([1,2,3], \"omitnan\");", "var"),
            ("diff([1,2,3]);", "diff"),
            ("diff([1,2,3], 2, 1);", "diff"),
            ("gradient([1,2,3]);", "gradient"),
            ("gradient([1,2,3], 0.5);", "gradient"),
            ("cumsum([1,2,3]);", "cumsum"),
            ("cumsum([1,2,3], \"reverse\");", "cumsum"),
            ("cumprod([1,2,3]);", "cumprod"),
            ("cumprod([1,2,3], \"omitnan\", \"reverse\");", "cumprod"),
            ("cummax([1,2,3]);", "cummax"),
            ("cummin([1,2,3]);", "cummin"),
            ("cummax([1,2,3], \"omitnan\", \"reverse\");", "cummax"),
            ("cummin([1,2,3], \"omitnan\", \"reverse\");", "cummin"),
            ("trapz([1,2,3]);", "trapz"),
            ("trapz([0,1,2], [1,2,3]);", "trapz"),
            ("cumtrapz([1,2,3]);", "cumtrapz"),
            ("cumtrapz([0,1,2], [1,2,3], 2);", "cumtrapz"),
        ];

        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_math_linalg_factor_descriptors() {
        let cases = [
            ("qr([1,2;3,4]);", "R = qr(A)"),
            ("qr([1,2;3,4], \"econ\");", "R = qr(A, option)"),
            (
                "qr([1,2;3,4], \"econ\", \"vector\");",
                "R = qr(A, option1, option2)",
            ),
            ("chol([2,0;0,3]);", "R = chol(A)"),
            ("chol([2,0;0,3], \"lower\");", "R = chol(A, triangle)"),
            ("lu([1,2;3,4]);", "LU = lu(A)"),
            ("lu([1,2;3,4], \"vector\");", "LU = lu(A, pivotMode)"),
            ("svd([1,2;3,4]);", "S = svd(A)"),
            ("svd([1,2;3,4], \"econ\");", "S = svd(A, option)"),
            ("eig([1,2;3,4]);", "d = eig(A)"),
            ("eig([1,2;3,4], \"vector\");", "d = eig(A, options...)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_linalg_ops_descriptors() {
        let cases = [
            ("transpose([1,2;3,4]);", "transpose"),
            ("ctranspose([1,2;3,4]);", "ctranspose"),
            ("trace([1,2;3,4]);", "trace"),
            ("dot([1,2], [3,4]);", "dot"),
            ("dot([1,2], [3,4], 2);", "dot"),
            ("cross([1,0,0], [0,1,0]);", "cross"),
            ("cross([1,0,0], [0,1,0], 2);", "cross"),
            ("mldivide([1,2;3,4], [5;6]);", "mldivide"),
            ("mrdivide([1,2], [1,2;3,4]);", "mrdivide"),
            ("mtimes([1,2;3,4], [5;6]);", "mtimes"),
            ("mpower([1,2;3,4], 2);", "mpower"),
        ];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_math_linalg_solve_descriptors() {
        let cases = [
            ("det([1,2;3,4]);", "d = det(A)"),
            ("inv([1,2;3,4]);", "X = inv(A)"),
            ("rank([1,2;3,4]);", "k = rank(A)"),
            ("rank([1,2;3,4], 1e-6);", "k = rank(A, tol)"),
            ("rcond([1,2;3,4]);", "c = rcond(A)"),
            ("pinv([1,2;3,4]);", "X = pinv(A)"),
            ("pinv([1,2;3,4], 1e-6);", "X = pinv(A, tol)"),
            ("cond([1,2;3,4]);", "c = cond(A)"),
            ("cond([1,2;3,4], \"fro\");", "c = cond(A, p)"),
            ("norm([1,2;3,4]);", "n = norm(A)"),
            ("norm([1,2;3,4], \"fro\");", "n = norm(A, p)"),
            ("linsolve([1,2;3,4], [5;6]);", "X = linsolve(A, B)"),
            (
                "linsolve([1,2;3,4], [5;6], struct(\"LT\", true));",
                "X = linsolve(A, B, opts)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_linalg_structure_descriptors() {
        let cases = [
            ("bandwidth([1,2;3,4]);", "[lower, upper] = bandwidth(A)"),
            (
                "bandwidth([1,2;3,4], \"lower\");",
                "b = bandwidth(A, selector)",
            ),
            ("issymmetric([1,2;2,1]);", "tf = issymmetric(A)"),
            (
                "issymmetric([1,2;2,1], \"skew\");",
                "tf = issymmetric(A, option)",
            ),
            (
                "issymmetric([1,2;2,1], \"skew\", 1e-9);",
                "tf = issymmetric(A, flag, tol)",
            ),
            ("ishermitian([1,2;2,1]);", "tf = ishermitian(A)"),
            (
                "ishermitian([1,2;2,1], \"skew\");",
                "tf = ishermitian(A, option)",
            ),
            (
                "ishermitian([1,2;2,1], \"skew\", 1e-9);",
                "tf = ishermitian(A, flag, tol)",
            ),
            ("symrcm([1,2;2,1]);", "p = symrcm(A)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_poly_descriptors() {
        let cases = [
            ("polyfit([0,1],[1,2],1);", "p = polyfit(X, Y, n)"),
            (
                "polyfit([0,1],[1,2],1,[1,1]);",
                "p = polyfit(X, Y, n, weights)",
            ),
            ("polyval([1,0,-1],[0,1]);", "y = polyval(p, x)"),
            ("polyval([1,0,-1],[0,1],struct());", "y = polyval(p, x, S)"),
            (
                "polyval([1,0,-1],[0,1],[],[0,1]);",
                "y = polyval(p, x, S, mu)",
            ),
            ("roots([1,0,-1]);", "r = roots(c)"),
            ("polyint([1,2,3]);", "q = polyint(p)"),
            ("polyint([1,2,3], 4);", "q = polyint(p, k)"),
            ("polyder([1,2,3]);", "d = polyder(p)"),
            ("polyder([1,2,3], [1,1]);", "d = polyder(a, b)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_interpolation_descriptors() {
        let cases = [
            ("interp1([10,20,40], [1.5,2.5]);", "Vq = interp1(Y, Xq)"),
            (
                "interp1([1,2,3], [10,20,40], [1.5,2.5], \"nearest\", \"extrap\");",
                "Vq = interp1(X, Y, Xq, method, extrap)",
            ),
            ("interp2([1,2;3,4], 1.5, 1.5);", "Vq = interp2(Z, Xq, Yq)"),
            (
                "interp2([1,2], [1,2], [1,2;3,4], 1.5, 1.5, \"nearest\");",
                "Vq = interp2(X, Y, Z, Xq, Yq, method)",
            ),
            ("spline([1,2,3], [1,4,9]);", "pp = spline(X, Y)"),
            (
                "pchip([1,2,3], [1,4,9], [1.5,2.5]);",
                "Vq = pchip(X, Y, Xq)",
            ),
            ("ppval(struct(), [1.5,2.5]);", "Vq = ppval(pp, Xq)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_ode_descriptors() {
        let cases = [
            ("ode23(1, [0 1], 1);", "y = ode23(odefun, tspan, y0)"),
            (
                "ode45(1, [0 1], 1, struct());",
                "y = ode45(odefun, tspan, y0, options)",
            ),
            ("ode15s(1, [0 1], 1);", "[t, y] = ode15s(odefun, tspan, y0)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).unwrap_or_else(|| {
                panic!(
                    "signature help missing for `{text}`; status={}",
                    analysis.status_message()
                )
            });
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_optim_descriptors() {
        let cases = [
            ("fzero(1, 0);", "x = fzero(fun, x0)"),
            ("fsolve(1, [0;0]);", "x = fsolve(fun, x0)"),
            (
                "optimset(\"TolX\", 1e-8);",
                "options = optimset(name, value, ...)",
            ),
            ("fminbnd(1, 0, 1);", "x = fminbnd(fun, x1, x2)"),
            ("integral(1, 0, 1);", "q = integral(fun, xmin, xmax)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).unwrap_or_else(|| {
                panic!(
                    "signature help missing for `{text}`; status={}",
                    analysis.status_message()
                )
            });
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_image_color_descriptors() {
        let cases = [
            ("gray2rgb([0.1,0.2;0.3,0.4]);", "RGB = gray2rgb(I)"),
            ("rgb2gray(ones(2,2,3));", "I = rgb2gray(RGB)"),
            ("rgb2hsv(ones(2,2,3));", "HSV = rgb2hsv(RGB)"),
            ("hsv2rgb(ones(2,2,3));", "RGB = hsv2rgb(HSV)"),
            ("ind2rgb([1,2;2,1],[1,0,0;0,1,0]);", "RGB = ind2rgb(X, map)"),
            ("rgb2lab(ones(2,2,3));", "LAB = rgb2lab(RGB, Name=Value)"),
            (
                "lab2rgb(ones(2,2,3));",
                "RGB = lab2rgb(LAB, OutputType=type)",
            ),
            ("im2double(ones(2,2));", "J = im2double(I)"),
            ("im2uint8(ones(2,2));", "J = im2uint8(I)"),
            ("im2uint16(ones(2,2));", "J = im2uint16(I)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn completion_detail_uses_math_linalg_factor_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);
        let qr_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("qr"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            qr_candidates.iter().any(|detail| detail.contains("qr(")),
            "expected descriptor signature detail for qr completion, got {:?}",
            qr_candidates
        );

        let chol_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("chol"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            chol_candidates
                .iter()
                .any(|detail| detail.contains("chol(")),
            "expected descriptor signature detail for chol completion, got {:?}",
            chol_candidates
        );

        let lu_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("lu"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            lu_candidates.iter().any(|detail| detail.contains("lu(")),
            "expected descriptor signature detail for lu completion, got {:?}",
            lu_candidates
        );

        let svd_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("svd"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            svd_candidates.iter().any(|detail| detail.contains("svd(")),
            "expected descriptor signature detail for svd completion, got {:?}",
            svd_candidates
        );

        let eig_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("eig"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            eig_candidates.iter().any(|detail| detail.contains("eig(")),
            "expected descriptor signature detail for eig completion, got {:?}",
            eig_candidates
        );
    }

    #[test]
    fn completion_detail_uses_math_linalg_ops_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        let transpose_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("transpose"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            transpose_candidates
                .iter()
                .any(|detail| detail.contains("transpose(")),
            "expected descriptor signature detail for transpose completion, got {:?}",
            transpose_candidates
        );

        let ctranspose_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("ctranspose"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            ctranspose_candidates
                .iter()
                .any(|detail| detail.contains("ctranspose(")),
            "expected descriptor signature detail for ctranspose completion, got {:?}",
            ctranspose_candidates
        );

        let trace_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("trace"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            trace_candidates
                .iter()
                .any(|detail| detail.contains("trace(")),
            "expected descriptor signature detail for trace completion, got {:?}",
            trace_candidates
        );

        let dot_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("dot"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            dot_candidates.iter().any(|detail| detail.contains("dot(")),
            "expected descriptor signature detail for dot completion, got {:?}",
            dot_candidates
        );

        let cross_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("cross"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            cross_candidates
                .iter()
                .any(|detail| detail.contains("cross(")),
            "expected descriptor signature detail for cross completion, got {:?}",
            cross_candidates
        );

        let mldivide_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("mldivide"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            mldivide_candidates
                .iter()
                .any(|detail| detail.contains("mldivide(")),
            "expected descriptor signature detail for mldivide completion, got {:?}",
            mldivide_candidates
        );

        let mrdivide_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("mrdivide"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            mrdivide_candidates
                .iter()
                .any(|detail| detail.contains("mrdivide(")),
            "expected descriptor signature detail for mrdivide completion, got {:?}",
            mrdivide_candidates
        );

        let mtimes_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("mtimes"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            mtimes_candidates
                .iter()
                .any(|detail| detail.contains("mtimes(")),
            "expected descriptor signature detail for mtimes completion, got {:?}",
            mtimes_candidates
        );

        let mpower_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("mpower"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            mpower_candidates
                .iter()
                .any(|detail| detail.contains("mpower(")),
            "expected descriptor signature detail for mpower completion, got {:?}",
            mpower_candidates
        );
    }

    #[test]
    fn completion_detail_uses_math_linalg_solve_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in [
            "det", "inv", "rank", "rcond", "pinv", "cond", "norm", "linsolve",
        ] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            assert!(
                details
                    .iter()
                    .any(|detail| detail.contains(&format!("{builtin}("))),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn completion_detail_uses_math_linalg_structure_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in ["bandwidth", "issymmetric", "ishermitian", "symrcm"] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            assert!(
                details
                    .iter()
                    .any(|detail| detail.contains(&format!("{builtin}("))),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn completion_detail_uses_math_poly_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in ["polyfit", "polyval", "roots", "polyint", "polyder"] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            assert!(
                details
                    .iter()
                    .any(|detail| detail.contains(&format!("{builtin}("))),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn completion_detail_uses_math_interpolation_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in ["interp1", "interp2", "spline", "pchip", "ppval"] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            assert!(
                details
                    .iter()
                    .any(|detail| detail.contains(&format!("{builtin}("))),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn completion_detail_uses_math_ode_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in ["ode23", "ode45", "ode15s"] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            assert!(
                details
                    .iter()
                    .any(|detail| detail.contains(&format!("{builtin}("))),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn completion_detail_uses_math_optim_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in ["fzero", "fsolve", "optimset", "fminbnd", "integral"] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            assert!(
                details
                    .iter()
                    .any(|detail| detail.contains(&format!("{builtin}("))),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn completion_detail_uses_image_color_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in [
            "gray2rgb",
            "rgb2gray",
            "rgb2hsv",
            "hsv2rgb",
            "ind2rgb",
            "rgb2lab",
            "lab2rgb",
            "im2double",
            "im2uint8",
            "im2uint16",
        ] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            assert!(
                details
                    .iter()
                    .any(|detail| detail.contains(&format!("{builtin}("))),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn signature_help_uses_image_filter_descriptors() {
        let cases = [
            ("filter2([1,0;-1,0], rand(4,4));", "B = filter2(h, X)"),
            (
                "filter2([1,0;-1,0], rand(4,4), \"full\", \"conv\");",
                "B = filter2(h, X, options...)",
            ),
            ("imfilter(rand(4,4), [1,0;-1,0]);", "B = imfilter(A, H)"),
            (
                "imfilter(rand(4,4), [1,0;-1,0], \"same\", \"replicate\");",
                "B = imfilter(A, H, options...)",
            ),
            ("fspecial(\"gaussian\");", "H = fspecial(type)"),
            (
                "fspecial(\"gaussian\", [5, 5], 1.0);",
                "H = fspecial(type, arg1, arg2)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn completion_detail_uses_image_filter_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in ["filter2", "imfilter", "fspecial"] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            assert!(
                details
                    .iter()
                    .any(|detail| detail.contains(&format!("{builtin}("))),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn signature_help_uses_image_io_descriptors() {
        let cases = [
            (
                "imread('img.png');",
                vec![
                    "I = imread(filename)",
                    "[I, map] = imread(filename)",
                    "[I, map, alpha] = imread(filename)",
                ],
            ),
            (
                "imread('img', 'png');",
                vec![
                    "I = imread(filename, fmt)",
                    "[I, map] = imread(filename, fmt)",
                    "[I, map, alpha] = imread(filename, fmt)",
                ],
            ),
        ];

        for (text, expected_labels) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            for expected_label in expected_labels {
                assert!(
                    labels.contains(&expected_label),
                    "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                    labels
                );
            }
        }
    }

    #[test]
    fn completion_detail_uses_image_io_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        let details: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("imread"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            details.iter().any(|detail| detail.contains("imread(")),
            "expected descriptor signature detail for imread completion, got {:?}",
            details
        );
    }

    #[test]
    fn completion_detail_uses_math_reduction_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        let all_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("all"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            all_candidates.iter().any(|detail| detail.contains("all(")),
            "expected descriptor signature detail for all completion, got {:?}",
            all_candidates
        );

        let any_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("any"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            any_candidates.iter().any(|detail| detail.contains("any(")),
            "expected descriptor signature detail for any completion, got {:?}",
            any_candidates
        );

        let nnz_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("nnz"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            nnz_candidates.iter().any(|detail| detail.contains("nnz(")),
            "expected descriptor signature detail for nnz completion, got {:?}",
            nnz_candidates
        );

        let prod_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("prod"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            prod_candidates
                .iter()
                .any(|detail| detail.contains("prod(")),
            "expected descriptor signature detail for prod completion, got {:?}",
            prod_candidates
        );

        let cumsum_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("cumsum"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            cumsum_candidates
                .iter()
                .any(|detail| detail.contains("cumsum(")),
            "expected descriptor signature detail for cumsum completion, got {:?}",
            cumsum_candidates
        );

        let cumprod_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("cumprod"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            cumprod_candidates
                .iter()
                .any(|detail| detail.contains("cumprod(")),
            "expected descriptor signature detail for cumprod completion, got {:?}",
            cumprod_candidates
        );

        let cummax_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("cummax"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            cummax_candidates
                .iter()
                .any(|detail| detail.contains("cummax(")),
            "expected descriptor signature detail for cummax completion, got {:?}",
            cummax_candidates
        );

        let cummin_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("cummin"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            cummin_candidates
                .iter()
                .any(|detail| detail.contains("cummin(")),
            "expected descriptor signature detail for cummin completion, got {:?}",
            cummin_candidates
        );

        let sum_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("sum"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            sum_candidates.iter().any(|detail| detail.contains("sum(")),
            "expected descriptor signature detail for sum completion, got {:?}",
            sum_candidates
        );

        let mean_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("mean"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            mean_candidates
                .iter()
                .any(|detail| detail.contains("mean(")),
            "expected descriptor signature detail for mean completion, got {:?}",
            mean_candidates
        );

        let max_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("max"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            max_candidates.iter().any(|detail| detail.contains("max(")),
            "expected descriptor signature detail for max completion, got {:?}",
            max_candidates
        );

        let min_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("min"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            min_candidates.iter().any(|detail| detail.contains("min(")),
            "expected descriptor signature detail for min completion, got {:?}",
            min_candidates
        );

        let median_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("median"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            median_candidates
                .iter()
                .any(|detail| detail.contains("median(")),
            "expected descriptor signature detail for median completion, got {:?}",
            median_candidates
        );

        let std_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("std"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            std_candidates.iter().any(|detail| detail.contains("std(")),
            "expected descriptor signature detail for std completion, got {:?}",
            std_candidates
        );

        let var_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("var"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            var_candidates.iter().any(|detail| detail.contains("var(")),
            "expected descriptor signature detail for var completion, got {:?}",
            var_candidates
        );

        let diff_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("diff"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            diff_candidates
                .iter()
                .any(|detail| detail.contains("diff(")),
            "expected descriptor signature detail for diff completion, got {:?}",
            diff_candidates
        );

        let gradient_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("gradient"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            gradient_candidates
                .iter()
                .any(|detail| detail.contains("gradient(")),
            "expected descriptor signature detail for gradient completion, got {:?}",
            gradient_candidates
        );

        let trapz_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("trapz"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            trapz_candidates
                .iter()
                .any(|detail| detail.contains("trapz(")),
            "expected descriptor signature detail for trapz completion, got {:?}",
            trapz_candidates
        );

        let cumtrapz_candidates: Vec<String> = completions
            .iter()
            .filter(|item| item.label.eq_ignore_ascii_case("cumtrapz"))
            .map(|item| item.detail.clone().unwrap_or_default())
            .collect();
        assert!(
            cumtrapz_candidates
                .iter()
                .any(|detail| detail.contains("cumtrapz(")),
            "expected descriptor signature detail for cumtrapz completion, got {:?}",
            cumtrapz_candidates
        );
    }

    #[test]
    fn signature_help_uses_math_elementwise_complex_descriptors() {
        let cases = [
            ("abs(-3.7);", "Y = abs(X)"),
            ("angle(1+2i);", "theta = angle(X)"),
            ("conj(1+2i);", "Y = conj(X)"),
            ("real(1+2i);", "Y = real(X)"),
            ("imag(1+2i);", "Y = imag(X)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_elementwise_binary_descriptors() {
        let cases = [
            ("plus(1, 2);", "C = plus(A, B)"),
            (
                "plus(1, 2, \"like\", 1);",
                "C = plus(A, B, \"like\", prototype)",
            ),
            ("minus(5, 3);", "C = minus(A, B)"),
            (
                "times(2, 4, \"like\", 1);",
                "C = times(A, B, \"like\", prototype)",
            ),
            ("rdivide(8, 2);", "C = rdivide(A, B)"),
            ("ldivide(2, 8);", "C = ldivide(A, B)"),
            ("power(2, 8);", "C = power(A, B)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_elementwise_transcendental_descriptors() {
        let cases = [
            ("exp(1);", "Y = exp(X)"),
            ("expm1(1);", "Y = expm1(X)"),
            ("log(2);", "Y = log(X)"),
            ("log1p(2);", "Y = log1p(X)"),
            ("log2(8);", "Y = log2(X)"),
            ("log2(8);", "[F,E] = log2(X)"),
            ("log10(100);", "Y = log10(X)"),
            ("sqrt(4);", "Y = sqrt(X)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_elementwise_cast_descriptors() {
        let cases = [
            ("double(1);", "Y = double(X)"),
            (
                "double(1, \"like\", gpuArray(1));",
                "Y = double(X, \"like\", prototype)",
            ),
            ("single(1);", "Y = single(X)"),
            (
                "single(1, \"like\", gpuArray(1));",
                "Y = single(X, \"like\", prototype)",
            ),
            ("int32(1);", "Y = int32(X)"),
            ("uint16(1);", "Y = uint16(X)"),
            ("uint8(1);", "Y = uint8(X)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_elementwise_special_descriptors() {
        let cases = [
            ("complex(1);", "complex"),
            ("complex(1, 2);", "complex"),
            ("factorial(5);", "factorial"),
            ("factorial(5, \"like\", 1);", "factorial"),
            ("gamma(5);", "gamma"),
            ("gammaln(5);", "gammaln"),
            ("hypot(3, 4);", "hypot"),
            ("nextpow2(9);", "nextpow2"),
            ("pow2(3);", "pow2"),
            ("pow2(1.5, 2);", "pow2"),
            ("sign(-7);", "sign"),
        ];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_math_trigonometry_descriptors() {
        let cases = [
            ("asin(0.5);", "asin"),
            ("acos(0.5);", "acos"),
            ("atan(1);", "atan"),
            ("atan(1, \"like\", 1);", "atan"),
            ("sinh(1);", "sinh"),
            ("cosh(1);", "cosh"),
            ("tanh(1);", "tanh"),
            ("asinh(1);", "asinh"),
            ("acosh(2);", "acosh"),
            ("atanh(0.5);", "atanh"),
            ("sind(90);", "sind"),
            ("cosd(90);", "cosd"),
            ("tand(45);", "tand"),
            ("deg2rad(180);", "deg2rad"),
            ("rad2deg(pi);", "rad2deg"),
            ("sin(1);", "sin"),
            ("sin(1, \"like\", 1);", "sin"),
            ("cos(1);", "cos"),
            ("cos(1, \"like\", 1);", "cos"),
            ("tan(1);", "tan"),
            ("tan(1, \"like\", 1);", "tan"),
            ("atan2(1, 1);", "atan2"),
        ];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_math_signal_waveform_descriptors() {
        let cases = [
            ("hann(8);", "w = hann(n)"),
            ("hann(8, \"periodic\");", "w = hann(n, sampling)"),
            ("hamming(8);", "w = hamming(n)"),
            ("hamming(8, \"periodic\");", "w = hamming(n, sampling)"),
            ("blackman(8);", "w = blackman(n)"),
            ("blackman(8, \"single\");", "w = blackman(n, precision)"),
            ("sinc(0.5);", "Y = sinc(X)"),
            ("square(0.5);", "Y = square(t)"),
            ("square(0.5, 25);", "Y = square(t, duty)"),
            ("sawtooth(0.5);", "Y = sawtooth(t)"),
            ("sawtooth(0.5, 0.5);", "Y = sawtooth(t, xmax)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_signal_convolution_descriptors() {
        let cases = [
            ("conv([1,2], [1,1]);", "C = conv(A, B)"),
            ("conv([1,2], [1,1], \"same\");", "C = conv(A, B, shape)"),
            ("conv2([1,2;3,4], [1,1;1,1]);", "C = conv2(A, B)"),
            (
                "conv2([1,2;3,4], [1,1;1,1], \"valid\");",
                "C = conv2(A, B, shape)",
            ),
            (
                "conv2([1;2], [1,2], [1,2;3,4]);",
                "C = conv2(hcol, hrow, A)",
            ),
            (
                "conv2([1;2], [1,2], [1,2;3,4], \"same\");",
                "C = conv2(hcol, hrow, A, shape)",
            ),
            (
                "deconv([1,3,3,1], [1,1]);",
                "Q = deconv(numerator, denominator)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_signal_filter_descriptors() {
        let cases = [
            ("filter([1, 1], [1], [1, 2, 3]);", "y = filter(b, a, x)"),
            (
                "filter([1, 1], [1], [1, 2, 3], [0]);",
                "y = filter(b, a, x, zi)",
            ),
            (
                "filter([1, 1], [1], [1, 2, 3], [], 2);",
                "y = filter(b, a, x, zi, dim)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_math_fft_core_descriptors() {
        let cases = [
            ("fft([1,2,3]);", "Y = fft(X)"),
            ("fft([1,2,3], 8);", "Y = fft(X, N)"),
            ("fft([1,2,3], 8, 2);", "Y = fft(X, N, DIM)"),
            ("fft2([1,2;3,4]);", "Y = fft2(X)"),
            ("fft2([1,2;3,4], [8, 8]);", "Y = fft2(X, SIZE)"),
            ("fft2([1,2;3,4], 8, 8);", "Y = fft2(X, M, N)"),
            ("fftn(reshape(1:8, [2,2,2]));", "Y = fftn(X)"),
            ("fftn(reshape(1:8, [2,2,2]), [4,4,4]);", "Y = fftn(X, SIZE)"),
            ("fftshift([1,2,3,4]);", "Y = fftshift(X)"),
            ("fftshift([1,2,3,4], 1);", "Y = fftshift(X, DIM)"),
            ("ifft([1,2,3]);", "Y = ifft(X)"),
            ("ifft([1,2,3], 8);", "Y = ifft(X, N)"),
            ("ifft([1,2,3], \"symmetric\");", "Y = ifft(X, symflag)"),
            ("ifft([1,2,3], 8, 2);", "Y = ifft(X, N, DIM)"),
            (
                "ifft([1,2,3], 8, \"nonsymmetric\");",
                "Y = ifft(X, N, symflag)",
            ),
            (
                "ifft([1,2,3], 8, 2, \"symmetric\");",
                "Y = ifft(X, N, DIM, symflag)",
            ),
            ("ifft2([1,2;3,4]);", "Y = ifft2(X)"),
            ("ifft2([1,2;3,4], [8, 8]);", "Y = ifft2(X, SIZE)"),
            ("ifft2([1,2;3,4], 8, 8);", "Y = ifft2(X, M, N)"),
            ("ifft2([1,2;3,4], \"symmetric\");", "Y = ifft2(X, symflag)"),
            (
                "ifft2([1,2;3,4], [8, 8], \"nonsymmetric\");",
                "Y = ifft2(X, SIZE, symflag)",
            ),
            (
                "ifft2([1,2;3,4], 8, 8, \"symmetric\");",
                "Y = ifft2(X, M, N, symflag)",
            ),
            ("ifftn(reshape(1:8, [2,2,2]));", "Y = ifftn(X)"),
            (
                "ifftn(reshape(1:8, [2,2,2]), [4,4,4]);",
                "Y = ifftn(X, SIZE)",
            ),
            (
                "ifftn(reshape(1:8, [2,2,2]), \"symmetric\");",
                "Y = ifftn(X, symflag)",
            ),
            (
                "ifftn(reshape(1:8, [2,2,2]), [4,4,4], \"nonsymmetric\");",
                "Y = ifftn(X, SIZE, symflag)",
            ),
            ("ifftshift([1,2,3,4]);", "Y = ifftshift(X)"),
            ("ifftshift([1,2,3,4], 1);", "Y = ifftshift(X, DIM)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).expect("signature help");
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_structs_core_descriptors() {
        let cases = [
            ("fieldnames(struct());", "fieldnames"),
            ("getfield(struct('a', 1), 'a');", "getfield"),
            ("isfield(struct('a', 1), 'a');", "isfield"),
            ("orderfields(struct('a', 1));", "orderfields"),
            ("setfield(struct(), 'a', 1);", "setfield"),
            ("struct('a', 1);", "struct"),
            ("rmfield(struct('a', 1), 'a');", "rmfield"),
        ];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_cells_core_descriptors() {
        let cases = [
            ("cell();", "cell"),
            ("cell(2, 3);", "cell"),
            ("cell([2, 3], \"like\", zeros(1));", "cell"),
            ("cell2mat(cell(1, 1));", "cell2mat"),
            ("cellfun(@length, cell(1, 1));", "cellfun"),
            ("cellstr(\"RunMat\");", "cellstr"),
        ];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_containers_map_descriptors() {
        let cases = [
            ("containers.Map();", "containers.Map"),
            ("containers.Map.keys(m);", "containers.Map.keys"),
            ("containers.Map.values(m);", "containers.Map.values"),
            ("containers.Map.values(m, {'a'});", "containers.Map.values"),
            ("containers.Map.isKey(m, \"a\");", "containers.Map.isKey"),
            ("containers.Map.remove(m, \"a\");", "containers.Map.remove"),
            (
                "containers.Map.subsref(m, \"()\", {\"a\"});",
                "containers.Map.subsref",
            ),
            (
                "containers.Map.subsasgn(m, \"()\", {\"a\"}, 1);",
                "containers.Map.subsasgn",
            ),
        ];
        for (text, identity) in cases {
            assert_catalog_signature_projection(text, identity);
        }
    }

    #[test]
    fn signature_help_uses_confusionmat_descriptor() {
        let text = "confusionmat(group,grouphat);";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let sig = signature_help_at(text, &analysis, &position).expect("signature help");
        let labels: Vec<&str> = sig
            .signatures
            .iter()
            .map(|signature| signature.label.as_str())
            .collect();
        assert!(labels.contains(&"[C,order] = confusionmat(___)"));
    }

    #[test]
    fn completion_detail_uses_containers_map_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in [
            "containers.Map",
            "containers.Map.keys",
            "containers.Map.values",
            "containers.Map.isKey",
            "containers.Map.remove",
            "containers.Map.subsref",
            "containers.Map.subsasgn",
        ] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            let call_head = format!("{builtin}(");
            assert!(
                details.iter().any(|detail| detail.contains(&call_head)),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn signature_help_uses_strings_core_descriptors() {
        let cases = [
            ("strcmp(\"a\", \"b\");", "tf = strcmp(A, B)"),
            ("strcmpi(\"a\", \"b\");", "tf = strcmpi(A, B)"),
            ("strncmp(\"abc\", \"abd\", 2);", "tf = strncmp(A, B, N)"),
            ("strings();", "S = strings()"),
            ("strings([2,3]);", "S = strings(sz)"),
            ("strlength(\"abc\");", "L = strlength(str)"),
            ("str2double(\"1.0\");", "X = str2double(str)"),
            ("char(65);", "C = char(X)"),
            ("num2str(42);", "txt = num2str(A)"),
            ("sprintf(\"%d\", 1);", "txt = sprintf(formatSpec, A...)"),
            ("compose(\"v=%d\", 1);", "S = compose(formatSpec, A...)"),
            ("string(42);", "S = string(X)"),
            ("string(\"%d\", 1);", "S = string(formatSpec, A...)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position)
                .unwrap_or_else(|| panic!("expected signature help for strings-core case: {text}"));
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn completion_detail_uses_strings_core_descriptors() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);

        for builtin in [
            "strcmp",
            "strcmpi",
            "strncmp",
            "strings",
            "strlength",
            "str2double",
            "char",
            "num2str",
            "sprintf",
            "compose",
            "string",
        ] {
            let details: Vec<String> = completions
                .iter()
                .filter(|item| item.label.eq_ignore_ascii_case(builtin))
                .map(|item| item.detail.clone().unwrap_or_default())
                .collect();
            assert!(
                details
                    .iter()
                    .any(|detail| detail.contains(&format!("{builtin}("))),
                "expected descriptor signature detail for {builtin} completion, got {:?}",
                details
            );
        }
    }

    #[test]
    fn signature_help_uses_strings_search_descriptors() {
        let cases = [
            ("contains(\"runmat\", \"mat\");", "tf = contains(str, pat)"),
            (
                "startsWith(\"runmat\", \"run\", true);",
                "tf = startsWith(str, pat, ignoreCase)",
            ),
            (
                "endsWith(\"runmat\", \"mat\", \"IgnoreCase\", true);",
                "tf = endsWith(str, pat, \"IgnoreCase\", value)",
            ),
            ("strfind(\"runmat\", \"mat\");", "idx = strfind(str, pat)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).unwrap_or_else(|| {
                panic!("expected signature help for strings-search case: {text}")
            });
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_strings_transform_case_descriptors() {
        let cases = [
            ("lower(\"RunMat\");", "out = lower(str)"),
            ("upper(\"RunMat\");", "out = upper(str)"),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).unwrap_or_else(|| {
                panic!("expected signature help for strings-transform case: {text}")
            });
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_strings_transform_trim_descriptors() {
        let cases = [(
            "strip(\"..RunMat..\", \"both\", \".\");",
            "out = strip(str, direction, stripCharacters)",
        )];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).unwrap_or_else(|| {
                panic!("expected signature help for strings-transform trim case: {text}")
            });
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_strings_transform_replace_descriptors() {
        let cases = [
            (
                "strrep(\"runmat\", \"run\", \"RUN\");",
                "newStr = strrep(str, old, new)",
            ),
            (
                "replace(\"runmat\", \"run\", \"RUN\");",
                "newText = replace(str, oldText, newText)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).unwrap_or_else(|| {
                panic!("expected signature help for strings-transform replace case: {text}")
            });
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn signature_help_uses_strings_transform_erase_concat_descriptors() {
        let cases = [
            (
                "erase(\"runmat\", \"run\");",
                "newStr = erase(str, pattern)",
            ),
            ("strcat(\"run\", \"mat\");", "out = strcat(str1, str2, ...)"),
            ("join(\"runmat\");", "out = join(str)"),
            ("pad(\"runmat\");", "out = pad(str)"),
            ("split(\"a b c\");", "newStr = split(str)"),
            (
                "strsplit(\"a,b\", \",\");",
                "[parts, matches] = strsplit(str, delimiter)",
            ),
            (
                "extractBetween(\"A[GPU]B\", \"[\", \"]\");",
                "newText = extractBetween(str, start, end)",
            ),
            (
                "eraseBetween(\"A[GPU]B\", \"[\", \"]\");",
                "newText = eraseBetween(str, start, end)",
            ),
        ];

        for (text, expected_label) in cases {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let position = lsp_types::Position::new(0, 0);
            let sig = signature_help_at(text, &analysis, &position).unwrap_or_else(|| {
                panic!("expected signature help for strings-transform case: {text}")
            });
            let labels: Vec<&str> = sig.signatures.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.contains(&expected_label),
                "expected descriptor-backed signature '{expected_label}' for {text}, got {:?}",
                labels
            );
        }
    }

    #[test]
    fn completion_detail_prefers_descriptor_signature_label() {
        let text = "x = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let position = lsp_types::Position::new(0, 0);
        let completions = completion_at(text, &analysis, &position);
        let zeros = completions
            .iter()
            .find(|item| item.label.eq_ignore_ascii_case("zeros"))
            .expect("zeros completion item");
        let detail = zeros.detail.clone().unwrap_or_default();
        assert!(
            detail.contains("zeros("),
            "expected descriptor signature detail for zeros completion, got {detail}"
        );
    }

    #[test]
    fn hover_includes_inferred_tensor_shape() {
        let text = "x = 0:1:100; y = sin(x);";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let x_offset = text.find('x').expect("x offset");
        let y_offset = text.find('y').expect("y offset");
        let x_position = offset_to_position(text, x_offset);
        let y_position = offset_to_position(text, y_offset);

        let x_hover = hover_at(text, &analysis, &x_position).expect("x hover");
        let y_hover = hover_at(text, &analysis, &y_position).expect("y hover");

        let extract = |hover: Hover| match hover.contents {
            lsp_types::HoverContents::Markup(markup) => markup.value,
            other => panic!("unexpected hover contents {other:?}"),
        };

        let x_value = extract(x_hover);
        let y_value = extract(y_hover);
        assert!(x_value.contains("double"), "unexpected x hover {x_value}");
        assert!(x_value.contains("1 x 101"), "unexpected x hover {x_value}");
        assert!(y_value.contains("double"), "unexpected y hover {y_value}");
        assert!(y_value.contains("1 x 101"), "unexpected y hover {y_value}");
    }

    #[test]
    fn wave_reproduction_uses_program_point_facts() {
        let text = crate::core::semantic::fixtures::WAVE_SOURCE;
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        assert!(analysis.syntax_error.is_none());
        assert!(analysis.lowering_error.is_none());

        for (needle, dimensions) in crate::core::semantic::fixtures::WAVE_EXPECTATIONS {
            let offset = text
                .find(needle)
                .unwrap_or_else(|| panic!("missing {needle}"));
            let position = offset_to_position(text, offset);
            let information = quick_information_at(text, &analysis, &position)
                .unwrap_or_else(|| panic!("missing quick information for {needle}"));
            let fact = information
                .observation
                .and_then(|observation| observation.fact)
                .unwrap_or_else(|| panic!("missing fact for {needle}"));
            assert_eq!(
                fact.shape.known_dims(),
                Some(dimensions.iter().copied().map(Some).collect()),
                "wrong shape for {needle}: {fact:?}"
            );
        }
    }

    #[test]
    fn hover_includes_inferred_tensor_shape_for_negative_range() {
        let text = "XRange = -2:0.02:2;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let x_offset = text.find("XRange").expect("XRange offset");
        let x_position = offset_to_position(text, x_offset);

        let x_hover = hover_at(text, &analysis, &x_position).expect("XRange hover");
        let x_value = match x_hover.contents {
            lsp_types::HoverContents::Markup(markup) => markup.value,
            other => panic!("unexpected hover contents {other:?}"),
        };

        assert!(x_value.contains("double"), "unexpected hover {x_value}");
        assert!(x_value.contains("1 x 201"), "unexpected hover {x_value}");
    }

    #[test]
    fn debug_full_script_analysis_errors() {
        let text = r#"% Grid
XRange = -2:0.02:2;
YRange = -2:0.02:2;
[X, Y] = meshgrid(XRange, YRange);

% Constants
T = 5;
FPS = 30;
dT = 1/FPS;
noise = 1.0;

for t = 0:dT:T

    % Elementwise math
    R = sqrt(X.^2 + Y.^2) + 1e-6;
    W = sin(t*R) ./ R;
    Z = W + rand(W) * noise;

    scatter3(X, Y, Z);

    % Hold the frame for dT seconds
    pause(dT);

end
"#;
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        if let Some(err) = &analysis.syntax_error {
            eprintln!("syntax error: {} at {}", err.message, err.position);
        }
        if let Some(err) = &analysis.lowering_error {
            eprintln!("lowering error: {err}");
        }
        if let Some(err) = &analysis.compile_error {
            eprintln!("compile error: {err}");
        }
        assert!(analysis.syntax_error.is_none());
        assert!(analysis.lowering_error.is_none());
    }

    #[test]
    fn hover_full_script_range_shape() {
        let text = r#"% Grid
XRange = -2:0.02:2;
YRange = -2:0.02:2;
[X, Y] = meshgrid(XRange, YRange);

% Constants
T = 5;
FPS = 30;
dT = 1/FPS;
noise = 1.0;

for t = 0:dT:T

    % Elementwise math
    R = sqrt(X.^2 + Y.^2) + 1e-6;
    W = sin(t*R) ./ R;
    Z = W + rand(W) * noise;

    scatter3(X, Y, Z);

    % Hold the frame for dT seconds
    pause(dT);

end
"#;
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let x_offset = text.find("XRange").expect("XRange offset");
        let x_position = offset_to_position(text, x_offset);
        let hover = hover_at(text, &analysis, &x_position).expect("hover result");
        let value = match hover.contents {
            lsp_types::HoverContents::Markup(markup) => markup.value,
            other => panic!("unexpected hover {other:?}"),
        };
        assert!(value.contains("1 x 201"), "unexpected hover {value}");
    }

    #[test]
    fn diagnostics_include_shape_lints() {
        let text = "a = ones(2,3); b = ones(4,2); c = a * b;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let diags = diagnostics_for_document(text, &analysis);
        let diag = diags.iter().find(|d| match &d.code {
            Some(lsp_types::NumberOrString::String(code)) => code == "RM-TYPE-MATMUL",
            _ => false,
        });
        let diag = diag.expect("expected matmul lint");
        assert!(
            diag.message.contains("inner dimensions") && diag.message.contains("do not agree"),
            "unexpected lint message: {}",
            diag.message
        );
    }

    #[test]
    fn diagnostics_include_shared_missing_function_resolution() {
        let text = "value = missing_project_function(1);";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let diagnostics = diagnostics_for_document(text, &analysis);
        let diagnostic = diagnostics
            .iter()
            .find(|diagnostic| {
                matches!(
                    &diagnostic.code,
                    Some(lsp_types::NumberOrString::String(code)) if code == "RM-RES0001"
                )
            })
            .expect("shared resolution diagnostic");
        assert_eq!(diagnostic.severity, Some(DiagnosticSeverity::WARNING));
        assert_eq!(diagnostic.range.start, Position::new(0, 8));
        assert_eq!(
            diagnostic
                .data
                .as_ref()
                .and_then(|data| data.get("state"))
                .and_then(serde_json::Value::as_str),
            Some("unresolved")
        );
    }

    #[test]
    fn parse_failures_keep_the_shared_frontend_code() {
        for (text, expected) in [
            ("value = ;", Position::new(0, 8)),
            ("if true\n    value = ;\nend", Position::new(1, 12)),
            ("function out = f()\nif true\n    value = ;\nend\nend", Position::new(2, 12)),
            ("% 😀é\nclassdef C\nmethods\nfunction out = f(obj)\nif true\n    value = ;\nend\nend\nend\nend", Position::new(5, 12)),
            ("if true\n    label = \"😀é\"; value = ;\nend", Position::new(1, 27)),
        ] {
            let analysis = analyze_document_with_compat(text, CompatMode::default());
            let syntax = analysis.syntax_error.as_ref().expect("syntax failure");
            assert_eq!(syntax.position, text.find("value = ;").unwrap() + "value = ".len(), "{text}");
            let diagnostics = diagnostics_for_document(text, &analysis);
            let diagnostic = diagnostics.iter().find(|diagnostic| matches!(
                &diagnostic.code,
                Some(lsp_types::NumberOrString::String(code)) if code == "RunMat:ParseError"
            )).expect("shared parse diagnostic");
            assert_eq!(diagnostic.range.start, expected, "{text}");
            assert_eq!(diagnostic.range.end, Position::new(expected.line, expected.character + 1), "{text}");
        }
    }

    #[test]
    fn diagnostics_do_not_report_logical_index_contract_for_numeric_indexing() {
        let text = "a = 0:pi/100:2*pi; b = sin(a); c = a(1:10);";
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        let diags = diagnostics_for_document(text, &analysis);
        let has_logical_index_diagnostic = diags.iter().any(|d| match &d.code {
            Some(lsp_types::NumberOrString::String(code)) => code == "RM-TYPE-LOGICAL-INDEX",
            _ => false,
        });
        assert!(
            !has_logical_index_diagnostic,
            "unexpected logical index diagnostic: {:?}",
            diags
        );
    }

    #[test]
    fn hover_reports_indexed_slice_shape_for_assigned_symbol() {
        let text = "a = 0:pi/100:2*pi; c = a(1:22); d = c;";
        let analysis = analyze_document_with_compat(text, CompatMode::default());

        let c_use_offset = text.rfind('c').expect("c usage offset");
        let c_use_position = offset_to_position(text, c_use_offset);
        let c_hover = hover_at(text, &analysis, &c_use_position).expect("c hover");
        let c_value = match c_hover.contents {
            lsp_types::HoverContents::Markup(markup) => markup.value,
            other => panic!("unexpected hover contents {other:?}"),
        };

        assert!(c_value.contains("double"), "unexpected c hover {c_value}");
        assert!(c_value.contains("1 x 22"), "unexpected c hover {c_value}");
    }

    #[test]
    fn semantic_tokens_encode_function_parameter_and_local_roles() {
        let text = "function y = foo(x)\nlocal = x + 1;\ny = local;\nend\nz = foo(1);";
        let analysis = analyze_document_with_compat(text, CompatMode::RunMat);
        let tokens = semantic_tokens_full(text, &analysis).expect("semantic tokens");
        let legend = semantic_tokens_legend();
        let decoded = decode_semantic_tokens(text, &tokens);

        let foo_decl = text.find("foo").expect("foo decl");
        let x_param = text.find("(x)").expect("x param") + 1;
        let local_decl = text.find("local =").expect("local decl");
        let foo_call = text.rfind("foo").expect("foo call");

        assert_role_at(
            text,
            &decoded,
            &legend,
            foo_decl,
            lsp_types::SemanticTokenType::FUNCTION,
        );
        assert_modifier_at(text, &decoded, foo_decl, 0);
        assert_role_at(
            text,
            &decoded,
            &legend,
            x_param,
            lsp_types::SemanticTokenType::PARAMETER,
        );
        assert_role_at(
            text,
            &decoded,
            &legend,
            local_decl,
            lsp_types::SemanticTokenType::VARIABLE,
        );
        assert_role_at(
            text,
            &decoded,
            &legend,
            foo_call,
            lsp_types::SemanticTokenType::FUNCTION,
        );
    }

    #[test]
    fn rename_subject_retains_parameter_owner() {
        let source = "function y = impactPulse(t, scale)\n    y = t * scale;\nend\n";
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        let offset = source.find("scale").unwrap();
        let subject = rename_subject_at(source, &analysis, &offset_to_position(source, offset))
            .expect("parameter should be renameable");
        assert_eq!(subject.name, "scale");
        assert!(subject.is_local);
        assert_eq!(subject.parameter_owner.as_deref(), Some("impactPulse"));
    }

    #[test]
    fn rename_subject_rejects_a_name_already_bound_in_its_scope() {
        let source =
            "function y = impactPulse(t, scale)\n    gain = 2;\n    y = t * scale * gain;\nend\n";
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        let offset = source.find("scale").unwrap();
        let position = offset_to_position(source, offset);
        assert!(rename_subject_has_collision(
            source, &analysis, &position, "gain"
        ));
        assert!(!rename_subject_has_collision(
            source,
            &analysis,
            &position,
            "amplitude"
        ));
    }

    #[test]
    fn semantic_tokens_mark_builtin_calls_as_default_library() {
        let text = "y = sin(1);";
        let analysis = analyze_document_with_compat(text, CompatMode::RunMat);
        let tokens = semantic_tokens_full(text, &analysis).expect("semantic tokens");
        let legend = semantic_tokens_legend();
        let decoded = decode_semantic_tokens(text, &tokens);
        let sin_offset = text.find("sin").expect("sin offset");

        assert_role_at(
            text,
            &decoded,
            &legend,
            sin_offset,
            lsp_types::SemanticTokenType::FUNCTION,
        );
        assert_modifier_at(text, &decoded, sin_offset, 1);
    }

    #[test]
    fn lexical_semantic_tokens_cover_incomplete_source_before_analysis() {
        let text = "% title\nvalue = sin(42\n";
        let tokens = semantic_tokens_lexical(text).expect("lexical tokens");
        let legend = semantic_tokens_legend();
        let decoded = decode_semantic_tokens(text, &tokens);

        assert_role_at(
            text,
            &decoded,
            &legend,
            text.find('%').expect("comment start"),
            lsp_types::SemanticTokenType::COMMENT,
        );
        assert_role_at(
            text,
            &decoded,
            &legend,
            text.find("42").expect("number"),
            lsp_types::SemanticTokenType::NUMBER,
        );
    }

    #[test]
    fn semantic_tokens_include_comments() {
        let text = "% Grid\nx = 1;";
        let analysis = analyze_document_with_compat(text, CompatMode::RunMat);
        let tokens = semantic_tokens_full(text, &analysis).expect("semantic tokens");
        let legend = semantic_tokens_legend();
        let decoded = decode_semantic_tokens(text, &tokens);
        let comment_offset = text.find('%').expect("comment offset");

        assert_role_at(
            text,
            &decoded,
            &legend,
            comment_offset,
            lsp_types::SemanticTokenType::COMMENT,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn source_context_symbol_discovery_reads_manifest_project_symbols() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("runmat_lsp_symbol_discovery_{suffix}"));
        fs::create_dir_all(root.join("+stats")).expect("create package dir");
        fs::write(
            root.join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
        )
        .expect("write manifest");
        fs::write(
            root.join("+stats/summarize.m"),
            "function y = summarize(x); y = x; end",
        )
        .expect("write package function");
        let source_text = "x = stats.summarize(1);";
        fs::write(root.join("main.m"), source_text).expect("write source file");

        let source_name = root.join("main.m");
        let symbols = super::discover_known_project_symbols(source_name.to_str());
        assert!(
            symbols.contains("stats.summarize"),
            "expected project symbol discovery to include package-qualified names"
        );
        let analysis = analyze_document_with_compat_and_source(
            source_text,
            CompatMode::RunMat,
            source_name.to_str(),
        );
        assert!(analysis.project_revision.is_some());
        assert!(analysis.resolution.iter().any(|evidence| {
            evidence.name == "stats.summarize"
                && evidence.definition.as_ref().is_some_and(|definition| {
                    definition.package_instance.is_some() && definition.source_id.is_some()
                })
        }));
        let rename_ranges =
            function_rename_references_in_document(source_text, &analysis, "summarize");
        assert_eq!(rename_ranges.len(), 1);
        let rename_range = rename_ranges[0];
        assert_eq!(
            &source_text[position_to_offset(source_text, &rename_range.start)
                ..position_to_offset(source_text, &rename_range.end)],
            "summarize"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn source_context_symbol_discovery_exposes_class_folder_constructor_identity() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("runmat_lsp_class_symbol_{suffix}"));
        fs::create_dir_all(root.join("src/@Report")).expect("create class folder");
        fs::write(
            root.join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]
"#,
        )
        .expect("write manifest");
        fs::write(root.join("src/@Report/Report.m"), "classdef Report; end")
            .expect("write class source");
        let source = root.join("src/main.m");
        fs::write(&source, "report = Report();").expect("write entry source");

        let symbols = super::discover_known_project_symbols(source.to_str());
        assert!(symbols.contains("Report"));
        assert!(symbols.contains("Report.Report"));

        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn source_catalog_failures_are_not_silently_dropped() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("runmat_lsp_catalog_error_{suffix}"));
        fs::create_dir_all(&root).expect("create project");
        fs::write(
            root.join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["missing-source-root"]
"#,
        )
        .expect("write manifest");
        let source = root.join("main.m");
        fs::write(&source, "value = 1;").expect("write source");

        let analysis = analyze_document_with_compat_and_source(
            "value = 1;",
            CompatMode::RunMat,
            source.to_str(),
        );
        assert!(analysis
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CAT0001"));

        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn cross_file_definition_and_references_resolve_project_symbols() {
        let root = create_project_fixture("lsp_cross_file");
        let main_source = root.join("src/main.m");
        let main_text = fs::read_to_string(&main_source).expect("read main source");
        let uri = Url::from_file_path(&main_source).expect("main uri");

        let analysis = analyze_document_with_compat_and_source(
            &main_text,
            CompatMode::RunMat,
            main_source.to_str(),
        );
        let call_offset = main_text.find("summarize").expect("call offset");
        let call_position = offset_to_position(&main_text, call_offset);

        let defs = definition_locations_at(&main_text, &analysis, &call_position, &uri);
        assert!(
            defs.iter()
                .any(|loc| loc.uri.path().ends_with("/src/+stats/summarize.m")),
            "expected cross-file definition to resolve stats.summarize, got: {defs:?}"
        );

        let refs = references_locations_at(&main_text, &analysis, &call_position, &uri);
        assert!(
            refs.iter()
                .any(|loc| loc.uri.path().ends_with("/src/main.m")),
            "expected references to include callsite in main.m, got: {refs:?}"
        );
        assert!(
            refs.iter()
                .any(|loc| loc.uri.path().ends_with("/src/+stats/summarize.m")),
            "expected references to include package function file, got: {refs:?}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn workspace_symbols_include_unopened_project_files() {
        let root = create_project_fixture("lsp_workspace_symbols");
        let main_source = root.join("src/main.m");
        let helper_source = root.join("src/helpers/extra.m");
        let main_text = fs::read_to_string(&main_source).expect("read main source");

        let main_analysis = analyze_document_with_compat_and_source(
            &main_text,
            CompatMode::RunMat,
            main_source.to_str(),
        );
        let docs = vec![(
            Url::from_file_path(&main_source).expect("main uri"),
            main_text,
            main_analysis,
        )];

        let symbols = workspace_symbols_with_project(&docs, CompatMode::RunMat, None);
        assert!(
            symbols
                .iter()
                .any(|sym| sym.location.uri.path().ends_with("/src/helpers/extra.m")),
            "expected unopened helper symbol to appear in workspace symbols, got: {symbols:?}"
        );
        assert!(
            symbols
                .iter()
                .any(|sym| sym.location.uri.path().ends_with("/src/@MyClass/scale.m")),
            "expected class-folder method symbol to appear in workspace symbols, got: {symbols:?}"
        );

        // sanity: unopened helper file exists and was not in docs input
        assert!(helper_source.is_file(), "helper source missing");
        assert_eq!(docs.len(), 1, "expected only main doc to be open");

        let _ = fs::remove_dir_all(root);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sync_async_project_navigation_and_symbols_are_equivalent() {
        let root = create_project_fixture("lsp_parity");
        let main_source = root.join("src/main.m");
        let main_text = fs::read_to_string(&main_source).expect("read main source");
        let uri = Url::from_file_path(&main_source).expect("main uri");
        let analysis = analyze_document_with_compat_and_source(
            &main_text,
            CompatMode::RunMat,
            main_source.to_str(),
        );
        let call_offset = main_text.find("summarize").expect("call offset");
        let call_position = offset_to_position(&main_text, call_offset);

        let sync_defs = definition_locations_at(&main_text, &analysis, &call_position, &uri);
        let async_defs = block_on(definition_locations_at_async(
            &main_text,
            &analysis,
            &call_position,
            &uri,
        ));
        assert_eq!(
            location_keys(&sync_defs),
            location_keys(&async_defs),
            "sync/async definitions diverged"
        );

        let sync_refs = references_locations_at(&main_text, &analysis, &call_position, &uri);
        let async_refs = block_on(references_locations_at_async(
            &main_text,
            &analysis,
            &call_position,
            &uri,
        ));
        assert_eq!(
            location_keys(&sync_refs),
            location_keys(&async_refs),
            "sync/async references diverged"
        );

        let docs = vec![(uri, main_text, analysis)];
        let sync_syms = workspace_symbols_with_project(&docs, CompatMode::RunMat, None);
        let async_syms = block_on(
            crate::core::workspace::workspace_symbols_with_project_async(
                &docs,
                CompatMode::RunMat,
                None,
            ),
        );
        assert_eq!(
            symbol_keys(&sync_syms),
            symbol_keys(&async_syms),
            "sync/async workspace symbols diverged"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn create_project_fixture(prefix: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("{prefix}_{suffix}"));
        fs::create_dir_all(root.join("src/+stats")).expect("create package dir");
        fs::create_dir_all(root.join("src/helpers")).expect("create helper dir");
        fs::create_dir_all(root.join("src/@MyClass")).expect("create class folder dir");
        fs::create_dir_all(root.join("deps/tools/src")).expect("create dep dir");

        fs::write(
            root.join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[dependencies]
tools = { path = "deps/tools" }
"#,
        )
        .expect("write root manifest");
        fs::write(
            root.join("deps/tools/runmat.toml"),
            r#"
[package]
name = "tools"

[sources]
roots = ["src"]
"#,
        )
        .expect("write dep manifest");
        fs::write(
            root.join("src/+stats/summarize.m"),
            "function y = summarize(x); y = x + 1; end",
        )
        .expect("write package function");
        fs::write(
            root.join("src/helpers/extra.m"),
            "function y = extra(x); y = x; end",
        )
        .expect("write helper function");
        fs::write(
            root.join("src/@MyClass/scale.m"),
            "function y = scale(x); y = x; end",
        )
        .expect("write class method");
        fs::write(
            root.join("deps/tools/src/util.m"),
            "function y = util(x); y = x; end",
        )
        .expect("write dependency function");
        fs::write(
            root.join("src/main.m"),
            "x = summarize(41);\ny = tools.util(x);",
        )
        .expect("write main");
        root
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn location_keys(locations: &[Location]) -> std::collections::BTreeSet<String> {
        locations
            .iter()
            .map(|loc| {
                format!(
                    "{}:{}:{}:{}:{}",
                    loc.uri,
                    loc.range.start.line,
                    loc.range.start.character,
                    loc.range.end.line,
                    loc.range.end.character
                )
            })
            .collect()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn symbol_keys(symbols: &[lsp_types::SymbolInformation]) -> std::collections::BTreeSet<String> {
        symbols
            .iter()
            .map(|sym| {
                format!(
                    "{}:{}:{}:{}:{}:{}",
                    sym.name,
                    sym.location.uri,
                    sym.location.range.start.line,
                    sym.location.range.start.character,
                    sym.location.range.end.line,
                    sym.location.range.end.character
                )
            })
            .collect()
    }

    fn decode_semantic_tokens(
        text: &str,
        tokens: &lsp_types::SemanticTokens,
    ) -> Vec<(lsp_types::Position, u32, u32)> {
        let mut out = Vec::new();
        let mut line = 0u32;
        let mut col = 0u32;
        for token in &tokens.data {
            line += token.delta_line;
            if token.delta_line > 0 {
                col = token.delta_start;
            } else {
                col += token.delta_start;
            }
            let position = lsp_types::Position::new(line, col);
            let offset = position_to_offset(text, &position);
            let _ = offset;
            out.push((position, token.token_type, token.token_modifiers_bitset));
        }
        out
    }

    fn assert_role_at(
        text: &str,
        decoded: &[(lsp_types::Position, u32, u32)],
        legend: &lsp_types::SemanticTokensLegend,
        offset: usize,
        expected: lsp_types::SemanticTokenType,
    ) {
        let position = offset_to_position(text, offset);
        let Some((_, token_type_idx, _)) = decoded.iter().find(|(pos, _, _)| *pos == position)
        else {
            panic!("no semantic token at {position:?}");
        };
        let actual = legend
            .token_types
            .get(*token_type_idx as usize)
            .expect("token type in range");
        assert_eq!(actual, &expected, "unexpected token type at {position:?}");
    }

    fn assert_modifier_at(
        text: &str,
        decoded: &[(lsp_types::Position, u32, u32)],
        offset: usize,
        modifier_idx: u32,
    ) {
        let position = offset_to_position(text, offset);
        let Some((_, _, bitset)) = decoded.iter().find(|(pos, _, _)| *pos == position) else {
            panic!("no semantic token at {position:?}");
        };
        let mask = 1u32 << modifier_idx;
        assert!(
            (*bitset & mask) != 0,
            "expected modifier bit {modifier_idx} at {position:?}"
        );
    }
}
