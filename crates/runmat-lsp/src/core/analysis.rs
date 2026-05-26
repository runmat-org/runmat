use crate::core::docs;
use crate::core::position::{offset_to_position, position_to_offset};
use crate::core::project::ProjectContext;
use crate::core::semantic_tokens::{IdentifierRole, SemanticHint};
use lsp_types::{
    CompletionItem, Diagnostic, DiagnosticSeverity, DocumentSymbol, Hover, Location, Position,
    Range, SignatureHelp, Url,
};
use runmat_builtins::{self, BuiltinFunction, Constant, Type};
use runmat_hir::{
    CallKind, DefPath, FunctionKind, HirDiagnostic, HirDiagnosticSeverity, HirError,
    LoweringContext, LoweringResult, ReferenceKind,
};
use runmat_lexer::{tokenize_detailed, SpannedToken, Token};
pub use runmat_parser::CompatMode;
use runmat_parser::{parse_with_options, ParserOptions};
use runmat_vm::CompileError;
use std::collections::HashMap;
use std::collections::HashSet;
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

#[derive(Clone, Copy, Debug)]
pub struct TextRange {
    pub start: usize,
    pub end: usize,
}

impl TextRange {
    pub fn contains(&self, offset: usize) -> bool {
        self.start <= offset && offset < self.end
    }

    pub fn to_lsp_range(self, text: &str) -> Range {
        Range {
            start: offset_to_position(text, self.start),
            end: offset_to_position(text, self.end),
        }
    }
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
    let known_project_symbols = discover_known_project_symbols(source_name);
    match parse_with_options(text, ParserOptions::new(compat)) {
        Ok(ast) => {
            let mut lowering_context = LoweringContext::empty()
                .with_runmat_extensions_enabled(compat.allows_runmat_extensions());
            if !known_project_symbols.is_empty() {
                lowering_context =
                    lowering_context.with_known_project_symbols(&known_project_symbols);
            }
            let lowering = match runmat_hir::lower(&ast, &lowering_context) {
                Ok(result) => result,
                Err(err) => {
                    return DocumentAnalysis {
                        tokens,
                        syntax_error: None,
                        lowering_error: Some(err),
                        compile_error: None,
                        lowering: None,
                        semantic: None,
                    };
                }
            };
            let compile_error = compile_error_for_lowering(&lowering);

            let semantic = build_semantic_model(&lowering, &tokens, text);

            DocumentAnalysis {
                tokens,
                syntax_error: None,
                lowering_error: None,
                compile_error,
                lowering: Some(lowering),
                semantic: Some(semantic),
            }
        }
        Err(err) => {
            let mut message = err.message.clone();
            if let Some(expected) = &err.expected {
                message = format!("{message} (expected {expected})");
            }
            if let Some(found) = &err.found_token {
                message = format!("{message} (found '{found}')");
            }

            DocumentAnalysis {
                tokens,
                syntax_error: Some(SyntaxErrorInfo {
                    message,
                    position: err.position,
                }),
                lowering_error: None,
                compile_error: None,
                lowering: None,
                semantic: None,
            }
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub async fn analyze_document_with_compat_and_source_async(
    text: &str,
    compat: CompatMode,
    source_name: Option<&str>,
) -> DocumentAnalysis {
    let tokens = tokenize_detailed(text);
    let known_project_symbols = discover_known_project_symbols_async(source_name).await;
    match parse_with_options(text, ParserOptions::new(compat)) {
        Ok(ast) => {
            let mut lowering_context = LoweringContext::empty()
                .with_runmat_extensions_enabled(compat.allows_runmat_extensions());
            if !known_project_symbols.is_empty() {
                lowering_context =
                    lowering_context.with_known_project_symbols(&known_project_symbols);
            }
            let lowering = match runmat_hir::lower(&ast, &lowering_context) {
                Ok(result) => result,
                Err(err) => {
                    return DocumentAnalysis {
                        tokens,
                        syntax_error: None,
                        lowering_error: Some(err),
                        compile_error: None,
                        lowering: None,
                        semantic: None,
                    };
                }
            };
            let compile_error = compile_error_for_lowering(&lowering);
            let semantic = build_semantic_model(&lowering, &tokens, text);

            DocumentAnalysis {
                tokens,
                syntax_error: None,
                lowering_error: None,
                compile_error,
                lowering: Some(lowering),
                semantic: Some(semantic),
            }
        }
        Err(err) => {
            let mut message = err.message.clone();
            if let Some(expected) = &err.expected {
                message = format!("{message} (expected {expected})");
            }
            if let Some(found) = &err.found_token {
                message = format!("{message} (found '{found}')");
            }

            DocumentAnalysis {
                tokens,
                syntax_error: Some(SyntaxErrorInfo {
                    message,
                    position: err.position,
                }),
                lowering_error: None,
                compile_error: None,
                lowering: None,
                semantic: None,
            }
        }
    }
}

fn discover_known_project_symbols(source_name: Option<&str>) -> HashSet<String> {
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
        return HashSet::new();
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        futures::executor::block_on(
            runmat_config::discover_known_project_symbols_from_source_name_async(source_name, &cwd),
        )
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = source_name;
        let _ = cwd;
        HashSet::new()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
async fn discover_known_project_symbols_async(source_name: Option<&str>) -> HashSet<String> {
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
        return HashSet::new();
    };
    runmat_config::discover_known_project_symbols_from_source_name_async(source_name, &cwd).await
}

fn compile_error_for_lowering(lowering: &LoweringResult) -> Option<CompileError> {
    let entrypoint = lowering
        .assembly
        .entrypoints
        .first()
        .ok_or_else(|| CompileError::new("semantic bytecode compile requires an entrypoint"));
    let entrypoint = match entrypoint {
        Ok(entrypoint) => entrypoint,
        Err(err) => return Some(err),
    };
    let mir = match runmat_mir::lowering::lower_assembly(&lowering.assembly) {
        Ok(mir) => mir,
        Err(err) => return Some(CompileError::from(err)),
    };
    let _analysis = runmat_mir::analysis::analyze_assembly(&mir);
    runmat_vm::compile(&lowering.assembly, &mir, entrypoint.id).err()
}

pub fn diagnostics_for_document(text: &str, analysis: &DocumentAnalysis) -> Vec<Diagnostic> {
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
    _text: &str,
    _analysis: &DocumentAnalysis,
    _position: &Position,
) -> Vec<CompletionItem> {
    if let Some(semantic) = &_analysis.semantic {
        completion_from_semantic(semantic)
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
                return Some(Hover {
                    contents: lsp_types::HoverContents::Markup(lsp_types::MarkupContent {
                        kind: lsp_types::MarkupKind::Markdown,
                        value: format_variable_hover(&ident, var),
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
            return Some(Hover {
                contents: lsp_types::HoverContents::Markup(lsp_types::MarkupContent {
                    kind: lsp_types::MarkupKind::Markdown,
                    value: format_variable_hover(&ident, glob),
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

    // Built-in functions (rich docs)
    if let Some(func) = runmat_builtins::builtin_functions()
        .into_iter()
        .find(|f| f.name.eq_ignore_ascii_case(&ident))
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
    let semantic = _analysis.semantic.as_ref()?;
    let offset = position_to_offset(_text, _position);
    let token = token_at_offset(&_analysis.tokens, offset)?;
    let name = token.lexeme.clone();
    if let Some(func) = runmat_builtins::builtin_functions()
        .into_iter()
        .find(|builtin| builtin.name.eq_ignore_ascii_case(&name))
    {
        if let Some(labels) = docs::signature_labels(func) {
            let signatures = labels
                .into_iter()
                .map(|label| lsp_types::SignatureInformation {
                    label,
                    documentation: None,
                    parameters: None,
                    active_parameter: None,
                })
                .collect();
            return Some(SignatureHelp {
                signatures,
                active_signature: Some(0),
                active_parameter: None,
            });
        }
    }

    let funcs = semantic.function_lookup.get(&name)?;

    let mut sigs = Vec::new();
    for idx in funcs {
        if let Some(f) = semantic.functions.get(*idx) {
            sigs.push(lsp_types::SignatureInformation {
                label: f.signature.display(),
                documentation: None,
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
        active_parameter: None,
    })
}

pub fn document_symbols(_text: &str, _analysis: &DocumentAnalysis) -> Vec<DocumentSymbol> {
    let semantic = match &_analysis.semantic {
        Some(s) => s,
        None => return vec![],
    };
    let mut symbols = Vec::new();
    for func in &semantic.functions {
        #[allow(deprecated)]
        symbols.push(DocumentSymbol {
            name: func.signature.name.clone(),
            detail: Some(func.signature.display()),
            kind: lsp_types::SymbolKind::FUNCTION,
            tags: None,
            deprecated: None,
            range: func.range.to_lsp_range(_text),
            selection_range: func.selection.to_lsp_range(_text),
            children: None,
        });
    }
    symbols
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
    for source_file in project.all_source_files() {
        let Some(text) = read_file_text(source_file) else {
            continue;
        };
        let Some(source_name) = source_file.to_str() else {
            continue;
        };
        let analysis = analyze_document_with_compat_and_source(&text, compat, Some(source_name));
        for range in function_definitions_in_document(&text, &analysis, symbol_name) {
            if let Some(uri) = file_path_to_url(source_file) {
                locations.push(Location { uri, range });
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
    for source_file in project.all_source_files() {
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
    for source_file in project.all_source_files() {
        let Ok(text) = runmat_filesystem::read_to_string_async(source_file).await else {
            continue;
        };
        let Some(source_name) = source_file.to_str() else {
            continue;
        };
        let analysis =
            analyze_document_with_compat_and_source_async(&text, compat, Some(source_name)).await;
        for range in function_definitions_in_document(&text, &analysis, symbol_name) {
            if let Some(uri) = file_path_to_url(source_file) {
                locations.push(Location { uri, range });
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
    for source_file in project.all_source_files() {
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
        return uri
            .to_file_path()
            .ok()
            .and_then(|path| path.to_str().map(str::to_owned));
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
        return Url::from_file_path(path).ok();
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

#[derive(Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub outputs: Vec<String>,
    pub inputs: Vec<String>,
    pub name_range: TextRange,
}

impl FunctionSignature {
    pub fn display(&self) -> String {
        let mut buf = String::new();
        if !self.outputs.is_empty() {
            if self.outputs.len() == 1 {
                let _ = write!(buf, "{} = ", self.outputs[0]);
            } else {
                let _ = write!(buf, "[{}] = ", self.outputs.join(", "));
            }
        }
        let _ = write!(buf, "{}", self.name);
        let args = self.inputs.join(", ");
        let _ = write!(buf, "({args})");
        buf
    }
}

#[derive(Clone)]
pub struct VariableSymbol {
    pub name: String,
    pub ty: Type,
    pub kind: VariableKind,
    pub declared_span: Option<TextRange>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VariableKind {
    Global,
    Parameter,
    Output,
    Local,
}

impl VariableKind {
    pub fn as_label(&self) -> &'static str {
        match self {
            VariableKind::Global => "global",
            VariableKind::Parameter => "parameter",
            VariableKind::Output => "output",
            VariableKind::Local => "local",
        }
    }
}

#[derive(Clone)]
pub struct FunctionSemantic {
    pub name: String,
    pub signature: FunctionSignature,
    pub range: TextRange,
    pub selection: TextRange,
    pub variables: HashMap<String, VariableSymbol>,
}

#[derive(Clone)]
pub struct AnalysisModel {
    pub globals: HashMap<String, VariableSymbol>,
    pub functions: Vec<FunctionSemantic>,
    pub function_lookup: HashMap<String, Vec<usize>>,
    pub token_hints: Vec<SemanticHint>,
    pub exported_symbols: HashSet<String>,
    pub referenced_symbols: HashSet<String>,
    pub status_message: String,
    pub diagnostics: Vec<HirDiagnostic>,
}

impl AnalysisModel {
    fn function_at_offset(&self, offset: usize) -> Option<&FunctionSemantic> {
        self.functions.iter().find(|f| f.range.contains(offset))
    }
}

fn completion_from_semantic(semantic: &AnalysisModel) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    for var in semantic.globals.values() {
        items.push(variable_completion(var));
    }
    for func in &semantic.functions {
        for var in func.variables.values() {
            items.push(variable_completion(var));
        }
        items.push(function_completion(func));
    }
    for func in runmat_builtins::builtin_functions() {
        items.push(builtin_completion(func));
    }
    for constant in runmat_builtins::constants() {
        items.push(constant_completion(constant));
    }
    items
}

fn variable_completion(var: &VariableSymbol) -> CompletionItem {
    CompletionItem {
        label: var.name.clone(),
        kind: Some(lsp_types::CompletionItemKind::VARIABLE),
        detail: Some(format!("{}: {}", var.kind.as_label(), format_type(&var.ty))),
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

fn constant_completion(constant: &Constant) -> CompletionItem {
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

fn format_variable_hover(name: &str, symbol: &VariableSymbol) -> String {
    let mut buf = String::new();
    let _ = writeln!(
        buf,
        "```runmat\n{kind} {name}: {ty}\n```",
        kind = symbol.kind.as_label(),
        ty = format_type(&symbol.ty)
    );
    if matches!(symbol.kind, VariableKind::Global) {
        let _ = writeln!(buf, "Global variable available across the workspace.");
    }
    buf
}

fn build_semantic_model(
    lowering: &LoweringResult,
    tokens: &[SpannedToken],
    text: &str,
) -> AnalysisModel {
    let mut functions = Vec::new();
    let mut function_lookup: HashMap<String, Vec<usize>> = HashMap::new();
    let mut globals = HashMap::new();

    let binding_shapes = runmat_static_analysis::infer_binding_shapes(&lowering);

    for binding in &lowering.assembly.bindings {
        if matches!(
            binding.workspace_visibility,
            runmat_hir::WorkspaceVisibility::Hidden
        ) {
            continue;
        }
        let name = binding.name.0.clone();
        let symbol = globals
            .entry(name.clone())
            .or_insert_with(|| VariableSymbol {
                name: name.clone(),
                ty: Type::Unknown,
                kind: VariableKind::Global,
                declared_span: span_to_text_range(binding.declared_span, text.len()),
            });
        if let Some(shape) = binding_shapes.get(&binding.id) {
            symbol.ty = type_from_shape(shape.clone());
        }
    }

    for function in &lowering.assembly.functions {
        if matches!(function.kind, FunctionKind::SyntheticEntrypoint) {
            continue;
        }
        let func_name = function.name.0.clone();

        let mut variables = HashMap::new();
        for param in &function.params {
            let Some(binding) = lowering.assembly.bindings.get(param.0) else {
                continue;
            };
            let name = binding.name.0.clone();
            let ty = binding_shapes
                .get(param)
                .cloned()
                .map(type_from_shape)
                .unwrap_or(Type::Unknown);
            variables.insert(
                name.clone(),
                VariableSymbol {
                    name: name.clone(),
                    ty,
                    kind: VariableKind::Parameter,
                    declared_span: span_to_text_range(binding.declared_span, text.len()),
                },
            );
        }
        for out in &function.outputs {
            let Some(binding) = lowering.assembly.bindings.get(out.0) else {
                continue;
            };
            let name = binding.name.0.clone();
            let ty = binding_shapes
                .get(out)
                .cloned()
                .map(type_from_shape)
                .unwrap_or(Type::Unknown);
            variables.insert(
                name.clone(),
                VariableSymbol {
                    name: name.clone(),
                    ty,
                    kind: VariableKind::Output,
                    declared_span: span_to_text_range(binding.declared_span, text.len()),
                },
            );
        }
        for local in &function.locals {
            let Some(binding) = lowering.assembly.bindings.get(local.0) else {
                continue;
            };
            let name = binding.name.0.clone();
            let ty = binding_shapes
                .get(local)
                .cloned()
                .map(type_from_shape)
                .unwrap_or(Type::Unknown);
            variables
                .entry(name.clone())
                .or_insert_with(|| VariableSymbol {
                    name,
                    ty,
                    kind: VariableKind::Local,
                    declared_span: span_to_text_range(binding.declared_span, text.len()),
                });
        }
        for capture in &function.captures {
            let Some(binding) = lowering.assembly.bindings.get(capture.binding.0) else {
                continue;
            };
            let name = binding.name.0.clone();
            let ty = binding_shapes
                .get(&capture.binding)
                .cloned()
                .map(type_from_shape)
                .unwrap_or(Type::Unknown);
            variables
                .entry(name.clone())
                .or_insert_with(|| VariableSymbol {
                    name,
                    ty,
                    kind: VariableKind::Local,
                    declared_span: span_to_text_range(binding.declared_span, text.len()),
                });
        }

        let name_range =
            find_symbol_range(tokens, &func_name, None).unwrap_or(TextRange { start: 0, end: 0 });
        let body_range = TextRange {
            start: function.span.start,
            end: function.span.end.min(text.len()),
        };

        let selection = name_range;

        let signature = FunctionSignature {
            name: func_name.clone(),
            outputs: function
                .outputs
                .iter()
                .filter_map(|o| lowering.assembly.bindings.get(o.0))
                .map(|binding| binding.name.0.clone())
                .collect(),
            inputs: function
                .params
                .iter()
                .filter_map(|p| lowering.assembly.bindings.get(p.0))
                .map(|binding| binding.name.0.clone())
                .collect(),
            name_range,
        };

        let semantic = FunctionSemantic {
            name: func_name.clone(),
            signature,
            range: body_range,
            selection,
            variables,
        };
        functions.push(semantic);
    }

    for (idx, func) in functions.iter().enumerate() {
        function_lookup
            .entry(func.name.clone())
            .or_default()
            .push(idx);
    }

    let mut diagnostics = runmat_static_analysis::lint_shapes(&lowering);
    diagnostics.extend(runmat_static_analysis::lint_mir_analysis(&lowering));
    let token_hints = build_semantic_hints(lowering, tokens, &functions);
    let exported_symbols = build_exported_symbol_set(&functions);
    let referenced_symbols = build_referenced_symbol_set(lowering);

    AnalysisModel {
        globals,
        functions,
        function_lookup,
        token_hints,
        exported_symbols,
        referenced_symbols,
        status_message: String::new(),
        diagnostics,
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

fn build_semantic_hints(
    lowering: &LoweringResult,
    tokens: &[SpannedToken],
    functions: &[FunctionSemantic],
) -> Vec<SemanticHint> {
    let mut hint_map: HashMap<(usize, usize), (u8, SemanticHint)> = HashMap::new();
    for function in functions {
        insert_hint(
            &mut hint_map,
            function.signature.name_range,
            SemanticHint {
                start: function.signature.name_range.start,
                end: function.signature.name_range.end,
                role: IdentifierRole::Function,
                declaration: true,
                default_library: false,
            },
            80,
        );
        for token in tokens
            .iter()
            .filter(|token| matches!(token.token, Token::Ident))
            .filter(|token| function.range.contains(token.start))
        {
            let Some(variable) = function.variables.get(&token.lexeme) else {
                continue;
            };
            let role = match variable.kind {
                VariableKind::Parameter => IdentifierRole::Parameter,
                VariableKind::Output | VariableKind::Local | VariableKind::Global => {
                    IdentifierRole::Variable
                }
            };
            let declaration = variable
                .declared_span
                .is_some_and(|span| span.start == token.start && span.end == token.end);
            insert_hint(
                &mut hint_map,
                TextRange {
                    start: token.start,
                    end: token.end,
                },
                SemanticHint {
                    start: token.start,
                    end: token.end,
                    role,
                    declaration,
                    default_library: false,
                },
                20,
            );
        }
    }

    for reference in &lowering.hir_index.references {
        let role = match &reference.kind {
            ReferenceKind::Imported(_) | ReferenceKind::Package(_) => {
                Some(IdentifierRole::Namespace)
            }
            _ => None,
        };
        let Some(role) = role else {
            continue;
        };
        if let Some(range) =
            find_token_range_by_lexeme_in_span(tokens, &reference.name.0, reference.span)
        {
            insert_hint(
                &mut hint_map,
                range,
                SemanticHint {
                    start: range.start,
                    end: range.end,
                    role,
                    declaration: false,
                    default_library: false,
                },
                50,
            );
        }
    }

    for call in &lowering.hir_index.calls {
        let Some(name) = call.name.display_name() else {
            continue;
        };
        let Some(range) = find_token_range_by_lexeme_in_span(tokens, &name, call.span) else {
            continue;
        };
        insert_hint(
            &mut hint_map,
            range,
            SemanticHint {
                start: range.start,
                end: range.end,
                role: IdentifierRole::Function,
                declaration: false,
                default_library: matches!(call.kind, CallKind::Builtin(_)),
            },
            if matches!(call.kind, CallKind::Builtin(_)) {
                95
            } else {
                70
            },
        );
    }

    hint_map
        .into_values()
        .map(|(_, hint)| hint)
        .collect::<Vec<_>>()
}

fn build_exported_symbol_set(functions: &[FunctionSemantic]) -> HashSet<String> {
    let mut symbols = HashSet::new();
    for function in functions {
        symbols.insert(function.name.clone());
    }
    symbols
}

fn build_referenced_symbol_set(lowering: &LoweringResult) -> HashSet<String> {
    let mut symbols = HashSet::new();
    for call in &lowering.hir_index.calls {
        if let Some(name) = call.name.display_name() {
            symbols.insert(name);
        }
        if let CallKind::PackageFunction(path) = &call.kind {
            for name in def_path_symbol_variants(path) {
                symbols.insert(name);
            }
        }
    }
    symbols
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

fn insert_hint(
    hints: &mut HashMap<(usize, usize), (u8, SemanticHint)>,
    range: TextRange,
    hint: SemanticHint,
    priority: u8,
) {
    let key = (range.start, range.end);
    match hints.get(&key) {
        Some((existing, _)) if *existing >= priority => {}
        _ => {
            hints.insert(key, (priority, hint));
        }
    }
}

fn find_token_range_by_lexeme_in_span(
    tokens: &[SpannedToken],
    lexeme: &str,
    span: runmat_hir::Span,
) -> Option<TextRange> {
    tokens
        .iter()
        .filter(|token| matches!(token.token, Token::Ident))
        .find(|token| token.lexeme == lexeme && token.start >= span.start && token.end <= span.end)
        .map(|token| TextRange {
            start: token.start,
            end: token.end,
        })
}

fn type_from_shape(shape: Vec<Option<usize>>) -> Type {
    Type::Tensor { shape: Some(shape) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::workspace::workspace_symbols_with_project;
    use futures::executor::block_on;
    use lsp_types::Url;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

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
        let builtin_names: Vec<&str> = runmat_builtins::builtin_functions()
            .iter()
            .map(|f| f.name)
            .collect();
        assert!(
            builtin_names
                .iter()
                .any(|name| name.eq_ignore_ascii_case("plot")),
            "plot builtin should be registered for hover tests (registered: {:?})",
            builtin_names
        );
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
                    markup.value.contains("```runmat\nplot(...)\n```"),
                    "expected placeholder signature header, got:\n{}",
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
    fn signature_help_uses_empty_and_magic_descriptors() {
        let cases = [
            ("empty(0, 3);", "A = empty(m, n, ...)"),
            ("magic(5);", "A = magic(n)"),
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
    fn signature_help_uses_colon_fill_meshgrid_peaks_descriptors() {
        let cases = [
            ("colon(1, 0.5, 3);", "x = colon(start, step, stop)"),
            ("fill(2, 3, 4);", "A = fill(value, m, n, ...)"),
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
            ("ismatrix([1 2; 3 4]);", "tf = ismatrix(A)"),
            ("isscalar(1);", "tf = isscalar(A)"),
            ("isvector([1 2 3]);", "tf = isvector(A)"),
            ("length([1 2 3]);", "n = length(A)"),
            ("ndims(ones(2,2,2));", "n = ndims(A)"),
            ("numel([1 2; 3 4], 1, 2);", "n = numel(A, dim, ...)"),
            ("size([1 2; 3 4], 1);", "d = size(A, dim)"),
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
    fn signature_help_uses_structs_core_descriptors() {
        let cases = [
            ("fieldnames(struct());", "names = fieldnames(S)"),
            (
                "getfield(struct('a', 1), 'a');",
                "value = getfield(S, field)",
            ),
            ("isfield(struct('a', 1), 'a');", "tf = isfield(S, name)"),
            ("orderfields(struct('a', 1));", "S = orderfields(S)"),
            (
                "setfield(struct(), 'a', 1);",
                "S = setfield(S, field, value)",
            ),
            ("struct('a', 1);", "S = struct(field, value, ...)"),
            ("rmfield(struct('a', 1), 'a');", "S = rmfield(S, field)"),
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
        assert!(x_value.contains("Tensor"), "unexpected x hover {x_value}");
        assert!(x_value.contains("1 x 101"), "unexpected x hover {x_value}");
        assert!(y_value.contains("Tensor"), "unexpected y hover {y_value}");
        assert!(y_value.contains("1 x 101"), "unexpected y hover {y_value}");
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

        assert!(x_value.contains("Tensor"), "unexpected hover {x_value}");
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
            Some(lsp_types::NumberOrString::String(code)) => code == "lint.shape.matmul",
            _ => false,
        });
        let diag = diag.expect("expected matmul lint");
        assert!(
            diag.message.contains("inner dimensions") && diag.message.contains("must match"),
            "unexpected lint message: {}",
            diag.message
        );
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
        fs::write(root.join("main.m"), "x = 1;").expect("write source file");

        let source_name = root.join("main.m");
        let symbols = super::discover_known_project_symbols(source_name.to_str());
        assert!(
            symbols.contains("stats.summarize"),
            "expected project symbol discovery to include package-qualified names"
        );

        let _ = fs::remove_dir_all(&root);
    }

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
