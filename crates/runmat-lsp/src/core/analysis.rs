#![allow(dead_code)]

use crate::core::docs;
use crate::core::position::{offset_to_position, position_to_offset};
use lsp_types::{
    CompletionItem, Diagnostic, DiagnosticSeverity, DocumentSymbol, Hover, Position, Range,
    SignatureHelp,
};
use runmat_builtins::{self, BuiltinFunction, Constant, Type};
use runmat_hir::{HirStmt, LoweringContext, LoweringResult, SemanticError};
use runmat_ignition::{compile, CompileError};
use runmat_lexer::{tokenize_detailed, SpannedToken, Token};
pub use runmat_parser::CompatMode;
use runmat_parser::{parse_with_options, ParserOptions};
use std::collections::HashMap;
use std::fmt::Write;

#[derive(Clone)]
pub struct DocumentAnalysis {
    pub tokens: Vec<SpannedToken>,
    pub syntax_error: Option<SyntaxErrorInfo>,
    pub lowering_error: Option<SemanticError>,
    pub compile_error: Option<CompileError>,
    pub semantic: Option<SemanticModel>,
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

pub fn analyze_document(text: &str) -> DocumentAnalysis {
    analyze_document_with_compat(text, CompatMode::default())
}

pub fn analyze_document_with_compat(text: &str, compat: CompatMode) -> DocumentAnalysis {
    let tokens = tokenize_detailed(text);
    match parse_with_options(text, ParserOptions::new(compat)) {
        Ok(ast) => {
            let lowering = match runmat_hir::lower(&ast, &LoweringContext::empty()) {
                Ok(result) => result,
                Err(err) => {
                    return DocumentAnalysis {
                        tokens,
                        syntax_error: None,
                        lowering_error: Some(err),
                        compile_error: None,
                        semantic: None,
                    };
                }
            };
            if let Err(err) = compile(&lowering.hir, &HashMap::new()) {
                return DocumentAnalysis {
                    tokens,
                    syntax_error: None,
                    lowering_error: None,
                    compile_error: Some(err),
                    semantic: None,
                };
            }

            let semantic = build_semantic_model(lowering, &tokens, text);

            DocumentAnalysis {
                tokens,
                syntax_error: None,
                lowering_error: None,
                compile_error: None,
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
                semantic: None,
            }
        }
    }
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
        if !semantic.status_message.is_empty() {
            return vec![Diagnostic {
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
            }];
        }
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

pub fn signature_help_at(
    _text: &str,
    _analysis: &DocumentAnalysis,
    _position: &Position,
) -> Option<SignatureHelp> {
    let semantic = _analysis.semantic.as_ref()?;
    let offset = position_to_offset(_text, _position);
    let token = token_at_offset(&_analysis.tokens, offset)?;
    let name = token.lexeme.clone();
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
    crate::core::semantic_tokens::full(text, &analysis.tokens)
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
    error: &SemanticError,
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
        source: Some("runmat-ignition".into()),
        message,
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
    pub return_types: Vec<Type>,
}

#[derive(Clone)]
pub struct SemanticModel {
    pub globals: HashMap<String, VariableSymbol>,
    pub functions: Vec<FunctionSemantic>,
    pub function_lookup: HashMap<String, Vec<usize>>,
    pub status_message: String,
}

impl SemanticModel {
    fn function_at_offset(&self, offset: usize) -> Option<&FunctionSemantic> {
        self.functions.iter().find(|f| f.range.contains(offset))
    }
}

fn completion_from_semantic(semantic: &SemanticModel) -> Vec<CompletionItem> {
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
    let detail = if !func.param_types.is_empty() {
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
    format!("{ty:?}")
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
    lowering: LoweringResult,
    tokens: &[SpannedToken],
    text: &str,
) -> SemanticModel {
    let mut functions = Vec::new();
    let mut function_lookup: HashMap<String, Vec<usize>> = HashMap::new();
    let mut globals = HashMap::new();

    for (name, var_id) in &lowering.variables {
        let ty = lowering
            .inferred_globals
            .get(&runmat_hir::VarId(*var_id))
            .cloned()
            .or_else(|| lowering.var_types.get(*var_id).cloned())
            .unwrap_or(Type::Unknown);
        globals.insert(
            name.clone(),
            VariableSymbol {
                name: name.clone(),
                ty,
                kind: VariableKind::Global,
            },
        );
    }

    for stmt in lowering.functions.values() {
        let HirStmt::Function {
            name: func_name,
            params,
            outputs,
            body: _,
            has_varargin: _,
            has_varargout: _,
            ..
        } = stmt.clone()
        else {
            continue;
        };

        let mut variables = HashMap::new();
        let inferred_env = lowering.inferred_function_envs.get(&func_name);
        for param in &params {
            let ty = inferred_env
                .and_then(|env| env.get(param).cloned())
                .or_else(|| lowering.var_types.get(param.0).cloned())
                .unwrap_or(Type::Unknown);
            let name = lowering
                .var_names
                .get(param)
                .cloned()
                .unwrap_or_else(|| format!("v{}", param.0));
            variables.insert(
                name.clone(),
                VariableSymbol {
                    name: name.clone(),
                    ty,
                    kind: VariableKind::Parameter,
                },
            );
        }
        for out in &outputs {
            let ty = inferred_env
                .and_then(|env| env.get(out).cloned())
                .or_else(|| lowering.var_types.get(out.0).cloned())
                .unwrap_or(Type::Unknown);
            let name = lowering
                .var_names
                .get(out)
                .cloned()
                .unwrap_or_else(|| format!("v{}", out.0));
            variables.insert(
                name.clone(),
                VariableSymbol {
                    name: name.clone(),
                    ty,
                    kind: VariableKind::Output,
                },
            );
        }

        let name_range =
            find_symbol_range(tokens, &func_name, None).unwrap_or(TextRange { start: 0, end: 0 });
        let body_range = TextRange {
            start: name_range.start,
            end: text.len(),
        };

        let selection = name_range;

        let signature = FunctionSignature {
            name: func_name.clone(),
            outputs: outputs
                .iter()
                .filter_map(|o| lowering.var_names.get(o).cloned())
                .collect(),
            inputs: params
                .iter()
                .filter_map(|p| lowering.var_names.get(p).cloned())
                .collect(),
            name_range,
        };

        let semantic = FunctionSemantic {
            name: func_name.clone(),
            signature,
            range: body_range,
            selection,
            variables,
            return_types: lowering
                .inferred_function_returns
                .get(&func_name)
                .cloned()
                .unwrap_or_else(|| {
                    outputs
                        .iter()
                        .filter_map(|o| lowering.var_types.get(o.0).cloned())
                        .collect()
                }),
        };
        functions.push(semantic);
    }

    for (idx, func) in functions.iter().enumerate() {
        function_lookup
            .entry(func.name.clone())
            .or_default()
            .push(idx);
    }

    SemanticModel {
        globals,
        functions,
        function_lookup,
        status_message: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hover_returns_builtin_docs() {
        let text = "plot(1, 2);";
        let analysis = analyze_document(text);
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
                        .contains("Docs: https://runmat.org/docs/reference/builtins/"),
                    "expected docs link, got:\n{}",
                    markup.value
                );
            }
            other => panic!("expected Markup hover contents, got {other:?}"),
        }
    }

    #[test]
    fn hover_includes_inferred_tensor_shape() {
        let text = "x = 0:1:100; y = sin(x);";
        let analysis = analyze_document(text);
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
        assert!(
            x_value.contains("Some(101)"),
            "unexpected x hover {x_value}"
        );
        assert!(y_value.contains("Tensor"), "unexpected y hover {y_value}");
        assert!(
            y_value.contains("Some(101)"),
            "unexpected y hover {y_value}"
        );
    }
}
