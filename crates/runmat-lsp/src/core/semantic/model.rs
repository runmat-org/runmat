use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use runmat_hir::{CallKind, FunctionKind, HirDiagnostic, LoweringResult, ReferenceKind};
use runmat_lexer::{SpannedToken, Token};

use crate::core::semantic_tokens::{IdentifierRole, SemanticHint};

#[derive(Clone, Copy, Debug)]
pub struct TextRange {
    pub start: usize,
    pub end: usize,
}

impl TextRange {
    pub fn contains(&self, offset: usize) -> bool {
        self.start <= offset && offset < self.end
    }

    pub fn to_lsp_range(self, text: &str) -> lsp_types::Range {
        lsp_types::Range {
            start: crate::core::position::offset_to_position(text, self.start),
            end: crate::core::position::offset_to_position(text, self.end),
        }
    }
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
        let mut output = String::new();
        if self.outputs.len() == 1 {
            let _ = write!(output, "{} = ", self.outputs[0]);
        } else if !self.outputs.is_empty() {
            let _ = write!(output, "[{}] = ", self.outputs.join(", "));
        }
        let _ = write!(output, "{}({})", self.name, self.inputs.join(", "));
        output
    }
}

#[derive(Clone)]
pub struct VariableSymbol {
    pub name: String,
    pub binding: runmat_hir::BindingId,
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
            Self::Global => "global",
            Self::Parameter => "parameter",
            Self::Output => "output",
            Self::Local => "local",
        }
    }
}

#[derive(Clone)]
pub struct FunctionSemantic {
    pub function: runmat_types::ProgramFunctionId,
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
    pub facts: Option<runmat_static_analysis::semantic::SemanticDocumentFacts>,
}

impl AnalysisModel {
    pub(crate) fn function_at_offset(&self, offset: usize) -> Option<&FunctionSemantic> {
        self.functions
            .iter()
            .find(|function| function.range.contains(offset))
    }
}

pub(crate) fn build(
    frontend: &runmat_static_analysis::frontend::FrontendAnalysis,
    tokens: &[SpannedToken],
    text: &str,
) -> AnalysisModel {
    let lowering = frontend
        .lowering
        .as_ref()
        .expect("semantic model requires successful HIR lowering");
    let mut functions = Vec::new();
    let mut globals = HashMap::new();

    for binding in &lowering.assembly.bindings {
        if matches!(
            binding.workspace_visibility,
            runmat_hir::WorkspaceVisibility::Hidden
        ) {
            continue;
        }
        globals.insert(
            binding.name.0.clone(),
            VariableSymbol {
                name: binding.name.0.clone(),
                binding: binding.id,
                kind: VariableKind::Global,
                declared_span: span_to_text_range(binding.declared_span, text.len()),
            },
        );
    }

    for function in &lowering.assembly.functions {
        if matches!(function.kind, FunctionKind::SyntheticEntrypoint) {
            continue;
        }
        let name = function.name.0.clone();
        let mut variables = HashMap::new();
        insert_bindings(
            &mut variables,
            lowering,
            &function.params,
            VariableKind::Parameter,
            text.len(),
        );
        insert_bindings(
            &mut variables,
            lowering,
            &function.outputs,
            VariableKind::Output,
            text.len(),
        );
        insert_bindings(
            &mut variables,
            lowering,
            &function.locals,
            VariableKind::Local,
            text.len(),
        );
        let captures = function
            .captures
            .iter()
            .map(|capture| capture.binding)
            .collect::<Vec<_>>();
        insert_bindings(
            &mut variables,
            lowering,
            &captures,
            VariableKind::Local,
            text.len(),
        );
        let name_range =
            find_symbol_range(tokens, &name, None).unwrap_or(TextRange { start: 0, end: 0 });
        let signature = FunctionSignature {
            name: name.clone(),
            outputs: binding_names(lowering, &function.outputs),
            inputs: binding_names(lowering, &function.params),
            name_range,
        };
        let Ok(function_ordinal) = u32::try_from(function.id.0) else {
            continue;
        };
        functions.push(FunctionSemantic {
            function: runmat_types::ProgramFunctionId(function_ordinal),
            name,
            signature,
            range: TextRange {
                start: function.span.start,
                end: function.span.end.min(text.len()),
            },
            selection: name_range,
            variables,
        });
    }

    let mut function_lookup: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, function) in functions.iter().enumerate() {
        function_lookup
            .entry(function.name.clone())
            .or_default()
            .push(index);
    }
    AnalysisModel {
        globals,
        function_lookup,
        token_hints: semantic_hints(
            lowering,
            frontend.semantic_facts.as_ref(),
            tokens,
            &functions,
        ),
        exported_symbols: functions
            .iter()
            .map(|function| function.name.clone())
            .collect(),
        referenced_symbols: referenced_symbols(lowering),
        functions,
        status_message: String::new(),
        diagnostics: frontend.diagnostics.clone(),
        facts: frontend.semantic_facts.clone(),
    }
}

fn insert_bindings(
    variables: &mut HashMap<String, VariableSymbol>,
    lowering: &LoweringResult,
    bindings: &[runmat_hir::BindingId],
    kind: VariableKind,
    text_len: usize,
) {
    for binding_id in bindings {
        let Some(binding) = lowering.assembly.bindings.get(binding_id.0) else {
            continue;
        };
        variables
            .entry(binding.name.0.clone())
            .or_insert_with(|| VariableSymbol {
                name: binding.name.0.clone(),
                binding: *binding_id,
                kind,
                declared_span: span_to_text_range(binding.declared_span, text_len),
            });
    }
}

fn binding_names(lowering: &LoweringResult, bindings: &[runmat_hir::BindingId]) -> Vec<String> {
    bindings
        .iter()
        .filter_map(|binding| lowering.assembly.bindings.get(binding.0))
        .map(|binding| binding.name.0.clone())
        .collect()
}

fn semantic_hints(
    lowering: &LoweringResult,
    facts: Option<&runmat_static_analysis::semantic::SemanticDocumentFacts>,
    tokens: &[SpannedToken],
    functions: &[FunctionSemantic],
) -> Vec<SemanticHint> {
    let mut hints = HashMap::new();
    for function in functions {
        insert_hint(
            &mut hints,
            function.signature.name_range,
            IdentifierRole::Function,
            true,
            false,
            80,
        );
        for token in tokens.iter().filter(|token| {
            matches!(token.token, Token::Ident) && function.range.contains(token.start)
        }) {
            let Some(variable) = function.variables.get(&token.lexeme) else {
                continue;
            };
            let callable = facts
                .and_then(|facts| facts.fact_at(variable.binding, token.start))
                .and_then(|observation| observation.fact.as_ref())
                .is_some_and(|fact| matches!(fact.kind, runmat_types::ValueKindFact::Callable(_)));
            let role = if callable {
                IdentifierRole::Function
            } else if variable.kind == VariableKind::Parameter {
                IdentifierRole::Parameter
            } else {
                IdentifierRole::Variable
            };
            let declaration = variable
                .declared_span
                .is_some_and(|span| span.start == token.start && span.end == token.end);
            insert_hint(
                &mut hints,
                TextRange {
                    start: token.start,
                    end: token.end,
                },
                role,
                declaration,
                false,
                20,
            );
        }
    }
    for reference in &lowering.hir_index.references {
        let role = match &reference.kind {
            ReferenceKind::Imported(_) | ReferenceKind::Package(_) => IdentifierRole::Namespace,
            _ => continue,
        };
        if let Some(range) = token_range(tokens, &reference.name.0, reference.span) {
            insert_hint(&mut hints, range, role, false, false, 50);
        }
    }
    for call in &lowering.hir_index.calls {
        let Some(name) = call.name.display_name() else {
            continue;
        };
        let Some(range) = token_range(tokens, &name, call.span) else {
            continue;
        };
        let builtin = matches!(call.kind, CallKind::Builtin(_));
        insert_hint(
            &mut hints,
            range,
            IdentifierRole::Function,
            false,
            builtin,
            if builtin { 95 } else { 70 },
        );
    }
    hints.into_values().map(|(_, hint)| hint).collect()
}

fn referenced_symbols(lowering: &LoweringResult) -> HashSet<String> {
    let mut symbols = HashSet::new();
    for call in &lowering.hir_index.calls {
        if let Some(name) = call.name.display_name() {
            symbols.insert(name);
        }
        if let CallKind::PackageFunction(path) = &call.kind {
            let item = path.display_name();
            let module = path.module.display_name();
            if let Some(item) = &item {
                symbols.insert(item.clone());
            }
            if let (Some(module), Some(item)) = (module, item) {
                symbols.insert(format!("{module}.{item}"));
                symbols.insert(format!("{}.{}.{item}", path.package.0, module));
            }
        }
    }
    symbols
}

fn insert_hint(
    hints: &mut HashMap<(usize, usize), (u8, SemanticHint)>,
    range: TextRange,
    role: IdentifierRole,
    declaration: bool,
    default_library: bool,
    priority: u8,
) {
    let value = SemanticHint {
        start: range.start,
        end: range.end,
        role,
        declaration,
        default_library,
    };
    match hints.get(&(range.start, range.end)) {
        Some((existing, _)) if *existing >= priority => {}
        _ => {
            hints.insert((range.start, range.end), (priority, value));
        }
    }
}

fn token_range(tokens: &[SpannedToken], lexeme: &str, span: runmat_hir::Span) -> Option<TextRange> {
    tokens
        .iter()
        .filter(|token| matches!(token.token, Token::Ident))
        .find(|token| token.lexeme == lexeme && token.start >= span.start && token.end <= span.end)
        .map(|token| TextRange {
            start: token.start,
            end: token.end,
        })
}

fn find_symbol_range(
    tokens: &[SpannedToken],
    name: &str,
    scope: Option<&TextRange>,
) -> Option<TextRange> {
    tokens
        .iter()
        .filter(|token| matches!(token.token, Token::Ident) && token.lexeme == name)
        .map(|token| TextRange {
            start: token.start,
            end: token.end,
        })
        .find(|range| scope.is_none_or(|scope| scope.contains(range.start)))
}

fn span_to_text_range(span: runmat_hir::Span, text_len: usize) -> Option<TextRange> {
    (span.end > span.start && span.start < text_len).then_some(TextRange {
        start: span.start,
        end: span.end.min(text_len),
    })
}
