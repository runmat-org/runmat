//! Hierarchical `.m` document symbols from the static-analysis facts (UI target decision 11):
//!
//! - `%%` sections group the items that start inside them;
//! - script variables appear once, at their first assignment, with the inferred type and shape
//!   as the detail ([`format_fact`]); each assignment is a child with the type it assigns, and a
//!   variable no statement reads carries the [`UNUSED`] marker after its detail;
//! - functions show `(inputs) → outputs` and hold their own variables and nested functions;
//! - classes hold their properties (with the analysed type) and methods.
//!
//! Assignments come from the lowered HIR; types and references come from
//! `runmat-static-analysis`'s `SemanticDocumentFacts`, so the outline, hover and quick
//! information agree.

use std::collections::BTreeMap;

use lsp_types::{DocumentSymbol, SymbolKind};
use runmat_hir::{
    BindingId, BindingRole, FunctionKind, HirBlock, HirExpr, HirExprKind, HirPlace,
    HirSequenceTarget, HirStmtKind, LoweringResult, OutputTarget, Span,
};
use runmat_static_analysis::semantic::SemanticDocumentFacts;

use super::model::AnalysisModel;
use super::presentation::format_fact;

/// Appended to a variable's detail (after ` · `) when nothing reads it.
pub const UNUSED: &str = "unused";
/// The name of an assignment child under a variable.
pub const ASSIGNMENT: &str = "assignment";

/// The `.m` outline. Sections come from the text; everything else needs a successful lowering.
pub fn document_outline(
    text: &str,
    lowering: Option<&LoweringResult>,
    semantic: Option<&AnalysisModel>,
) -> Vec<DocumentSymbol> {
    let mut items = Vec::new();
    if let Some(lowering) = lowering {
        let facts = semantic.and_then(|model| model.facts.as_ref());
        let names = selection_ranges(semantic);
        items = program_items(text, lowering, facts, &names);
    }
    group_into_sections(text, items)
}

/* ------------------------------------------------------------------------------------------ */
/* Assignments from the HIR                                                                   */
/* ------------------------------------------------------------------------------------------ */

/// Statement spans that assign each binding, in source order.
fn collect_assignments(block: &HirBlock, out: &mut BTreeMap<usize, Vec<Span>>) {
    fn record(out: &mut BTreeMap<usize, Vec<Span>>, binding: Option<BindingId>, span: Span) {
        if let Some(binding) = binding {
            out.entry(binding.0).or_default().push(span);
        }
    }
    for stmt in &block.statements {
        let assign = |out: &mut BTreeMap<usize, Vec<Span>>,
                      binding: Option<BindingId>,
                      span: Span| record(out, binding, span);
        match &stmt.kind {
            HirStmtKind::Assign(place, _, _) => assign(out, place_binding(place), stmt.span),
            HirStmtKind::MultiAssign(targets, _, _) => {
                for target in &targets.targets {
                    match target {
                        OutputTarget::Place(place) => assign(out, place_binding(place), stmt.span),
                        OutputTarget::Sequence(sequence) => {
                            assign(out, sequence_binding(sequence), stmt.span)
                        }
                        OutputTarget::Discard => {}
                    }
                }
            }
            HirStmtKind::SequenceAssign { target, .. } => {
                assign(out, sequence_binding(target), stmt.span)
            }
            HirStmtKind::If {
                then_body,
                elseif_blocks,
                else_body,
                ..
            } => {
                collect_assignments(then_body, out);
                for (_, body) in elseif_blocks {
                    collect_assignments(body, out);
                }
                if let Some(body) = else_body {
                    collect_assignments(body, out);
                }
            }
            HirStmtKind::While { body, .. } | HirStmtKind::Spmd { body, .. } => {
                collect_assignments(body, out)
            }
            HirStmtKind::For {
                binding,
                range,
                body,
            }
            | HirStmtKind::ParFor {
                binding,
                range,
                body,
                ..
            } => {
                // The loop header assigns the variable: `for t = 0:0.1:1`.
                assign(
                    out,
                    Some(*binding),
                    Span {
                        start: stmt.span.start,
                        end: range.span.end.max(stmt.span.start),
                    },
                );
                collect_assignments(body, out);
            }
            HirStmtKind::Switch {
                cases, otherwise, ..
            } => {
                for (_, body) in cases {
                    collect_assignments(body, out);
                }
                if let Some(body) = otherwise {
                    collect_assignments(body, out);
                }
            }
            HirStmtKind::TryCatch {
                try_body,
                catch_binding,
                catch_body,
            } => {
                collect_assignments(try_body, out);
                assign(out, *catch_binding, stmt.span);
                collect_assignments(catch_body, out);
            }
            HirStmtKind::ExprStmt(..)
            | HirStmtKind::Global(_)
            | HirStmtKind::Persistent(_)
            | HirStmtKind::Break
            | HirStmtKind::Continue
            | HirStmtKind::Return
            | HirStmtKind::Import(_) => {}
        }
    }
}

/// The variable an assignment target writes: `x`, `x(3)`, `s.field`, `c{2}`.
fn place_binding(place: &HirPlace) -> Option<BindingId> {
    match place {
        HirPlace::Binding(binding) => Some(*binding),
        HirPlace::Member(base, _)
        | HirPlace::MemberDynamic(base, _)
        | HirPlace::Index(base, _)
        | HirPlace::IndexCell(base, _) => expr_binding(base),
    }
}

fn sequence_binding(target: &HirSequenceTarget) -> Option<BindingId> {
    match target {
        HirSequenceTarget::Member { base, .. }
        | HirSequenceTarget::DynamicMember { base, .. }
        | HirSequenceTarget::CellContents { base, .. } => expr_binding(base),
    }
}

fn expr_binding(expr: &HirExpr) -> Option<BindingId> {
    match &expr.kind {
        HirExprKind::Binding(binding) => Some(*binding),
        HirExprKind::Index(base, _)
        | HirExprKind::Member { base, .. }
        | HirExprKind::MemberDynamic { base, .. } => expr_binding(base),
        HirExprKind::SubscriptChain(chain) => expr_binding(&chain.root),
        _ => None,
    }
}

/* ------------------------------------------------------------------------------------------ */
/* Items                                                                                      */
/* ------------------------------------------------------------------------------------------ */

/// Function name ranges the semantic model found, keyed by the function's span start.
fn selection_ranges(semantic: Option<&AnalysisModel>) -> BTreeMap<usize, (usize, usize)> {
    semantic
        .map(|model| {
            model
                .functions
                .iter()
                .filter(|function| function.selection.end > function.selection.start)
                .map(|function| {
                    (
                        function.range.start,
                        (function.selection.start, function.selection.end),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn program_items(
    text: &str,
    lowering: &LoweringResult,
    facts: Option<&SemanticDocumentFacts>,
    names: &BTreeMap<usize, (usize, usize)>,
) -> Vec<DocumentSymbol> {
    let assembly = &lowering.assembly;
    let mut items = Vec::new();
    // Functions by id, so nested functions and methods can find their parents.
    let mut function_symbols: BTreeMap<usize, DocumentSymbol> = BTreeMap::new();
    let mut parents: BTreeMap<usize, usize> = BTreeMap::new();
    let mut methods: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

    for function in &assembly.functions {
        let mut assignments = BTreeMap::new();
        collect_assignments(&function.body, &mut assignments);
        let own: Vec<BindingId> = assignments
            .keys()
            .map(|id| BindingId(*id))
            .filter(|binding| {
                assembly.bindings.get(binding.0).is_some_and(|b| {
                    matches!(b.role, BindingRole::Local | BindingRole::ModuleBinding)
                        && b.name.0 != "ans"
                })
            })
            .collect();
        let variables: Vec<DocumentSymbol> = own
            .iter()
            .filter_map(|binding| variable_symbol(text, lowering, facts, *binding, &assignments))
            .collect();
        match function.kind {
            FunctionKind::SyntheticEntrypoint => items.extend(variables),
            FunctionKind::Anonymous => {}
            FunctionKind::Named | FunctionKind::ClassMethod { .. } => {
                let inputs = binding_names(lowering, &function.params);
                let outputs = binding_names(lowering, &function.outputs);
                let outputs = match outputs.len() {
                    0 => String::new(),
                    1 => format!(" \u{2192} {}", outputs[0]),
                    _ => format!(" \u{2192} [{}]", outputs.join(", ")),
                };
                let range = clamp(function.span, text.len());
                let selection = names
                    .get(&function.span.start)
                    .copied()
                    .unwrap_or((range.0, range.0));
                let kind = if matches!(function.kind, FunctionKind::ClassMethod { .. }) {
                    SymbolKind::METHOD
                } else {
                    SymbolKind::FUNCTION
                };
                // Methods are named `Class.method`; under their class the method name reads alone.
                let name = if kind == SymbolKind::METHOD {
                    function
                        .name
                        .0
                        .rsplit('.')
                        .next()
                        .unwrap_or(&function.name.0)
                        .to_owned()
                } else {
                    function.name.0.clone()
                };
                let mut symbol = symbol(
                    text,
                    name,
                    Some(format!("({}){outputs}", inputs.join(", "))),
                    kind,
                    range,
                    selection,
                );
                if !variables.is_empty() {
                    symbol.children = Some(variables);
                }
                if let Some(parent) = function.parent {
                    parents.insert(function.id.0, parent.0);
                }
                if let Some(class) = function.enclosing_class {
                    methods.entry(class.0).or_default().push(function.id.0);
                }
                function_symbols.insert(function.id.0, symbol);
            }
        }
    }

    // Nested functions under their parents, deepest first so each parent is complete.
    let mut nested: Vec<(usize, usize)> = parents.into_iter().collect();
    nested.sort_by_key(|(child, _)| std::cmp::Reverse(*child));
    for (child, parent) in nested {
        let Some(symbol) = function_symbols.remove(&child) else {
            continue;
        };
        match function_symbols.get_mut(&parent) {
            Some(parent) => push_child(parent, symbol),
            None => {
                function_symbols.insert(child, symbol);
            }
        }
    }

    for class in &assembly.classes {
        let declaration = &class.declaration;
        let name = declaration
            .name
            .display_name()
            .unwrap_or_else(|| "class".to_owned());
        let range = clamp(declaration.span, text.len());
        let mut symbol = symbol(
            text,
            name,
            None,
            SymbolKind::CLASS,
            range,
            (range.0, range.0),
        );
        let analysed = facts.and_then(|facts| {
            facts
                .classes
                .iter()
                .find(|analysis| analysis.declaration.id == declaration.id)
        });
        for property in &declaration.properties {
            let fact = analysed.and_then(|analysis| {
                analysis
                    .properties
                    .iter()
                    .find(|p| p.property == property.name)
                    .map(|p| &p.fact)
            });
            let range = clamp(property.span, text.len());
            push_child(
                &mut symbol,
                self::symbol(
                    text,
                    property.name.0.clone(),
                    fact.map(|fact| format_fact(Some(fact))),
                    SymbolKind::PROPERTY,
                    range,
                    range,
                ),
            );
        }
        for id in methods.remove(&declaration.id.0).unwrap_or_default() {
            if let Some(method) = function_symbols.remove(&id) {
                push_child(&mut symbol, method);
            }
        }
        items.push(symbol);
    }

    items.extend(function_symbols.into_values());
    items.sort_by_key(|item| (item.range.start.line, item.range.start.character));
    items
}

fn variable_symbol(
    text: &str,
    lowering: &LoweringResult,
    facts: Option<&SemanticDocumentFacts>,
    binding: BindingId,
    assignments: &BTreeMap<usize, Vec<Span>>,
) -> Option<DocumentSymbol> {
    let hir = lowering.assembly.bindings.get(binding.0)?;
    let spans = assignments.get(&binding.0)?;
    let first = *spans.first()?;
    let fact_at = |span: Span| {
        facts
            .and_then(|facts| facts.fact_at(binding, span.start))
            .and_then(|observation| observation.fact.as_ref())
    };
    let bound = facts.and_then(|facts| facts.binding(binding));
    // A read is a reference outside every statement that assigns the variable.
    let unused = bound.is_some_and(|bound| {
        bound.references.iter().all(|reference| {
            spans.iter().any(|span| {
                reference.start >= span.start as u64 && reference.end <= span.end as u64
            })
        })
    });
    let mut detail = format_fact(fact_at(first));
    if unused {
        detail.push_str(" \u{b7} ");
        detail.push_str(UNUSED);
    }
    let range = clamp(first, text.len());
    let declared = clamp(hir.declared_span, text.len());
    let selection = if declared.1 > declared.0 && declared.0 >= range.0 && declared.1 <= range.1 {
        declared
    } else {
        range
    };
    let mut variable = symbol(
        text,
        hir.name.0.clone(),
        Some(detail),
        SymbolKind::VARIABLE,
        range,
        selection,
    );
    variable.children = Some(
        spans
            .iter()
            .map(|span| {
                let range = clamp(*span, text.len());
                symbol(
                    text,
                    ASSIGNMENT.to_owned(),
                    Some(format_fact(fact_at(*span))),
                    SymbolKind::FIELD,
                    range,
                    range,
                )
            })
            .collect(),
    );
    Some(variable)
}

fn binding_names(lowering: &LoweringResult, bindings: &[BindingId]) -> Vec<String> {
    bindings
        .iter()
        .filter_map(|binding| lowering.assembly.bindings.get(binding.0))
        .map(|binding| binding.name.0.clone())
        .collect()
}

/* ------------------------------------------------------------------------------------------ */
/* Sections                                                                                   */
/* ------------------------------------------------------------------------------------------ */

/// `%%` cell markers: (title, start offset of the line, start offset of the next section).
fn sections(text: &str) -> Vec<(String, usize, usize)> {
    let mut starts: Vec<(String, usize)> = Vec::new();
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("%%") {
            let title = rest.trim();
            let title = if title.is_empty() {
                format!("Section {}", starts.len() + 1)
            } else {
                title.to_owned()
            };
            starts.push((title, offset));
        }
        offset += line.len();
    }
    let mut out = Vec::new();
    for (index, (title, start)) in starts.iter().enumerate() {
        let end = starts.get(index + 1).map_or(text.len(), |next| next.1);
        out.push((title.clone(), *start, end));
    }
    out
}

fn group_into_sections(text: &str, items: Vec<DocumentSymbol>) -> Vec<DocumentSymbol> {
    let found = sections(text);
    if found.is_empty() {
        return items;
    }
    let first = crate::core::position::offset_to_position(text, found[0].1);
    let mut top: Vec<DocumentSymbol> = Vec::new();
    let mut groups: Vec<DocumentSymbol> = found
        .iter()
        .map(|(title, start, end)| {
            let title_end = text[*start..*end].find('\n').map_or(*end, |at| *start + at);
            symbol(
                text,
                title.clone(),
                None,
                SymbolKind::NAMESPACE,
                (*start, *end),
                (*start, title_end),
            )
        })
        .collect();
    for item in items {
        let at = item.range.start;
        if (at.line, at.character) < (first.line, first.character) {
            top.push(item);
            continue;
        }
        let index = groups
            .iter()
            .rposition(|group| {
                (group.range.start.line, group.range.start.character) <= (at.line, at.character)
            })
            .unwrap_or(0);
        push_child(&mut groups[index], item);
    }
    top.extend(groups);
    top
}

/* ------------------------------------------------------------------------------------------ */
/* Helpers                                                                                    */
/* ------------------------------------------------------------------------------------------ */

fn clamp(span: Span, len: usize) -> (usize, usize) {
    let start = span.start.min(len);
    (start, span.end.min(len).max(start))
}

fn symbol(
    text: &str,
    name: String,
    detail: Option<String>,
    kind: SymbolKind,
    range: (usize, usize),
    selection: (usize, usize),
) -> DocumentSymbol {
    let to_range = |(start, end): (usize, usize)| lsp_types::Range {
        start: crate::core::position::offset_to_position(text, start),
        end: crate::core::position::offset_to_position(text, end),
    };
    crate::core::analysis::make_document_symbol(
        name,
        detail,
        kind,
        to_range(range),
        to_range(selection),
    )
}

fn push_child(parent: &mut DocumentSymbol, child: DocumentSymbol) {
    let children = parent.children.get_or_insert_with(Vec::new);
    children.push(child);
    children.sort_by_key(|item| (item.range.start.line, item.range.start.character));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::analysis::{analyze_document_with_compat, document_symbols, CompatMode};

    fn outline(text: &str) -> Vec<DocumentSymbol> {
        let analysis = analyze_document_with_compat(text, CompatMode::default());
        document_symbols(text, &analysis)
    }

    fn names(symbols: &[DocumentSymbol]) -> Vec<String> {
        symbols.iter().map(|s| s.name.clone()).collect()
    }

    fn child<'a>(symbols: &'a [DocumentSymbol], name: &str) -> &'a DocumentSymbol {
        symbols
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("no {name} in {:?}", names(symbols)))
    }

    #[test]
    fn script_variables_show_type_and_shape_at_first_assignment() {
        let text =
            "x = linspace(0, 1, 201);\nY = zeros(201, 201);\nY = Y + 1;\nplot(x, Y(1, :));\n";
        let symbols = outline(text);
        assert_eq!(names(&symbols), ["x", "Y"]);
        let y = child(&symbols, "Y");
        assert_eq!(y.kind, SymbolKind::VARIABLE);
        assert_eq!(y.range.start.line, 1, "first assignment");
        let detail = y.detail.clone().unwrap_or_default();
        assert!(detail.contains("201 x 201"), "{detail}");
        assert!(!detail.ends_with(UNUSED), "{detail}");
        let assignments = y.children.as_deref().unwrap_or_default();
        assert_eq!(assignments.len(), 2);
        assert!(assignments.iter().all(|a| a.name == ASSIGNMENT));
        assert_eq!(assignments[1].range.start.line, 2);
    }

    #[test]
    fn variables_nothing_reads_are_tagged_unused() {
        let text = "a = 1;\nb = 2;\ndisp(a);\n";
        let symbols = outline(text);
        let b = child(&symbols, "b").detail.clone().unwrap_or_default();
        assert!(b.ends_with(&format!("\u{b7} {UNUSED}")), "{b}");
        let a = child(&symbols, "a").detail.clone().unwrap_or_default();
        assert!(!a.ends_with(UNUSED), "{a}");
    }

    #[test]
    fn sections_group_the_items_that_start_in_them() {
        let text = "%% Setup\nn = 10;\n%% Solve\nv = ones(n, 1);\ndisp(v);\n";
        let symbols = outline(text);
        assert_eq!(names(&symbols), ["Setup", "Solve"]);
        assert!(symbols.iter().all(|s| s.kind == SymbolKind::NAMESPACE));
        assert_eq!(
            names(symbols[0].children.as_deref().unwrap_or_default()),
            ["n"]
        );
        assert_eq!(
            names(symbols[1].children.as_deref().unwrap_or_default()),
            ["v"]
        );
    }

    #[test]
    fn function_files_show_inputs_and_outputs_and_their_variables() {
        let text = "function [s, p] = stats(x, w)\n  t = x .* w;\n  s = sum(t);\n  p = prod(x);\nend\n\nfunction y = helper(v)\n  y = v * 2;\nend\n";
        let symbols = outline(text);
        assert_eq!(names(&symbols), ["stats", "helper"]);
        let stats = child(&symbols, "stats");
        assert_eq!(stats.kind, SymbolKind::FUNCTION);
        assert_eq!(stats.detail.as_deref(), Some("(x, w) \u{2192} [s, p]"));
        assert!(names(stats.children.as_deref().unwrap_or_default()).contains(&"t".to_owned()));
        assert_eq!(
            child(&symbols, "helper").detail.as_deref(),
            Some("(v) \u{2192} y")
        );
    }

    #[test]
    fn classdef_files_list_properties_and_methods() {
        let text = "classdef Account\n  properties\n    balance = 0\n    owner\n  end\n  methods\n    function obj = deposit(obj, amount)\n      obj.balance = obj.balance + amount;\n    end\n  end\nend\n";
        let symbols = outline(text);
        let class = child(&symbols, "Account");
        assert_eq!(class.kind, SymbolKind::CLASS);
        let members = class.children.as_deref().unwrap_or_default();
        assert!(
            names(members).contains(&"balance".to_owned()),
            "{:?}",
            names(members)
        );
        assert!(names(members).contains(&"owner".to_owned()));
        let deposit = child(members, "deposit");
        assert_eq!(deposit.kind, SymbolKind::METHOD);
        assert_eq!(
            deposit.detail.as_deref(),
            Some("(obj, amount) \u{2192} obj")
        );
    }

    #[test]
    fn a_syntax_error_still_lists_sections() {
        let symbols = outline("%% First\nx = (1 + ;\n%% Second\n");
        assert_eq!(names(&symbols), ["First", "Second"]);
    }
}
