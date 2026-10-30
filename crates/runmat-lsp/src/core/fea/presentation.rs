use lsp_types::{CompletionItemKind, Range, SymbolKind};
use runmat_analysis_language::FeaSymbolKind;
use runmat_parser::fea::{FeaDeclarationKind, FeaImportKind};
use runmat_types::{OperationParameter, OperationParameterArity, OperationTypePattern, Span};

use super::super::position::offset_to_position;

pub(super) fn span_range(source: &str, span: Span) -> Range {
    let start = span.start.min(source.len());
    let end = span.end.max(start).min(source.len());
    Range::new(
        offset_to_position(source, start),
        offset_to_position(source, end),
    )
}

pub(super) fn parameter_label(parameter: &OperationParameter) -> String {
    let suffix = match parameter.arity {
        OperationParameterArity::Required => "",
        OperationParameterArity::Optional => "?",
        OperationParameterArity::Variadic => "...",
    };
    format!(
        "{}{}: {}",
        parameter.name,
        suffix,
        type_pattern_name(&parameter.value)
    )
}

fn type_pattern_name(pattern: &OperationTypePattern) -> String {
    match pattern {
        OperationTypePattern::Any => "any".to_string(),
        OperationTypePattern::NumericScalar => "number".to_string(),
        OperationTypePattern::Quantity(dimension) => format!("quantity<{dimension:?}>"),
        OperationTypePattern::VectorQuantity { dimension, length } => length.map_or_else(
            || format!("vector_quantity<{dimension:?}>"),
            |length| format!("vector_quantity<{dimension:?}, {length}>"),
        ),
        OperationTypePattern::IntegerScalar => "integer".to_string(),
        OperationTypePattern::StringScalar => "string".to_string(),
        OperationTypePattern::NumericArray => "numeric_array".to_string(),
        OperationTypePattern::LogicalArray => "logical_array".to_string(),
        OperationTypePattern::LogicalScalar => "logical".to_string(),
        OperationTypePattern::Record(fields) => format!(
            "record<{}>",
            fields
                .iter()
                .map(|(name, value)| format!("{name}: {}", type_pattern_name(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        OperationTypePattern::Size => "size".to_string(),
        OperationTypePattern::Prototype => "prototype".to_string(),
        OperationTypePattern::Axes => "axes".to_string(),
        OperationTypePattern::Style => "style".to_string(),
        OperationTypePattern::PropertyName => "property_name".to_string(),
        OperationTypePattern::PropertyValue => "property_value".to_string(),
        OperationTypePattern::Callable => "callable".to_string(),
        OperationTypePattern::Analysis(value) => format!("analysis<{value:?}>"),
        OperationTypePattern::List(value) => format!("list<{}>", type_pattern_name(value)),
    }
}

pub(super) fn active_parameter(source: &str, span: Span, offset: usize) -> usize {
    let end = offset.min(span.end).min(source.len());
    let start = span.start.min(end);
    let mut depth = 0usize;
    let mut found_open = false;
    let mut parameter = 0usize;
    for character in source[start..end].chars() {
        match character {
            '(' => {
                depth += 1;
                found_open = true;
            }
            ')' => depth = depth.saturating_sub(1),
            ',' if found_open && depth == 1 => parameter += 1,
            _ => {}
        }
    }
    parameter
}

pub(super) fn symbol_kind(kind: FeaSymbolKind) -> SymbolKind {
    match kind {
        FeaSymbolKind::Function => SymbolKind::FUNCTION,
        FeaSymbolKind::Material | FeaSymbolKind::Table => SymbolKind::STRUCT,
        FeaSymbolKind::Study | FeaSymbolKind::Sweep | FeaSymbolKind::Execution => {
            SymbolKind::OBJECT
        }
        FeaSymbolKind::Geometry | FeaSymbolKind::Mesh | FeaSymbolKind::Model => SymbolKind::CLASS,
    }
}

pub(super) fn completion_kind(kind: FeaSymbolKind) -> CompletionItemKind {
    match kind {
        FeaSymbolKind::Function => CompletionItemKind::FUNCTION,
        FeaSymbolKind::Material | FeaSymbolKind::Table => CompletionItemKind::STRUCT,
        FeaSymbolKind::Geometry | FeaSymbolKind::Mesh | FeaSymbolKind::Model => {
            CompletionItemKind::CLASS
        }
        FeaSymbolKind::Study | FeaSymbolKind::Sweep | FeaSymbolKind::Execution => {
            CompletionItemKind::VALUE
        }
    }
}

pub(super) const fn symbol_kind_name(kind: FeaSymbolKind) -> &'static str {
    kind.namespace()
}

pub(super) const fn declaration_kind_symbol(kind: FeaDeclarationKind) -> FeaSymbolKind {
    match kind {
        FeaDeclarationKind::Geometry => FeaSymbolKind::Geometry,
        FeaDeclarationKind::Mesh => FeaSymbolKind::Mesh,
        FeaDeclarationKind::Material => FeaSymbolKind::Material,
        FeaDeclarationKind::Model => FeaSymbolKind::Model,
        FeaDeclarationKind::Study => FeaSymbolKind::Study,
        FeaDeclarationKind::Sweep => FeaSymbolKind::Sweep,
        FeaDeclarationKind::Execution => FeaSymbolKind::Execution,
        FeaDeclarationKind::Table => FeaSymbolKind::Table,
    }
}

pub(super) const fn import_kind_symbol(kind: FeaImportKind) -> FeaSymbolKind {
    match kind {
        FeaImportKind::Geometry => FeaSymbolKind::Geometry,
        FeaImportKind::Mesh => FeaSymbolKind::Mesh,
        FeaImportKind::Material => FeaSymbolKind::Material,
        FeaImportKind::Model => FeaSymbolKind::Model,
        FeaImportKind::Study => FeaSymbolKind::Study,
        FeaImportKind::Sweep => FeaSymbolKind::Sweep,
        FeaImportKind::Execution => FeaSymbolKind::Execution,
        FeaImportKind::Function => FeaSymbolKind::Function,
    }
}

pub(super) const fn declaration_kind_name(kind: FeaDeclarationKind) -> &'static str {
    symbol_kind_name(declaration_kind_symbol(kind))
}

pub(super) const fn import_kind_name(kind: FeaImportKind) -> &'static str {
    symbol_kind_name(import_kind_symbol(kind))
}
