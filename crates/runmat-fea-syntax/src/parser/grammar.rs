use crate::lexical::{FeaKeyword, FeaTokenKind};
use crate::syntax::{FeaDeclarationKind, FeaExpressionKind, FeaImportKind, FeaName};
use runmat_types::Span;

pub(super) enum BlockMemberEnd {
    Nested(usize),
    Statement(usize),
    Close(usize),
}

pub(super) fn declaration_kind(keyword: FeaKeyword) -> Option<FeaDeclarationKind> {
    Some(match keyword {
        FeaKeyword::Geometry => FeaDeclarationKind::Geometry,
        FeaKeyword::Mesh => FeaDeclarationKind::Mesh,
        FeaKeyword::Material => FeaDeclarationKind::Material,
        FeaKeyword::Model => FeaDeclarationKind::Model,
        FeaKeyword::Study => FeaDeclarationKind::Study,
        FeaKeyword::Sweep => FeaDeclarationKind::Sweep,
        FeaKeyword::Execution => FeaDeclarationKind::Execution,
        FeaKeyword::Table => FeaDeclarationKind::Table,
        _ => return None,
    })
}

pub(super) fn import_kind(kind: FeaTokenKind) -> Option<FeaImportKind> {
    Some(match kind {
        FeaTokenKind::Keyword(FeaKeyword::Geometry) => FeaImportKind::Geometry,
        FeaTokenKind::Keyword(FeaKeyword::Mesh) => FeaImportKind::Mesh,
        FeaTokenKind::Keyword(FeaKeyword::Material) => FeaImportKind::Material,
        FeaTokenKind::Keyword(FeaKeyword::Model) => FeaImportKind::Model,
        FeaTokenKind::Keyword(FeaKeyword::Study) => FeaImportKind::Study,
        FeaTokenKind::Keyword(FeaKeyword::Sweep) => FeaImportKind::Sweep,
        FeaTokenKind::Keyword(FeaKeyword::Execution) => FeaImportKind::Execution,
        FeaTokenKind::Keyword(FeaKeyword::Function) => FeaImportKind::Function,
        _ => return None,
    })
}

pub(super) fn is_name_token(kind: FeaTokenKind) -> bool {
    matches!(kind, FeaTokenKind::Identifier | FeaTokenKind::Keyword(_))
}

pub(super) fn is_quantity_value(kind: &FeaExpressionKind) -> bool {
    matches!(
        kind,
        FeaExpressionKind::Number(_) | FeaExpressionKind::List(_) | FeaExpressionKind::Group(_)
    )
}

pub(super) fn starts_expression(kind: FeaTokenKind) -> bool {
    is_name_token(kind)
        || matches!(
            kind,
            FeaTokenKind::Number
                | FeaTokenKind::String
                | FeaTokenKind::LeftParen
                | FeaTokenKind::LeftBracket
                | FeaTokenKind::LeftBrace
                | FeaTokenKind::Plus
                | FeaTokenKind::Minus
        )
}

pub(super) fn continues_on_next_line(kind: FeaTokenKind) -> bool {
    matches!(
        kind,
        FeaTokenKind::Equal
            | FeaTokenKind::EqualEqual
            | FeaTokenKind::BangEqual
            | FeaTokenKind::Less
            | FeaTokenKind::LessEqual
            | FeaTokenKind::Greater
            | FeaTokenKind::GreaterEqual
            | FeaTokenKind::Plus
            | FeaTokenKind::Minus
            | FeaTokenKind::Star
            | FeaTokenKind::Slash
            | FeaTokenKind::Caret
            | FeaTokenKind::Range
            | FeaTokenKind::Comma
            | FeaTokenKind::Colon
            | FeaTokenKind::PlusMinus
    )
}

pub(super) fn missing_name(span: Span) -> FeaName {
    FeaName {
        text: String::new(),
        span,
    }
}

pub(crate) fn decode_string(source: &str) -> String {
    let value = source
        .strip_prefix(['"', '\''])
        .and_then(|value| value.strip_suffix(['"', '\'']))
        .unwrap_or(source);
    let mut decoded = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(current) = chars.next() {
        if current != '\\' {
            decoded.push(current);
            continue;
        }
        match chars.next() {
            Some('n') => decoded.push('\n'),
            Some('r') => decoded.push('\r'),
            Some('t') => decoded.push('\t'),
            Some(next) => decoded.push(next),
            None => decoded.push('\\'),
        }
    }
    decoded
}
