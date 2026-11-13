use super::*;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LabelKind {
    Numeric,
    Integer(NumericDType),
    String,
    Char,
    Cell,
    Logical,
    Categorical,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Label {
    MissingString,
    Numeric(f64),
    Integer(IntValue),
    Text(String),
    Logical(bool),
}

#[derive(Clone, Debug)]
pub(super) struct LabelVector {
    pub(super) labels: Vec<Label>,
    pub(super) kind: LabelKind,
    pub(super) categories: Option<Vec<String>>,
}

#[derive(Eq, PartialEq, Hash)]
pub(super) enum LabelKey {
    MissingString,
    Numeric(u64),
    Integer(i128),
    Text(String),
    Logical(bool),
}

pub(super) fn label_key(label: &Label) -> LabelKey {
    match label {
        Label::MissingString => LabelKey::MissingString,
        Label::Numeric(value) if *value == 0.0 => LabelKey::Numeric(0),
        Label::Numeric(value) => LabelKey::Numeric(value.to_bits()),
        Label::Integer(value) => LabelKey::Integer(integer_sort_key(value)),
        Label::Text(value) => LabelKey::Text(value.clone()),
        Label::Logical(value) => LabelKey::Logical(*value),
    }
}

pub(super) fn same_label(left: &Label, right: &Label) -> bool {
    match (left, right) {
        (Label::Numeric(a), Label::Numeric(b)) => (a == b) || (a.is_nan() && b.is_nan()),
        (Label::Integer(a), Label::Integer(b)) => a == b,
        (Label::Text(a), Label::Text(b)) => a == b,
        (Label::Logical(a), Label::Logical(b)) => a == b,
        _ => false,
    }
}

pub(super) fn is_missing_label(label: &Label) -> bool {
    match label {
        Label::Numeric(value) => value.is_nan(),
        Label::Integer(_) => false,
        Label::MissingString => true,
        Label::Text(value) => value.is_empty(),
        Label::Logical(_) => false,
    }
}

pub(super) fn label_kinds_compatible(left: LabelKind, right: LabelKind) -> bool {
    left == right || (is_text_kind(left) && is_text_kind(right))
}

pub(super) fn is_text_kind(kind: LabelKind) -> bool {
    matches!(
        kind,
        LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical
    )
}
