use super::facts::*;
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RecordMode {
    Elements,
    Rows,
    Unknown,
}
#[derive(Clone, Copy)]
pub(super) enum ElementShape {
    PreserveVector,
    BothRow,
}
#[derive(Clone, Copy)]
pub(super) enum ResultClass {
    Common,
    Left,
}

pub(super) fn binary_options(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> RecordMode {
    let mut rows = RecordMode::Elements;
    let mut order = None;
    for index in 2..request.arguments.len() {
        match text_option(request, index, diagnostics).as_deref() {
            Some("rows") => rows = RecordMode::Rows,
            Some(value @ ("sorted" | "stable")) => {
                if order
                    .as_ref()
                    .is_some_and(|previous: &String| previous != value)
                {
                    diagnostics.push(argument_error(
                        "RM-SET-ORDER",
                        "conflicting sorted/stable options",
                        index,
                    ));
                }
                order = Some(value.to_owned());
            }
            Some(_) => diagnostics.push(argument_error(
                "RM-SET-OPTION",
                "unsupported set option",
                index,
            )),
            None if rows != RecordMode::Rows => rows = RecordMode::Unknown,
            None => {}
        }
    }
    rows
}

pub(super) fn output_shape(
    a: &ValueFact,
    b: Option<&ValueFact>,
    rows: RecordMode,
    policy: ElementShape,
) -> ShapeFact {
    match rows {
        RecordMode::Rows => ShapeFact::from(vec![
            if dimension(&a.shape, 0) == Some(0)
                && b.is_none_or(|b| dimension(&b.shape, 0) == Some(0))
            {
                Some(0)
            } else {
                None
            },
            dimension(&a.shape, 1),
        ]),
        RecordMode::Unknown => ShapeFact::Unknown,
        RecordMode::Elements => {
            let empty = a.shape.element_count() == Some(0)
                && b.is_none_or(|b| b.shape.element_count() == Some(0));
            let length = if empty { Some(0) } else { None };
            let row = match policy {
                ElementShape::PreserveVector => is_row(&a.shape),
                ElementShape::BothRow => match (is_row(&a.shape), b.and_then(|b| is_row(&b.shape)))
                {
                    (Some(a), Some(b)) => Some(a && b),
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    _ => None,
                },
            };
            match row {
                Some(true) => ShapeFact::from(vec![Some(1), length]),
                Some(false) => column(length),
                None => ShapeFact::Unknown,
            }
        }
    }
}
fn is_row(shape: &ShapeFact) -> Option<bool> {
    match shape {
        ShapeFact::Scalar => Some(true),
        ShapeFact::Shaped { dims } => {
            if dims.len() == 1 {
                return Some(true);
            }
            match dimension(shape, 0) {
                Some(1) if dims.iter().skip(2).all(|d| *d == DimensionFact::Known(1)) => Some(true),
                Some(rows) if rows != 1 => Some(false),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Same-class outputs are proven without conflating logical and numeric values.
/// Mixed-class normalization stays runtime-owned unless its exact result is known.
pub(super) fn binary_kind(a: &ValueFact, b: &ValueFact) -> ValueKindFact {
    match (&a.kind, &b.kind) {
        (ValueKindFact::Numeric(a), ValueKindFact::Numeric(b)) if a == b => {
            ValueKindFact::Numeric(*a)
        }
        (ValueKindFact::Character, ValueKindFact::Character) => ValueKindFact::Character,
        (ValueKindFact::String, ValueKindFact::String) => ValueKindFact::String,
        (ValueKindFact::Logical, ValueKindFact::Logical) => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        (ValueKindFact::Numeric(a), ValueKindFact::Numeric(b)) if a.domain == b.domain => {
            if a.class == NumericClass::Double && b.class.integer_class().is_some() {
                ValueKindFact::Numeric(*b)
            } else if b.class == NumericClass::Double && a.class.integer_class().is_some() {
                ValueKindFact::Numeric(*a)
            } else if a.class == NumericClass::Double && b.class == NumericClass::Single {
                ValueKindFact::Numeric(*b)
            } else if b.class == NumericClass::Double && a.class == NumericClass::Single {
                ValueKindFact::Numeric(*a)
            } else {
                ValueKindFact::Unknown
            }
        }
        _ => ValueKindFact::Unknown,
    }
}
pub(super) fn finish_binary(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    rows: RecordMode,
    shape: ElementShape,
    class: ResultClass,
    outputs: usize,
    mut diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    count(request, 2, &mut diagnostics);
    let a = argument(request, 0);
    let b = argument(request, 1);
    accepted_data(&a, true, false, &mut diagnostics, 0);
    accepted_data(&b, true, false, &mut diagnostics, 1);
    if rows == RecordMode::Rows {
        row_compatibility(&a, &b, &mut diagnostics);
    }
    let residency = first_resident(&a, &b);
    let mut values = ValueFact::proven(
        match class {
            ResultClass::Common => binary_kind(&a, &b),
            ResultClass::Left => a.kind.clone(),
        },
        output_shape(&a, Some(&b), rows, shape),
        StorageFact::Dense,
    );
    values.residency = residency.clone();
    let mut result = vec![values];
    result.extend((1..outputs).map(|_| index(column(None), residency.clone())));
    finish_fixed_outputs(entry, request, result, diagnostics)
}
