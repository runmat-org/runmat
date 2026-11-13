//! Definite option-shape rejection; contents remain dynamic until runtime admission.
use super::{MnrfitConstant, MnrfitOption};
use runmat_types::{
    CallRequest, DimensionFact, LiteralValue, NumericClass, NumericDomain, NumericFact, ShapeFact,
    ValueFact, ValueKindFact,
};

/// The last admitted Constant option wins. An unknown selector can itself be
/// Constant; a later known Constant restores a known state.
pub(super) fn constant_state(request: &CallRequest) -> Option<bool> {
    let mut state = Some(true);
    for index in (2..request.arguments.len().saturating_sub(1)).step_by(2) {
        let Some(name) = request.literals.literal_string_at(index) else {
            state = None;
            continue;
        };
        if MnrfitOption::from_public(&name) == Some(MnrfitOption::Constant) {
            state = constant_literal(
                &request.arguments[index + 1],
                request.literals.literal_args.get(index + 1),
            );
        }
    }
    state
}

fn constant_literal(value: &ValueFact, literal: Option<&LiteralValue>) -> Option<bool> {
    match (&value.kind, literal?) {
        (ValueKindFact::Logical, LiteralValue::Bool(flag)) => Some(*flag),
        (
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
                ..
            }),
            LiteralValue::Number(number),
        ) if value.shape.element_count() == Some(1) && (*number == 0.0 || *number == 1.0) => {
            Some(*number != 0.0)
        }
        (
            ValueKindFact::String | ValueKindFact::Character,
            LiteralValue::String(text) | LiteralValue::Character(text),
        ) => MnrfitConstant::from_public(text).map(MnrfitConstant::enabled),
        _ => None,
    }
}

pub(super) fn definitely_invalid_name(value: &ValueFact) -> bool {
    !matches!(
        value.kind,
        ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
    ) || text_is_nonscalar(value)
}

pub(super) fn definitely_invalid_value(option: MnrfitOption, value: &ValueFact) -> bool {
    match option {
        MnrfitOption::Model
        | MnrfitOption::Link
        | MnrfitOption::Interactions
        | MnrfitOption::Display
        | MnrfitOption::EstDisp => text_is_invalid(value),
        MnrfitOption::Constant => match value.kind {
            ValueKindFact::String | ValueKindFact::Character => text_is_nonscalar(value),
            ValueKindFact::Logical
            | ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
                ..
            }) => definitely_nonscalar(&value.shape),
            ValueKindFact::Unknown => false,
            _ => true,
        },
        MnrfitOption::Weights => {
            !real_numeric_or_logical(value) || definitely_nonvector(&value.shape)
        }
        MnrfitOption::IterationLimit
        | MnrfitOption::MaxIter
        | MnrfitOption::Tolerance
        | MnrfitOption::TolX => {
            !real_numeric_or_logical(value) || definitely_nonscalar(&value.shape)
        }
        MnrfitOption::Options => match value.kind {
            ValueKindFact::Struct(_) => definitely_nonscalar(&value.shape),
            ValueKindFact::Unknown => false,
            _ => true,
        },
    }
}

pub(super) fn definitely_invalid_constant_literal(
    value: &ValueFact,
    literal: Option<&LiteralValue>,
) -> bool {
    match (&value.kind, literal) {
        (
            ValueKindFact::String | ValueKindFact::Character,
            Some(LiteralValue::String(text) | LiteralValue::Character(text)),
        ) => MnrfitConstant::from_public(text).is_none(),
        (
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            Some(LiteralValue::Number(number)),
        ) if value.shape.element_count() == Some(1) => *number != 0.0 && *number != 1.0,
        _ => false,
    }
}

fn real_numeric_or_logical(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    )
}

fn text_is_invalid(value: &ValueFact) -> bool {
    !matches!(
        value.kind,
        ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
    ) || text_is_nonscalar(value)
}

fn text_is_nonscalar(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::String => definitely_nonscalar(&value.shape),
        ValueKindFact::Character => {
            matches!(&value.shape, ShapeFact::Shaped { dims } if matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows != 1))
        }
        _ => false,
    }
}

fn definitely_nonscalar(shape: &ShapeFact) -> bool {
    matches!(shape, ShapeFact::Shaped { dims } if dims.iter().any(|dim| matches!(dim, DimensionFact::Known(extent) if *extent != 1)))
}

fn definitely_nonvector(shape: &ShapeFact) -> bool {
    matches!(shape, ShapeFact::Shaped { dims } if dims.iter().filter(|dim| matches!(dim, DimensionFact::Known(extent) if *extent > 1)).count() > 1)
}
