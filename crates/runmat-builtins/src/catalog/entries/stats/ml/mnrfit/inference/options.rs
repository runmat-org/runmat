//! Definite option-shape rejection; contents remain dynamic until runtime admission.
use super::MnrfitOption;
use runmat_types::{
    DimensionFact, NumericDomain, NumericFact, ShapeFact, ValueFact, ValueKindFact,
};

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
