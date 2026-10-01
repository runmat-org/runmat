//! Option-value admission and normalization facts shared by statistics option inference.

use super::{StatOptionField, StatOptionValueKind};
use crate::catalog::inference::{gathered_value_fact, possible_character_row};
use runmat_types::{
    DynamicReason, FactJoin, NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact,
};

pub(crate) fn runtime_empty_fact(value: &ValueFact) -> Option<bool> {
    match &value.kind {
        ValueKindFact::Numeric(_)
        | ValueKindFact::Logical
        | ValueKindFact::Cell(_)
        | ValueKindFact::String
        | ValueKindFact::Character => value.shape.element_count().map(|count| count == 0),
        ValueKindFact::Unknown => None,
        _ => Some(false),
    }
}

pub(super) fn normalized_value(field: StatOptionField, input: &ValueFact) -> ValueFact {
    let gathered = gathered_value_fact(input);
    if runtime_empty_fact(input) == Some(true) {
        return gathered;
    }
    // Empty values bypass normalization and retain their original class/shape.
    if runtime_empty_fact(input).is_none()
        && !matches!(
            field.value_kind(),
            StatOptionValueKind::PositiveArray | StatOptionValueKind::Opaque
        )
    {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    match field.value_kind() {
        StatOptionValueKind::PositiveArray
            if input
                .numeric()
                .is_some_and(|numeric| numeric.class.integer_class().is_some())
                && input.shape.element_count().is_none_or(|count| count == 1) =>
        {
            // Scalar integer carriers normalize to Double; integer tensors
            // retain storage. The numeric fact does not distinguish carriers.
            gathered.join(&ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })))
        }
        StatOptionValueKind::PositiveArray if matches!(input.kind, ValueKindFact::Logical) => {
            if input.shape.element_count() != Some(1) {
                return ValueFact::unknown(DynamicReason::RuntimeValue);
            }
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))
        }
        StatOptionValueKind::PositiveCount | StatOptionValueKind::PositiveScalar => {
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))
        }
        StatOptionValueKind::Boolean => ValueFact::scalar(ValueKindFact::Logical),
        StatOptionValueKind::Display
        | StatOptionValueKind::OnOff
        | StatOptionValueKind::ToleranceType => ValueFact::scalar(ValueKindFact::String),
        StatOptionValueKind::RobustWeight => match &gathered.kind {
            ValueKindFact::Callable(_) => gathered,
            ValueKindFact::String | ValueKindFact::Character => {
                ValueFact::scalar(ValueKindFact::String)
            }
            _ => ValueFact::unknown(DynamicReason::RuntimeValue),
        },
        StatOptionValueKind::PositiveArray | StatOptionValueKind::Opaque => gathered,
    }
}

pub(super) fn invalid_option_value(field: StatOptionField, input: &ValueFact) -> bool {
    if runtime_empty_fact(input) == Some(true) || matches!(input.kind, ValueKindFact::Unknown) {
        return false;
    }
    if !matches!(
        field.value_kind(),
        StatOptionValueKind::PositiveArray | StatOptionValueKind::Opaque
    ) && !matches!(input.kind, ValueKindFact::Character)
        && input.shape.element_count().is_some_and(|count| count > 1)
    {
        return true;
    }
    match field.value_kind() {
        StatOptionValueKind::PositiveCount
        | StatOptionValueKind::PositiveScalar
        | StatOptionValueKind::PositiveArray => match &input.kind {
            ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Complex,
            ValueKindFact::Logical => input.shape.element_count().is_some_and(|count| count > 1),
            _ => true,
        },
        StatOptionValueKind::Boolean => match &input.kind {
            ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Complex,
            ValueKindFact::Logical | ValueKindFact::String => false,
            ValueKindFact::Character => !possible_character_row(&input.shape),
            _ => true,
        },
        StatOptionValueKind::Display
        | StatOptionValueKind::OnOff
        | StatOptionValueKind::ToleranceType => match input.kind {
            ValueKindFact::String => false,
            ValueKindFact::Character => !possible_character_row(&input.shape),
            _ => true,
        },
        StatOptionValueKind::RobustWeight => match input.kind {
            ValueKindFact::String | ValueKindFact::Callable(_) => false,
            ValueKindFact::Character => !possible_character_row(&input.shape),
            _ => true,
        },
        StatOptionValueKind::Opaque => false,
    }
}
