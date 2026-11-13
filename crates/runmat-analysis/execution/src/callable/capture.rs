use runmat_execution::value::{ElementType, InlineValue, ValuePayload};
use runmat_types::{
    DimensionFact, NumericClass, NumericDomain, ShapeFact, ValueFact, ValueKindFact,
};

use crate::{AnalysisBridgeError, AnalysisBridgeResult};

pub(super) fn validate_capture_contract(
    captures: &[ValuePayload],
    facts: &[ValueFact],
) -> AnalysisBridgeResult<()> {
    validate_capture_contract_iter(captures.iter(), facts)
}

pub(super) fn validate_capture_contract_iter<'a>(
    captures: impl IntoIterator<Item = &'a ValuePayload>,
    facts: &[ValueFact],
) -> AnalysisBridgeResult<()> {
    for (capture, fact) in captures.into_iter().zip(facts) {
        let compatible = match (&fact.kind, capture) {
            (ValueKindFact::Unknown, _) => true,
            (ValueKindFact::Logical, ValuePayload::Inline(value)) => match value.as_ref() {
                InlineValue::Logical(_) => accepts_scalar(&fact.shape),
                InlineValue::Dense(value) if value.element_type == ElementType::Logical => {
                    dimensions_match(&value.shape, &fact.shape)
                }
                _ => false,
            },
            (ValueKindFact::String, ValuePayload::Inline(value)) => {
                matches!(value.as_ref(), InlineValue::String(_)) && accepts_scalar(&fact.shape)
            }
            (ValueKindFact::Numeric(expected), ValuePayload::Inline(value)) => {
                portable_numeric_fact(value.as_ref()) == Some(*expected)
                    && portable_shape_matches(value.as_ref(), &fact.shape)
            }
            _ => false,
        };
        if !compatible {
            return Err(AnalysisBridgeError::Invalid(
                "frozen callable capture does not match its exact static type and shape".into(),
            ));
        }
    }
    Ok(())
}

fn portable_numeric_fact(value: &InlineValue) -> Option<runmat_types::NumericFact> {
    let (class, domain) = match value {
        InlineValue::F64Bits(_) => (NumericClass::Double, NumericDomain::Real),
        InlineValue::ComplexF64Bits { .. } => (NumericClass::Double, NumericDomain::Complex),
        InlineValue::I8(_) => (NumericClass::Int8, NumericDomain::Real),
        InlineValue::I16(_) => (NumericClass::Int16, NumericDomain::Real),
        InlineValue::I32(_) => (NumericClass::Int32, NumericDomain::Real),
        InlineValue::I64(_) => (NumericClass::Int64, NumericDomain::Real),
        InlineValue::U8(_) => (NumericClass::UInt8, NumericDomain::Real),
        InlineValue::U16(_) => (NumericClass::UInt16, NumericDomain::Real),
        InlineValue::U32(_) => (NumericClass::UInt32, NumericDomain::Real),
        InlineValue::U64(_) => (NumericClass::UInt64, NumericDomain::Real),
        InlineValue::Dense(value) => return element_numeric_fact(value.element_type),
        InlineValue::Sparse(value) => return element_numeric_fact(value.element_type),
        _ => return None,
    };
    Some(runmat_types::NumericFact { class, domain })
}

fn element_numeric_fact(element: ElementType) -> Option<runmat_types::NumericFact> {
    let (class, domain) = match element {
        ElementType::F32 => (NumericClass::Single, NumericDomain::Real),
        ElementType::F64 => (NumericClass::Double, NumericDomain::Real),
        ElementType::ComplexF32 => (NumericClass::Single, NumericDomain::Complex),
        ElementType::ComplexF64 => (NumericClass::Double, NumericDomain::Complex),
        ElementType::I8 => (NumericClass::Int8, NumericDomain::Real),
        ElementType::I16 => (NumericClass::Int16, NumericDomain::Real),
        ElementType::I32 => (NumericClass::Int32, NumericDomain::Real),
        ElementType::I64 => (NumericClass::Int64, NumericDomain::Real),
        ElementType::U8 => (NumericClass::UInt8, NumericDomain::Real),
        ElementType::U16 => (NumericClass::UInt16, NumericDomain::Real),
        ElementType::U32 => (NumericClass::UInt32, NumericDomain::Real),
        ElementType::U64 => (NumericClass::UInt64, NumericDomain::Real),
        ElementType::ComplexI8 => (NumericClass::Int8, NumericDomain::Complex),
        ElementType::ComplexI16 => (NumericClass::Int16, NumericDomain::Complex),
        ElementType::ComplexI32 => (NumericClass::Int32, NumericDomain::Complex),
        ElementType::ComplexI64 => (NumericClass::Int64, NumericDomain::Complex),
        ElementType::ComplexU8 => (NumericClass::UInt8, NumericDomain::Complex),
        ElementType::ComplexU16 => (NumericClass::UInt16, NumericDomain::Complex),
        ElementType::ComplexU32 => (NumericClass::UInt32, NumericDomain::Complex),
        ElementType::ComplexU64 => (NumericClass::UInt64, NumericDomain::Complex),
        ElementType::Logical => return None,
    };
    Some(runmat_types::NumericFact { class, domain })
}

fn portable_shape_matches(value: &InlineValue, expected: &ShapeFact) -> bool {
    match value {
        InlineValue::F64Bits(_)
        | InlineValue::ComplexF64Bits { .. }
        | InlineValue::I8(_)
        | InlineValue::I16(_)
        | InlineValue::I32(_)
        | InlineValue::I64(_)
        | InlineValue::U8(_)
        | InlineValue::U16(_)
        | InlineValue::U32(_)
        | InlineValue::U64(_) => accepts_scalar(expected),
        InlineValue::Dense(value) => dimensions_match(&value.shape, expected),
        InlineValue::Sparse(value) => dimensions_match(&[value.rows, value.columns], expected),
        _ => false,
    }
}

fn accepts_scalar(shape: &ShapeFact) -> bool {
    matches!(shape, ShapeFact::Unknown | ShapeFact::Scalar)
        || matches!(
            shape,
            ShapeFact::Shaped { dims }
                if !dims.is_empty()
                    && dims
                        .iter()
                        .all(|dimension| matches!(dimension, DimensionFact::Known(1)))
        )
}

fn dimensions_match(actual: &[u64], expected: &ShapeFact) -> bool {
    match expected {
        ShapeFact::Unknown => true,
        ShapeFact::Scalar => actual.iter().all(|extent| *extent == 1),
        ShapeFact::Ranked { rank } => actual.len() == *rank,
        ShapeFact::Shaped { dims } => {
            actual.len() == dims.len()
                && actual.iter().zip(dims).all(|(actual, expected)| {
                    !matches!(expected, DimensionFact::Known(value) if u64::try_from(*value).ok() != Some(*actual))
                })
        }
    }
}
