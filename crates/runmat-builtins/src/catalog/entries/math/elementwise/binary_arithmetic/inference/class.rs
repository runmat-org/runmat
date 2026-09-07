use super::policy::{BinaryArithmeticInferencePolicy, RealResultDomain};
use runmat_types::{
    DynamicReason, NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact,
};

pub(super) fn result(
    policy: BinaryArithmeticInferencePolicy,
    left: &ValueFact,
    right: &ValueFact,
) -> Result<ValueKindFact, DynamicReason> {
    if matches!(left.kind, ValueKindFact::Unknown) || matches!(right.kind, ValueKindFact::Unknown) {
        return Err(DynamicReason::RuntimeValue);
    }
    if matches!(left.kind, ValueKindFact::Symbolic) || matches!(right.kind, ValueKindFact::Symbolic)
    {
        return Ok(ValueKindFact::Symbolic);
    }
    let left_numeric = arithmetic_input(left)?;
    let right_numeric = arithmetic_input(right)?;
    let class = output_class(left_numeric.class, right_numeric.class, left, right)?;
    let domain = output_domain(policy, left_numeric.domain, right_numeric.domain)?;
    Ok(ValueKindFact::Numeric(NumericFact { class, domain }))
}

fn arithmetic_input(value: &ValueFact) -> Result<NumericFact, DynamicReason> {
    match value.kind {
        ValueKindFact::Numeric(numeric) => Ok(numeric),
        ValueKindFact::Logical | ValueKindFact::Character => Ok(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => Err(DynamicReason::UnsupportedRepresentation),
    }
}

fn output_class(
    left: NumericClass,
    right: NumericClass,
    left_fact: &ValueFact,
    right_fact: &ValueFact,
) -> Result<NumericClass, DynamicReason> {
    match (left.integer_class(), right.integer_class()) {
        (Some(left_integer), Some(right_integer)) if left_integer == right_integer => Ok(left),
        (Some(_), Some(_)) => Err(DynamicReason::UnsupportedRepresentation),
        (Some(_), None) if right == NumericClass::Double && right_fact.is_scalar() => Ok(left),
        (None, Some(_)) if left == NumericClass::Double && left_fact.is_scalar() => Ok(right),
        (Some(_), None) | (None, Some(_)) => Err(DynamicReason::UnsupportedRepresentation),
        (None, None) if left == NumericClass::Single || right == NumericClass::Single => {
            Ok(NumericClass::Single)
        }
        (None, None) => Ok(NumericClass::Double),
    }
}

fn output_domain(
    policy: BinaryArithmeticInferencePolicy,
    left: NumericDomain,
    right: NumericDomain,
) -> Result<NumericDomain, DynamicReason> {
    if left == NumericDomain::Complex || right == NumericDomain::Complex {
        return Ok(NumericDomain::Complex);
    }
    match policy.real_result_domain {
        RealResultDomain::Real => Ok(NumericDomain::Real),
        RealResultDomain::RuntimeDependent => Err(DynamicReason::RuntimeValue),
    }
}
