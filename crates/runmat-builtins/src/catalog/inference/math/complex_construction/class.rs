use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ComponentClass {
    Known(NumericClass),
    Unknown,
    Invalid,
}

pub(super) fn unary(input: &ValueFact) -> ComponentClass {
    match input.kind {
        ValueKindFact::Numeric(numeric) => ComponentClass::Known(numeric.class),
        ValueKindFact::Logical => ComponentClass::Known(NumericClass::Double),
        ValueKindFact::Unknown => ComponentClass::Unknown,
        _ => ComponentClass::Invalid,
    }
}

pub(super) fn binary(left: &ValueFact, right: &ValueFact) -> ComponentClass {
    let left = real_component(left);
    let right = real_component(right);
    match (left, right) {
        (ComponentClass::Invalid, _) | (_, ComponentClass::Invalid) => ComponentClass::Invalid,
        (ComponentClass::Unknown, _) | (_, ComponentClass::Unknown) => ComponentClass::Unknown,
        (ComponentClass::Known(left), ComponentClass::Known(right)) => combine(left, right),
    }
}

pub(super) fn output_kind(class: NumericClass) -> ValueKindFact {
    ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Complex,
    })
}

pub(super) fn valid_integer_component(input: &ValueFact, output_class: NumericClass) -> bool {
    matches!(
        input.kind,
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Real
                && (numeric.class == output_class
                    || (numeric.class == NumericClass::Double && input.is_scalar()))
    )
}

fn real_component(input: &ValueFact) -> ComponentClass {
    match input.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            class,
        }) => ComponentClass::Known(class),
        ValueKindFact::Logical => ComponentClass::Known(NumericClass::Double),
        ValueKindFact::Unknown => ComponentClass::Unknown,
        _ => ComponentClass::Invalid,
    }
}

fn combine(left: NumericClass, right: NumericClass) -> ComponentClass {
    match (left.integer_class(), right.integer_class()) {
        (Some(_), Some(_)) if left == right => ComponentClass::Known(left),
        (Some(_), Some(_)) => ComponentClass::Invalid,
        (Some(_), None) if right == NumericClass::Double => ComponentClass::Known(left),
        (None, Some(_)) if left == NumericClass::Double => ComponentClass::Known(right),
        (Some(_), None) | (None, Some(_)) => ComponentClass::Invalid,
        (None, None) if left == NumericClass::Single || right == NumericClass::Single => {
            ComponentClass::Known(NumericClass::Single)
        }
        (None, None) => ComponentClass::Known(NumericClass::Double),
    }
}
