use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact};

pub(super) fn input(kind: &ValueKindFact) -> Option<NumericFact> {
    match kind {
        ValueKindFact::Numeric(numeric) => Some(*numeric),
        ValueKindFact::Logical | ValueKindFact::Character => Some(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => None,
    }
}

pub(super) fn output(
    left: &ValueFact,
    left_numeric: NumericFact,
    right: &ValueFact,
    right_numeric: NumericFact,
) -> Option<NumericClass> {
    let left_integer = left_numeric.class.integer_class().is_some();
    let right_integer = right_numeric.class.integer_class().is_some();
    if left_integer || right_integer {
        return match (left_integer, right_integer) {
            (true, true) if left_numeric.class == right_numeric.class => Some(left_numeric.class),
            (true, false) if right.is_scalar() && right_numeric.class == NumericClass::Double => {
                Some(left_numeric.class)
            }
            (false, true) if left.is_scalar() && left_numeric.class == NumericClass::Double => {
                Some(right_numeric.class)
            }
            _ => None,
        };
    }
    Some(
        if left_numeric.class == NumericClass::Single || right_numeric.class == NumericClass::Single
        {
            NumericClass::Single
        } else {
            NumericClass::Double
        },
    )
}
