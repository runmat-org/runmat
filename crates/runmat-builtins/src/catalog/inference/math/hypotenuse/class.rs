use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueKindFact};

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

pub(super) fn output(left: NumericFact, right: NumericFact) -> NumericClass {
    if left.class == NumericClass::Single && right.class == NumericClass::Single {
        NumericClass::Single
    } else {
        NumericClass::Double
    }
}
