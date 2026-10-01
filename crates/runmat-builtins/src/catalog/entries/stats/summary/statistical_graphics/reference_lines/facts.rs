//! Conservative facts for line handles: zero, one, or several may be returned.
use runmat_types::{
    DynamicReason, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn handles() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Unknown,
        StorageFact::Unknown,
    )
}

pub(super) fn dynamic() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}
