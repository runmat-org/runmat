//! Abstract result representation of the existing host bit-buffer engine.
use runmat_types::{
    NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn result_kind(input: &ValueFact) -> Option<ValueKindFact> {
    if input.storage == StorageFact::Sparse
        && input
            .numeric()
            .is_some_and(|numeric| numeric.class.integer_class().is_some())
    {
        return None;
    }
    let class = match input.kind {
        ValueKindFact::Logical => NumericClass::Double,
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
            if numeric.class.integer_class().is_some() {
                numeric.class
            } else {
                NumericClass::Double
            }
        }
        _ => return None,
    };
    Some(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }))
}

pub(super) fn storage(input: &ValueFact, shape: &ShapeFact) -> StorageFact {
    // Sparse output legality depends on implicit-zero results. In particular a
    // scalar sparse input is not proof of a scalar carrier.
    if input.storage == StorageFact::Sparse {
        StorageFact::Unknown
    } else if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else if matches!(shape, ShapeFact::Unknown) {
        StorageFact::Unknown
    } else {
        StorageFact::Dense
    }
}

pub(super) fn scalar_or_equal(left: &ShapeFact, right: &ShapeFact) -> ShapeFact {
    if left.element_count() == Some(1) {
        return right.clone();
    }
    if right.element_count() == Some(1) || left == right {
        return left.clone();
    }
    let (Some(mut left), Some(mut right)) = (left.known_dims(), right.known_dims()) else {
        return ShapeFact::Unknown;
    };
    let rank = left.len().max(right.len());
    left.resize(rank, Some(1));
    right.resize(rank, Some(1));
    if left == right {
        ShapeFact::from(left)
    } else {
        ShapeFact::Unknown
    }
}
