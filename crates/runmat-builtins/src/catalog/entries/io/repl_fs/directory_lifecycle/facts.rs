use runmat_types::{ShapeFact, StorageFact, ValueFact, ValueKindFact};

pub(super) fn status() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Logical)
}

pub(super) fn diagnostic_text() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}

pub(super) fn outputs() -> Vec<ValueFact> {
    vec![status(), diagnostic_text(), diagnostic_text()]
}
