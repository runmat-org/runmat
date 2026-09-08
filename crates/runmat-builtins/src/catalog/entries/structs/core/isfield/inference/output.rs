use runmat_types::{DynamicReason, ShapeFact, StorageFact, ValueFact, ValueKindFact};

pub(super) fn fact(names: Option<&ValueFact>) -> ValueFact {
    let Some(names) = names else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let (shape, storage) = match &names.kind {
        ValueKindFact::Character => (ShapeFact::Scalar, StorageFact::Scalar),
        ValueKindFact::String if names.shape == ShapeFact::Scalar => {
            (ShapeFact::Scalar, StorageFact::Scalar)
        }
        ValueKindFact::String | ValueKindFact::Cell(_) => (names.shape.clone(), StorageFact::Dense),
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    ValueFact::proven(ValueKindFact::Logical, shape, storage)
}
