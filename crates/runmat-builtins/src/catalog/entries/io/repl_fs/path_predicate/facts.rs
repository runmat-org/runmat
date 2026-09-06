use runmat_types::{ShapeFact, StorageFact, ValueFact, ValueKindFact};

pub(super) fn logical_for_paths(input: Option<&ValueFact>) -> ValueFact {
    let shape = input.map_or(ShapeFact::Scalar, |input| match input.kind {
        ValueKindFact::String | ValueKindFact::Cell(_) => input.shape.clone(),
        _ => ShapeFact::Scalar,
    });
    ValueFact::proven(ValueKindFact::Logical, shape, StorageFact::Dense)
}
