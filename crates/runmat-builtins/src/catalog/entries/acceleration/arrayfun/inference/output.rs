use runmat_types::{CellFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact};

pub(super) fn nonuniform(element: ValueFact, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: false,
        }),
        shape,
        StorageFact::Dense,
    )
}

pub(super) fn uniform(
    mut element: ValueFact,
    shape: ShapeFact,
    has_device_input: bool,
) -> ValueFact {
    element.shape = shape;
    element.storage = if element.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    element.residency = if has_device_input {
        ResidencyFact::Unknown
    } else {
        ResidencyFact::Host
    };
    element
}
