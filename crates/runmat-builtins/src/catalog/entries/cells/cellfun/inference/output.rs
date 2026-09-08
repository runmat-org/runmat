use runmat_types::{
    CellFact, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

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

pub(super) fn uniform(mut element: ValueFact, shape: ShapeFact) -> ValueFact {
    if shape.element_count() == Some(0) {
        return ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Dense,
        );
    }
    element.shape = shape;
    element.storage = if element.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    element.residency = ResidencyFact::Host;
    element
}
