use runmat_types::{CellFact, DynamicReason, ShapeFact, StorageFact, ValueFact, ValueKindFact};

pub(super) fn character_row() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}

pub(super) fn character_scalar() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    )
}

pub(super) fn result_for_text(input: &ValueFact) -> ValueFact {
    match &input.kind {
        ValueKindFact::Character => character_row(),
        ValueKindFact::String => ValueFact::proven(
            ValueKindFact::String,
            input.shape.clone(),
            StorageFact::Dense,
        ),
        ValueKindFact::Cell(cell) => ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(character_row()),
                elements: cell.elements.iter().map(|_| character_row()).collect(),
                elements_complete: cell.elements_complete,
            }),
            input.shape.clone(),
            StorageFact::Dense,
        ),
        _ => ValueFact::unknown(DynamicReason::DynamicDispatch),
    }
}

pub(super) fn join_result(arguments: &[ValueFact]) -> ValueFact {
    let contains_unknown = arguments
        .iter()
        .any(|value| matches!(value.kind, ValueKindFact::Unknown));
    if let Some(value) = arguments
        .iter()
        .find(|value| matches!(value.kind, ValueKindFact::String))
    {
        let shape = if contains_unknown {
            ShapeFact::Unknown
        } else {
            super::validation::broadcast_shape(arguments).unwrap_or_else(|| value.shape.clone())
        };
        return ValueFact::proven(ValueKindFact::String, shape, StorageFact::Dense);
    }
    if contains_unknown {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    if let Some(value) = arguments
        .iter()
        .find(|value| matches!(value.kind, ValueKindFact::Cell(_)))
    {
        let shape =
            super::validation::broadcast_shape(arguments).unwrap_or_else(|| value.shape.clone());
        return ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(character_row()),
                elements: Vec::new(),
                elements_complete: false,
            }),
            shape,
            StorageFact::Dense,
        );
    }
    character_row()
}
