use std::collections::BTreeMap;

use runmat_types::{
    standard, CellFact, DynamicReason, NumericClass, NumericDomain, NumericFact, ObjectFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn double_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

pub(super) fn character_row() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}

pub(super) fn dictionary() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::DICTIONARY.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

pub(super) fn text_for_names(input: &ValueFact) -> ValueFact {
    match &input.kind {
        ValueKindFact::Character => character_for_character_names(&input.shape),
        ValueKindFact::String if input.shape == ShapeFact::Scalar => character_row(),
        ValueKindFact::String if matches!(input.shape, ShapeFact::Shaped { .. }) => {
            ValueFact::proven(
                ValueKindFact::String,
                input.shape.clone(),
                StorageFact::Dense,
            )
        }
        ValueKindFact::Cell(cell) => ValueFact::proven(
            ValueKindFact::Cell(text_cell(cell)),
            input.shape.clone(),
            StorageFact::Dense,
        ),
        _ => ValueFact::unknown(DynamicReason::DynamicDispatch),
    }
}

pub(super) fn logical_for_names(input: Option<&ValueFact>) -> ValueFact {
    let shape = input.map_or(ShapeFact::Scalar, |input| match input.kind {
        ValueKindFact::String | ValueKindFact::Cell(_) => input.shape.clone(),
        _ => ShapeFact::Scalar,
    });
    ValueFact::proven(ValueKindFact::Logical, shape, StorageFact::Dense)
}

fn character_for_character_names(shape: &ShapeFact) -> ValueFact {
    let rows = shape
        .known_dims()
        .and_then(|dimensions| dimensions.first().copied().flatten());
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![rows, None]),
        StorageFact::Dense,
    )
}

fn text_cell(cell: &CellFact) -> CellFact {
    CellFact {
        element: Box::new(text_cell_element(&cell.element)),
        elements: cell.elements.iter().map(text_cell_element).collect(),
        elements_complete: cell.elements_complete,
    }
}

fn text_cell_element(input: &ValueFact) -> ValueFact {
    match input.kind {
        ValueKindFact::Character => character_row(),
        ValueKindFact::String if input.shape == ShapeFact::Scalar => {
            ValueFact::scalar(ValueKindFact::String)
        }
        _ => ValueFact::unknown(DynamicReason::DynamicDispatch),
    }
}
