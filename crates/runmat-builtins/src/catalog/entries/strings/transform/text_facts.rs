//! Shared host text geometry, not identity selection or overload dispatch.
use crate::catalog::inference::possible_character_row;
use runmat_types::*;

#[derive(Clone, Copy)]
pub(in crate::catalog::entries::strings) enum CellTextPolicy {
    ScalarElements,
    ArrayElements,
    Recursive,
}

pub(in crate::catalog::entries::strings) fn invalid_text(
    value: &ValueFact,
    cells: CellTextPolicy,
) -> bool {
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::String => false,
        ValueKindFact::Character => nonmatrix(&value.shape),
        ValueKindFact::Cell(cell) => {
            value.shape.element_count() != Some(0)
                && cell
                    .elements
                    .iter()
                    .chain(std::iter::once(cell.element.as_ref()))
                    .any(|element| match cells {
                        CellTextPolicy::Recursive => invalid_text(element, cells),
                        CellTextPolicy::ArrayElements => match element.kind {
                            ValueKindFact::Unknown | ValueKindFact::String => false,
                            ValueKindFact::Character => nonmatrix(&element.shape),
                            _ => true,
                        },
                        CellTextPolicy::ScalarElements => match element.kind {
                            ValueKindFact::Unknown => false,
                            ValueKindFact::String => {
                                element.shape.element_count().is_some_and(|n| n != 1)
                            }
                            ValueKindFact::Character => !possible_character_row(&element.shape),
                            _ => true,
                        },
                    })
        }
        _ => true,
    }
}

pub(in crate::catalog::entries::strings) fn pattern(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::String => value.shape.element_count().is_none_or(|n| n == 1),
        ValueKindFact::Character => possible_character_row(&value.shape),
        ValueKindFact::Object(object) => object
            .runtime_class
            .as_ref()
            .is_none_or(|class| class == &standard::PATTERN.owned()),
        _ => false,
    }
}

pub(in crate::catalog::entries::strings) fn text_shape(value: &ValueFact) -> ShapeFact {
    match value.kind {
        ValueKindFact::String | ValueKindFact::Cell(_) => value.shape.clone(),
        ValueKindFact::Character => ShapeFact::from(vec![
            value
                .shape
                .known_dims()
                .and_then(|dims| dims.first().copied().flatten()),
            Some(1),
        ]),
        _ => ShapeFact::Unknown,
    }
}

pub(in crate::catalog::entries::strings) fn host(
    kind: ValueKindFact,
    shape: ShapeFact,
) -> ValueFact {
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    ValueFact::proven(kind, shape, storage)
}

pub(in crate::catalog::entries::strings) fn preserved(
    value: &ValueFact,
    changes_width: bool,
) -> ValueFact {
    match &value.kind {
        ValueKindFact::String => host(ValueKindFact::String, value.shape.clone()),
        ValueKindFact::Character => {
            let shape = if changes_width {
                ShapeFact::from(vec![
                    value
                        .shape
                        .known_dims()
                        .and_then(|dims| dims.first().copied().flatten()),
                    None,
                ])
            } else {
                value.shape.clone()
            };
            ValueFact::proven(ValueKindFact::Character, shape, StorageFact::Dense)
        }
        ValueKindFact::Cell(_) => host(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                elements: Vec::new(),
                elements_complete: false,
            }),
            value.shape.clone(),
        ),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

pub(in crate::catalog::entries::strings) fn real_control(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    )
}

pub(in crate::catalog::entries::strings) fn nonmatrix(shape: &ShapeFact) -> bool {
    shape
        .known_dims()
        .is_some_and(|dims| dims.iter().skip(2).any(|n| n.is_some_and(|n| n != 1)))
}
