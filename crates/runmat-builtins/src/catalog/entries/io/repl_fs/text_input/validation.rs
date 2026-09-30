use runmat_types::{ShapeFact, ValueFact, ValueKindFact};

pub(super) fn scalar(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Character => character_row(&value.shape),
        ValueKindFact::String => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Unknown => true,
        _ => false,
    }
}

pub(super) fn character_container(shape: &ShapeFact) -> bool {
    shape.known_dims().is_none_or(|dims| {
        dims.iter()
            .skip(2)
            .all(|extent| extent.is_none_or(|n| n == 1))
    })
}

fn character_row(shape: &ShapeFact) -> bool {
    character_container(shape)
        && !matches!(
            shape.known_dims().and_then(|dims| dims.first().copied().flatten()),
            Some(rows) if rows != 1
        )
}
