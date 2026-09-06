use runmat_types::{ShapeFact, ValueFact, ValueKindFact};

pub(super) fn scalar_text(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Character => character_row(&value.shape),
        ValueKindFact::String | ValueKindFact::Unknown => {
            value.shape.element_count().is_none_or(|count| count == 1)
        }
        _ => false,
    }
}

fn character_row(shape: &ShapeFact) -> bool {
    !matches!(
        shape.known_dims().and_then(|dims| dims.first().copied().flatten()),
        Some(rows) if rows > 1
    )
}
