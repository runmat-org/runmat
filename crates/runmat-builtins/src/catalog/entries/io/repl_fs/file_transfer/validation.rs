use runmat_types::{ShapeFact, ValueFact, ValueKindFact};

pub(super) fn is_text_scalar(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Character => character_is_row(&value.shape),
        ValueKindFact::String => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Unknown => true,
        _ => false,
    }
}

fn character_is_row(shape: &ShapeFact) -> bool {
    shape
        .known_dims()
        .is_none_or(|dimensions| dimensions.first() == Some(&Some(1)))
}
