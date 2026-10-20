//! Structural facts for scalar text controls shared by Text Analytics entries.

use runmat_types::{DimensionFact, ShapeFact, ValueFact, ValueKindFact};

pub(super) fn known_invalid_scalar_text_shape(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        // A character row is one scalar text control, regardless of width.
        // The Runtime text decoder also permits an empty, zero-row char array.
        ValueKindFact::Character => matches!(
            &value.shape,
            ShapeFact::Shaped { dims }
                if matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows > 1)
        ),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_multirow_char_is_not_a_scalar_text_control() {
        let multirow = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(2)]),
            runmat_types::StorageFact::Dense,
        );
        let row = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(4)]),
            runmat_types::StorageFact::Dense,
        );
        assert!(known_invalid_scalar_text_shape(&multirow));
        assert!(!known_invalid_scalar_text_shape(&row));
    }
}
