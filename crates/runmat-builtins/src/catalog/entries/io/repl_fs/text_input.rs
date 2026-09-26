use runmat_types::{ShapeFact, ValueFact, ValueKindFact};

pub(in crate::catalog::entries::io) fn scalar(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Character => character_row(&value.shape),
        ValueKindFact::String => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Unknown => true,
        _ => false,
    }
}

pub(in crate::catalog::entries::io) fn character_container(shape: &ShapeFact) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{DynamicReason, StorageFact};

    #[test]
    fn trailing_dimensions_and_unknown_kind_are_conservative() {
        for (dims, supported) in [
            (vec![Some(1), Some(3), Some(1)], true),
            (vec![Some(1), Some(3), Some(2)], false),
            (vec![Some(1), Some(3), None], true),
            (vec![Some(0), Some(3)], false),
        ] {
            assert_eq!(
                scalar(&ValueFact::proven(
                    ValueKindFact::Character,
                    ShapeFact::from(dims),
                    StorageFact::Dense
                )),
                supported
            );
        }
        let mut unknown = ValueFact::unknown(DynamicReason::Unspecified);
        unknown.shape = ShapeFact::from(vec![Some(1), Some(5)]);
        assert!(scalar(&unknown));
        unknown.kind = ValueKindFact::String;
        assert!(!scalar(&unknown));
        unknown.kind = ValueKindFact::Logical;
        assert!(!scalar(&unknown));
    }
}
