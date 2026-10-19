use runmat_types::{NumericDomain, ShapeFact, StorageFact, ValueFact, ValueKindFact};

pub(super) fn is_text_container(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Character => is_character_row(value),
        ValueKindFact::String | ValueKindFact::Unknown => true,
        ValueKindFact::Cell(cell) => cell
            .elements
            .iter()
            .chain(std::iter::once(cell.element.as_ref()))
            .all(|element| match element.kind {
                ValueKindFact::Character => is_character_row(element),
                ValueKindFact::Unknown => true,
                _ => false,
            }),
        _ => false,
    }
}

fn is_character_row(value: &ValueFact) -> bool {
    value.shape.known_dims().is_none_or(|dimensions| {
        dimensions.iter().skip(2).all(|n| n.is_none_or(|n| n == 1))
            && dimensions
                .first()
                .is_none_or(|rows| rows.is_none_or(|rows| rows == 1))
    })
}

pub(super) fn is_numeric_character_row(value: &ValueFact) -> bool {
    let ValueKindFact::Numeric(numeric) = value.kind else {
        return false;
    };
    numeric.domain == NumericDomain::Real
        && matches!(value.storage, StorageFact::Dense | StorageFact::Unknown)
        && value.shape.known_dims().is_none_or(|dimensions| {
            dimensions.iter().skip(2).all(|n| n.is_none_or(|n| n == 1))
                && dimensions
                    .first()
                    .is_none_or(|rows| rows.is_none_or(|rows| rows == 1))
        })
}

pub(super) fn broadcast_shape(arguments: &[ValueFact]) -> Option<ShapeFact> {
    if arguments.iter().any(|argument| {
        matches!(
            argument.kind,
            ValueKindFact::String | ValueKindFact::Cell(_)
        ) && argument.shape.element_count().is_none()
    }) {
        return Some(ShapeFact::Unknown);
    }
    let mut selected: Option<ShapeFact> = None;
    for argument in arguments {
        if !matches!(
            argument.kind,
            ValueKindFact::String | ValueKindFact::Cell(_)
        ) {
            continue;
        }
        if argument.shape.element_count() == Some(1) {
            continue;
        }
        match &selected {
            None => selected = Some(argument.shape.clone()),
            Some(existing) if known_mismatch(existing, &argument.shape) => return None,
            Some(_) => {}
        }
    }
    Some(selected.unwrap_or(ShapeFact::Scalar))
}

pub(super) fn has_known_shape_mismatch(arguments: &[ValueFact]) -> bool {
    let non_scalars: Vec<&ShapeFact> = arguments
        .iter()
        .filter(|argument| {
            matches!(
                argument.kind,
                ValueKindFact::String | ValueKindFact::Cell(_)
            )
        })
        .filter(|argument| argument.shape.element_count() != Some(1))
        .map(|argument| &argument.shape)
        .collect();
    non_scalars.iter().enumerate().any(|(index, left)| {
        non_scalars
            .iter()
            .skip(index + 1)
            .any(|right| known_mismatch(left, right))
    })
}

fn known_mismatch(left: &ShapeFact, right: &ShapeFact) -> bool {
    if left.element_count().is_none() || right.element_count().is_none() {
        return false;
    }
    match (left.known_dims(), right.known_dims()) {
        (Some(left), Some(right)) => left != right,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_dimensions_do_not_prove_a_shape_error_or_a_broadcast_shape() {
        let unknown = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        );
        let known = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Dense,
        );
        assert!(!has_known_shape_mismatch(&[unknown.clone(), known.clone()]));
        assert_eq!(broadcast_shape(&[unknown, known]), Some(ShapeFact::Unknown));
        let row = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![None, Some(4)]),
            StorageFact::Dense,
        );
        assert!(is_text_container(&row));
        let nd = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(2), Some(2)]),
            StorageFact::Dense,
        );
        assert!(!is_text_container(&nd));
    }
}
