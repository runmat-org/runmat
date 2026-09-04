use runmat_types::{DimensionFact, ShapeFact, ValueFact, ValueKindFact};

pub(super) fn known_expanded_count(input: &ValueFact) -> Option<usize> {
    if !splits_matrix_columns(&input.kind) {
        return Some(1);
    }
    let ShapeFact::Shaped { dims } = &input.shape else {
        return None;
    };
    let rows = known_dimension(dims.first())?;
    let columns = known_dimension(dims.get(1))?;
    Some(if rows > 1 && columns > 1 { columns } else { 1 })
}

pub(super) fn is_known_matrix(input: &ValueFact) -> bool {
    known_expanded_count(input).is_some_and(|count| count > 1)
}

fn splits_matrix_columns(kind: &ValueKindFact) -> bool {
    matches!(
        kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::String
    )
}

fn known_dimension(dimension: Option<&DimensionFact>) -> Option<usize> {
    match dimension {
        Some(DimensionFact::Known(value)) => Some(*value),
        Some(DimensionFact::Symbolic(_) | DimensionFact::Unknown) | None => None,
    }
}
