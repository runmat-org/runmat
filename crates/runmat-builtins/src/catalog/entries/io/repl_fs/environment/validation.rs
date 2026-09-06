use runmat_types::{standard, CellFact, ShapeFact, ValueFact, ValueKindFact};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NamePolicy {
    Matlab,
    GetenvExtensions,
}

pub(super) fn valid_name(input: &ValueFact, policy: NamePolicy) -> bool {
    match &input.kind {
        ValueKindFact::Character => {
            policy == NamePolicy::GetenvExtensions || !known_character_matrix(&input.shape)
        }
        ValueKindFact::String | ValueKindFact::Unknown => true,
        ValueKindFact::Cell(cell) => valid_name_cell(cell, policy),
        _ => false,
    }
}

pub(super) fn valid_name_or_dictionary(input: &ValueFact) -> bool {
    valid_name(input, NamePolicy::Matlab)
        || is_dictionary(input)
        || matches!(input.kind, ValueKindFact::Unknown)
}

pub(super) fn valid_value(input: &ValueFact) -> bool {
    match &input.kind {
        ValueKindFact::Character => !known_character_matrix(&input.shape),
        ValueKindFact::String | ValueKindFact::Unknown => true,
        ValueKindFact::Numeric(_) => {
            input.shape.element_count() != Some(0)
                && input.shape.element_count().is_none_or(|count| count == 1)
        }
        ValueKindFact::Cell(cell) => valid_value_cell(cell),
        _ => false,
    }
}

pub(super) fn shapes_are_compatible(name: &ValueFact, value: &ValueFact) -> bool {
    let name_shape = container_shape(name);
    let value_shape = container_shape(value);
    if name_shape.as_ref().and_then(ShapeFact::element_count) == Some(1)
        || value_shape.as_ref().and_then(ShapeFact::element_count) == Some(1)
    {
        return true;
    }
    match (
        name_shape.as_ref().and_then(known_shape),
        value_shape.as_ref().and_then(known_shape),
    ) {
        (Some(left), Some(right)) => left == right,
        _ => true,
    }
}

fn is_dictionary(input: &ValueFact) -> bool {
    matches!(
        &input.kind,
        ValueKindFact::Object(object)
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|class| class.is(standard::DICTIONARY))
    )
}

fn valid_name_cell(cell: &CellFact, policy: NamePolicy) -> bool {
    valid_name_cell_element(&cell.element, policy)
        && cell
            .elements
            .iter()
            .all(|element| valid_name_cell_element(element, policy))
}

fn valid_name_cell_element(input: &ValueFact, policy: NamePolicy) -> bool {
    match input.kind {
        ValueKindFact::Character => !known_character_matrix(&input.shape),
        ValueKindFact::String => {
            policy == NamePolicy::GetenvExtensions && input.shape == ShapeFact::Scalar
        }
        ValueKindFact::Unknown => true,
        _ => false,
    }
}

fn valid_value_cell(cell: &CellFact) -> bool {
    valid_value_cell_element(&cell.element) && cell.elements.iter().all(valid_value_cell_element)
}

fn valid_value_cell_element(input: &ValueFact) -> bool {
    match input.kind {
        ValueKindFact::Character => !known_character_matrix(&input.shape),
        ValueKindFact::String => input.shape == ShapeFact::Scalar,
        ValueKindFact::Numeric(_) => input.shape.element_count() == Some(1),
        ValueKindFact::Unknown => true,
        _ => false,
    }
}

fn known_character_matrix(shape: &ShapeFact) -> bool {
    matches!(
        shape.known_dims().and_then(|dimensions| dimensions.first().copied().flatten()),
        Some(rows) if rows > 1
    )
}

fn known_shape(shape: &ShapeFact) -> Option<Vec<usize>> {
    shape.known_dims()?.into_iter().collect()
}

fn container_shape(value: &ValueFact) -> Option<ShapeFact> {
    match value.kind {
        ValueKindFact::Character | ValueKindFact::Numeric(_) => Some(ShapeFact::Scalar),
        ValueKindFact::String | ValueKindFact::Cell(_) => Some(value.shape.clone()),
        ValueKindFact::Unknown => None,
        _ => Some(value.shape.clone()),
    }
}
