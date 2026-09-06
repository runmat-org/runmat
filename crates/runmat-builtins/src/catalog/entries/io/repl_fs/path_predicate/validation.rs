use runmat_types::{CellFact, ShapeFact, ValueFact, ValueKindFact};

pub(super) fn valid_paths(input: &ValueFact) -> bool {
    match &input.kind {
        ValueKindFact::Character => !known_character_matrix(&input.shape),
        ValueKindFact::String | ValueKindFact::Unknown => true,
        ValueKindFact::Cell(cell) => valid_cell(cell),
        _ => false,
    }
}

fn valid_cell(cell: &CellFact) -> bool {
    valid_cell_element(&cell.element) && cell.elements.iter().all(valid_cell_element)
}

fn valid_cell_element(input: &ValueFact) -> bool {
    matches!(input.kind, ValueKindFact::Unknown)
        || matches!(input.kind, ValueKindFact::Character) && !known_character_matrix(&input.shape)
}

fn known_character_matrix(shape: &ShapeFact) -> bool {
    matches!(
        shape.known_dims().and_then(|dims| dims.first().copied().flatten()),
        Some(rows) if rows > 1
    )
}
