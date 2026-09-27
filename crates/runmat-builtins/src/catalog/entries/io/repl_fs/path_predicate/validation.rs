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
    shape.known_dims().is_some_and(|dims| {
        dims.first()
            .is_some_and(|rows| rows.is_some_and(|n| n != 1))
            || dims.iter().skip(2).any(|n| n.is_some_and(|n| n != 1))
    })
}
