use runmat_types::{NumericDomain, ShapeFact, StorageFact, ValueFact, ValueKindFact};

#[derive(Clone, Copy)]
pub(in crate::catalog::entries::io::repl_fs) enum Policy {
    PathReplacement,
    AddFolders,
    RemoveFolders,
}

pub(in crate::catalog::entries::io::repl_fs) fn supports(
    argument: &ValueFact,
    policy: Policy,
) -> bool {
    match &argument.kind {
        ValueKindFact::Character => match policy {
            Policy::PathReplacement => is_character_row(&argument.shape),
            Policy::AddFolders | Policy::RemoveFolders => is_character_container(&argument.shape),
        },
        ValueKindFact::String => {
            !matches!(policy, Policy::PathReplacement)
                || argument
                    .shape
                    .element_count()
                    .is_none_or(|count| count == 1)
        }
        ValueKindFact::Cell(cell) => {
            if matches!(policy, Policy::PathReplacement) {
                false
            } else if cell.elements_complete {
                cell.elements.iter().all(|value| supports(value, policy))
            } else {
                supports(&cell.element, policy)
            }
        }
        ValueKindFact::Numeric(numeric) => {
            !matches!(policy, Policy::RemoveFolders)
                && numeric.domain != NumericDomain::Complex
                && matches!(argument.storage, StorageFact::Dense | StorageFact::Unknown)
                && is_character_row(&argument.shape)
        }
        ValueKindFact::Unknown => true,
        _ => false,
    }
}

fn is_character_container(shape: &ShapeFact) -> bool {
    shape.known_dims().is_none_or(|dims| dims.len() <= 2)
}

fn is_character_row(shape: &ShapeFact) -> bool {
    shape
        .known_dims()
        .is_none_or(|dims| dims.len() <= 2 && dims.first().is_none_or(|rows| *rows == Some(1)))
}
