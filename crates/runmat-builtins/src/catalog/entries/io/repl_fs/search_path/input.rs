use runmat_types::{NumericDomain, ShapeFact, StorageFact, ValueFact, ValueKindFact};

#[derive(Clone, Copy)]
pub(in crate::catalog::entries::io::repl_fs) enum Policy {
    PathReplacement,
    AddFolders,
    RemoveFolders,
    Generate,
}

pub(in crate::catalog::entries::io::repl_fs) fn supports(
    argument: &ValueFact,
    policy: Policy,
) -> bool {
    match &argument.kind {
        ValueKindFact::Character => match policy {
            Policy::PathReplacement | Policy::Generate => is_character_row(&argument.shape),
            Policy::AddFolders | Policy::RemoveFolders => is_character_container(&argument.shape),
        },
        ValueKindFact::String => {
            !matches!(policy, Policy::PathReplacement | Policy::Generate)
                || argument
                    .shape
                    .element_count()
                    .is_none_or(|count| count == 1)
        }
        ValueKindFact::Cell(cell) => {
            if matches!(policy, Policy::PathReplacement | Policy::Generate) {
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
                && is_numeric_row(&argument.shape, policy)
        }
        ValueKindFact::Unknown => true,
        _ => false,
    }
}

fn is_character_container(shape: &ShapeFact) -> bool {
    shape.known_dims().is_none_or(|dims| {
        dims.iter()
            .skip(2)
            .all(|dim| dim.is_none_or(|size| size == 1))
    })
}

fn is_character_row(shape: &ShapeFact) -> bool {
    shape.known_dims().is_none_or(|dims| {
        is_character_container(shape)
            && dims
                .first()
                .is_none_or(|rows| rows.is_none_or(|rows| rows == 1))
    })
}

fn is_numeric_row(shape: &ShapeFact, policy: Policy) -> bool {
    is_character_container(shape)
        && shape.known_dims().is_none_or(|dims| {
            dims.first().is_none_or(|rows| {
                rows.is_none_or(|rows| {
                    rows == 1
                        || (rows == 0
                            && matches!(policy, Policy::PathReplacement | Policy::AddFolders))
                })
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{CallRequest, LiteralContext, OutputSelection, RequestedOutputCount};

    #[test]
    fn path_consumers_preserve_unknown_extents_and_effective_rank() {
        for name in ["path", "genpath", "savepath", "rmpath", "addpath"] {
            let entry = crate::builtin_catalog_entry_by_name(name).unwrap();
            for (shape, rejected) in [
                (vec![Some(1), Some(4), Some(1)], false),
                (vec![None, Some(4), None], false),
                (vec![Some(1), Some(1), Some(4)], true),
            ] {
                let request = CallRequest {
                    arguments: vec![ValueFact::proven(
                        ValueKindFact::Character,
                        ShapeFact::from(shape),
                        StorageFact::Dense,
                    )],
                    literals: LiteralContext::default(),
                    outputs: OutputSelection::new(RequestedOutputCount::One),
                };
                let result = crate::infer_catalog_call(entry, &request);
                assert_eq!(!result.diagnostics.is_empty(), rejected, "{name}");
            }
        }
    }

    #[test]
    fn empty_numeric_acceptance_is_policy_specific() {
        let shape = ShapeFact::from(vec![Some(0), Some(0)]);
        assert!(is_numeric_row(&shape, Policy::PathReplacement));
        assert!(is_numeric_row(&shape, Policy::AddFolders));
        assert!(!is_numeric_row(&shape, Policy::Generate));
    }
}
