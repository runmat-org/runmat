use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, ShapeFact, StorageFact, ValueFact, ValueKindFact};

pub(in crate::catalog::entries::io::repl_fs::temporary_path) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-TEMPNAME-ARITY",
            "tempname accepts at most one input",
            1,
        ));
    }
    if let Some(folder) = request.arguments.first() {
        let valid = match folder.kind {
            ValueKindFact::Character => folder.shape.known_dims().is_none_or(|dims| {
                dims.iter().skip(2).all(|n| n.is_none_or(|n| n == 1))
                    && dims
                        .first()
                        .is_none_or(|rows| rows.is_none_or(|rows| rows == 1))
            }),
            ValueKindFact::String => folder.shape.element_count().is_none_or(|count| count == 1),
            ValueKindFact::Unknown => true,
            _ => false,
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-TEMPNAME-FOLDER",
                "tempname expects a character row or string scalar folder",
                0,
            ));
        }
    }
    finish_fixed(entry, request, character_row(), diagnostics)
}

fn character_row() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        CapabilityRequirement, LiteralContext, OutputSelection, RequestedOutputCount,
    };

    #[test]
    fn unknown_row_count_remains_possible_but_known_nd_folder_is_rejected() {
        for (shape, rejected) in [
            (ShapeFact::from(vec![None, Some(3)]), false),
            (ShapeFact::from(vec![Some(1), Some(2), Some(2)]), true),
            (ShapeFact::from(vec![Some(1), Some(2), Some(1)]), false),
            (ShapeFact::from(vec![Some(1), Some(2), None]), false),
        ] {
            let request = CallRequest {
                arguments: vec![ValueFact::proven(
                    ValueKindFact::Character,
                    shape,
                    StorageFact::Dense,
                )],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            };
            let result = infer(&request, &super::super::TEMPNAME_CATALOG_ENTRY);
            assert_eq!(
                super::super::TEMPNAME_CATALOG_ENTRY
                    .contract
                    .capabilities,
                &[CapabilityRequirement::Filesystem]
            );
            assert!(result
                .capabilities
                .0
                .contains(&CapabilityRequirement::Filesystem));
            assert_eq!(!result.diagnostics.is_empty(), rejected);
        }
    }
}
