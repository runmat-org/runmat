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
            ValueKindFact::Character => folder
                .shape
                .known_dims()
                .is_none_or(|dims| dims.first() == Some(&Some(1))),
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
