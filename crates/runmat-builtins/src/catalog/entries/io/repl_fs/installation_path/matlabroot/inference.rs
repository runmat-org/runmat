use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, ShapeFact, StorageFact, ValueFact, ValueKindFact};

pub(in crate::catalog::entries::io::repl_fs::installation_path) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let diagnostics = (!request.arguments.is_empty())
        .then(|| {
            argument_error(
                "RM-CATALOG-MATLABROOT-ARITY",
                "matlabroot accepts no inputs",
                0,
            )
        })
        .into_iter()
        .collect();
    finish_fixed(entry, request, character_row(), diagnostics)
}

fn character_row() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}
