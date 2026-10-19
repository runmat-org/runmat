use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

fn predicate(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.len() == 1 {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-TABLE-PREDICATE-ARITY",
            "expected exactly one input",
            0,
        )]
    };
    // Exact runtime class identity determines the boolean value, but the type
    // contract needs only its scalar logical shape, independent of input data.
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Logical),
        diagnostics,
    )
}

pub(super) fn iscategorical(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    predicate(request, entry)
}
pub(super) fn isordinal(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    predicate(request, entry)
}
pub(super) fn istable(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    predicate(request, entry)
}
pub(super) fn istimetable(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    predicate(request, entry)
}
