use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, ValueKindFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(runmat_types::InferenceDiagnostic::error(
            "RM-CATALOG-ASSERT-ARITY",
            "assert requires a condition",
        ));
    } else if matches!(
        request.arguments[0].kind,
        ValueKindFact::Cell(_)
            | ValueKindFact::Struct(_)
            | ValueKindFact::Object(_)
            | ValueKindFact::Exception(_)
            | ValueKindFact::Callable(_)
            | ValueKindFact::Foreign(_)
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ASSERT-CONDITION",
            "assert condition must be convertible to logical",
            0,
        ));
    }
    finish_fixed_outputs(entry, request, Vec::new(), diagnostics)
}
