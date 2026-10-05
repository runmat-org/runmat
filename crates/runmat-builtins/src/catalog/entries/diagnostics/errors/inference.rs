use crate::catalog::inference::{argument_error, finish_fixed_outputs, known_invalid_text_scalar};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, InferenceDiagnostic, ValueKindFact};

pub(super) fn error(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(InferenceDiagnostic::error(
            "RM-CATALOG-ERROR-ARITY",
            "error requires a message or error payload",
        ));
    } else if matches!(
        request.arguments[0].kind,
        ValueKindFact::Callable(_) | ValueKindFact::Foreign(_) | ValueKindFact::ClassReference(_)
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ERROR-PAYLOAD",
            "error expects text, an MException, or a message struct",
            0,
        ));
    }
    finish_fixed_outputs(entry, request, Vec::new(), diagnostics)
}

pub(super) fn rethrow(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(InferenceDiagnostic::error(
            "RM-CATALOG-RETHROW-ARITY",
            "rethrow accepts one exception or the active catch exception",
        ));
    } else if let Some(arg) = request.arguments.first() {
        if !matches!(
            arg.kind,
            ValueKindFact::Exception(_) | ValueKindFact::String | ValueKindFact::Unknown
        ) || (matches!(arg.kind, ValueKindFact::String) && known_invalid_text_scalar(arg))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-RETHROW-PAYLOAD",
                "rethrow expects an MException or string scalar",
                0,
            ));
        }
    }
    finish_fixed_outputs(entry, request, Vec::new(), diagnostics)
}
