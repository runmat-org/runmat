use crate::catalog::inference::argument_error;
use runmat_types::{CallRequest, InferenceDiagnostic, ValueKindFact};

pub(super) fn validate(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRUCTFUN-ARITY",
            "structfun requires a callable and scalar structure",
            request.arguments.len().min(1),
        ));
        return;
    }
    if !matches!(
        request.arguments[0].kind,
        ValueKindFact::Callable(_)
            | ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRUCTFUN-CALLABLE",
            "structfun requires a function handle or function name",
            0,
        ));
    }
    if !matches!(
        request.arguments[1].kind,
        ValueKindFact::Struct(_) | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRUCTFUN-STRUCT",
            "structfun requires a scalar structure",
            1,
        ));
    }
}
