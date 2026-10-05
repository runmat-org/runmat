use crate::catalog::inference::{argument_error, finish_fixed_outputs, known_invalid_text_scalar};
use crate::diagnostic_identifier::is_qualified;
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, InferenceDiagnostic, LiteralValue, ValueKindFact};

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
    if let Some(first) = request.arguments.first() {
        if matches!(
            first.kind,
            ValueKindFact::Exception(_) | ValueKindFact::Struct(_)
        ) && request.arguments.len() > 1
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ERROR-PAYLOAD-ARITY",
                "error exception and message-struct forms accept no additional arguments",
                1,
            ));
        }
        if matches!(first.kind, ValueKindFact::String | ValueKindFact::Character)
            && first.shape.element_count().is_some_and(|count| count > 1)
            && known_invalid_text_scalar(first)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ERROR-MESSAGE",
                "error message or identifier must be a string scalar or character row",
                0,
            ));
        }
        if request.arguments.len() >= 2
            && request
                .literals
                .literal_args
                .first()
                .is_some_and(|literal| match literal {
                    LiteralValue::String(text) | LiteralValue::Character(text) => {
                        is_qualified(text)
                    }
                    _ => false,
                })
            && known_invalid_text_scalar(&request.arguments[1])
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ERROR-MESSAGE",
                "error message template must be a string scalar or character row",
                1,
            ));
        }
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
