use crate::catalog::inference::{argument_error, finish_fixed_outputs, known_invalid_text_scalar};
use crate::diagnostic_identifier::{is_qualified, is_unqualified_field};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, LiteralValue, ValueFact, ValueKindFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(runmat_types::InferenceDiagnostic::error(
            "RM-CATALOG-ASSERT-ARITY",
            "assert requires a condition",
        ));
    } else if matches!(
        request.arguments[0].kind,
        ValueKindFact::String
            | ValueKindFact::Cell(_)
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
    // Failure text and replacements are evaluated only after a false condition.
    // Do not reject a passing or dynamically determined assertion for dead payloads.
    if matches!(
        request.literals.literal_args.first(),
        Some(LiteralValue::Bool(false))
    ) {
        infer_failure_payload(request, &mut diagnostics);
    }
    finish_fixed_outputs(entry, request, Vec::new(), diagnostics)
}

fn infer_failure_payload(
    request: &CallRequest,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    if request.arguments.len() < 2 {
        return;
    }
    let candidate = &request.arguments[1];
    if invalid_text(candidate) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ASSERT-MESSAGE",
            "assert failure text must be a string scalar or character row",
            1,
        ));
        return;
    }
    let literal = request.literals.literal_args.get(1);
    let qualified = literal
        .and_then(literal_text)
        .is_some_and(|text| text.contains(':'))
        && request.arguments.len() >= 3;
    let replacement_start = if qualified {
        if let Some(text) = literal.and_then(literal_text) {
            if !is_qualified(text) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ASSERT-IDENTIFIER",
                    "assert error identifier must contain valid colon-separated fields",
                    1,
                ));
            }
        }
        if invalid_text(&request.arguments[2]) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ASSERT-MESSAGE",
                "assert message template must be a string scalar or character row",
                2,
            ));
        }
        3
    } else {
        // An unqualified source-known word may be an identifier in RunMat
        // mode, so its following argument has mode-dependent meaning.
        if literal
            .and_then(literal_text)
            .is_some_and(is_unqualified_field)
            && request.arguments.len() >= 3
        {
            return;
        }
        2
    };
    for (index, value) in request.arguments.iter().enumerate().skip(replacement_start) {
        if invalid_replacement(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ASSERT-REPLACEMENT",
                "assert formatting replacements must be numeric scalars, string scalars, or character rows",
                index,
            ));
        }
    }
}

fn literal_text(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::String(text) | LiteralValue::Character(text) => Some(text),
        _ => None,
    }
}

fn invalid_text(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::String | ValueKindFact::Character => known_invalid_text_scalar(value),
        _ => true,
    }
}

fn invalid_replacement(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical => value.shape.is_definitely_nonscalar(),
        ValueKindFact::String | ValueKindFact::Character => known_invalid_text_scalar(value),
        // Cells and other aggregate values are flattened recursively by the
        // formatter; their type alone does not prove an invalid replacement.
        _ => false,
    }
}
