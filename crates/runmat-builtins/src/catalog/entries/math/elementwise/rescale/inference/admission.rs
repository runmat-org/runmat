use crate::catalog::inference::argument_error;
use runmat_types::{CallRequest, InferenceDiagnostic, ValueFact, ValueKindFact};

pub(super) struct AdmittedCall<'a> {
    pub source: Option<&'a ValueFact>,
    pub operands: Vec<(usize, &'a ValueFact)>,
    pub diagnostics: Vec<InferenceDiagnostic>,
    pub grammar_known: bool,
}

pub(super) fn inspect(request: &CallRequest) -> AdmittedCall<'_> {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len().is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-RESCALE-ARITY",
            "rescale requires A, optional lower and upper bounds, and complete name-value pairs",
            request.arguments.len().min(1),
        ));
    }
    let source = request.arguments.first();
    let Some(first_rest) = request.arguments.get(1) else {
        return AdmittedCall {
            source,
            operands: Vec::new(),
            diagnostics,
            grammar_known: true,
        };
    };

    let literal = request.literals.literal_string_at(1);
    let starts_options = literal.is_some()
        || matches!(
            first_rest.kind,
            ValueKindFact::String | ValueKindFact::Character
        );
    let grammar_known = literal.is_some() || !matches!(first_rest.kind, ValueKindFact::Unknown);
    let option_start = if starts_options { 1 } else { 3 };
    let mut operands = Vec::new();
    if !starts_options {
        if let Some(lower) = request.arguments.get(1) {
            operands.push((1, lower));
        }
        if let Some(upper) = request.arguments.get(2) {
            operands.push((2, upper));
        }
    }
    inspect_options(request, option_start, &mut operands, &mut diagnostics);
    AdmittedCall {
        source,
        operands,
        diagnostics,
        grammar_known,
    }
}

fn inspect_options<'a>(
    request: &'a CallRequest,
    start: usize,
    operands: &mut Vec<(usize, &'a ValueFact)>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let mut index = start;
    while index < request.arguments.len() {
        if index + 1 >= request.arguments.len() {
            break;
        }
        if let Some(name) = request.literals.literal_string_at(index) {
            if !matches!(name.as_str(), "inputmin" | "inputmax") {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RESCALE-OPTION",
                    "rescale option names must be InputMin or InputMax",
                    index,
                ));
            }
        } else if !matches!(
            request.arguments[index].kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-RESCALE-OPTION",
                "rescale name-value option names must be scalar text",
                index,
            ));
        }
        operands.push((index + 1, &request.arguments[index + 1]));
        index += 2;
    }
}
