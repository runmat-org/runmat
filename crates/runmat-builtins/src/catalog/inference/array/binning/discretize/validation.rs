use super::super::super::super::argument_error;
use super::super::super::super::literal_text;
use super::known_nonvector;
use runmat_types::{CallRequest, InferenceDiagnostic, NumericDomain, NumericFact, ValueKindFact};

pub(super) fn validate_tail(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(first) = request.arguments.get(2) else {
        return;
    };
    let text = request.literals.literal_args.get(2).and_then(literal_text);
    let option = text
        .as_deref()
        .is_some_and(|text| text.eq_ignore_ascii_case("IncludedEdge"));
    let ambiguous = text.is_none()
        && matches!(
            first.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        );
    if ambiguous {
        return;
    }
    let start = if option { 2 } else { 3 };
    if !option {
        if known_nonvector(&first.shape) {
            diagnostics.push(argument_error(
                "RM-CATALOG-DISCRETIZE-VALUES",
                "replacement values must be a vector",
                2,
            ));
        }
        if !matches!(
            first.kind,
            ValueKindFact::Unknown
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-DISCRETIZE-VALUES",
                "replacement values must be real numeric or text",
                2,
            ));
        }
    }
    if !(request.arguments.len() - start).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-DISCRETIZE-OPTIONS",
            "name-value options must occur in pairs",
            start,
        ));
        return;
    }
    for index in (start..request.arguments.len()).step_by(2) {
        for (position, accepted) in [
            (index, &["IncludedEdge"][..]),
            (index + 1, &["left", "right"][..]),
        ] {
            if request
                .literals
                .literal_args
                .get(position)
                .and_then(literal_text)
                .is_some_and(|text| {
                    !accepted
                        .iter()
                        .any(|candidate| text.eq_ignore_ascii_case(candidate))
                })
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-DISCRETIZE-OPTIONS",
                    "unsupported bin edge option",
                    position,
                ));
            }
        }
    }
}
