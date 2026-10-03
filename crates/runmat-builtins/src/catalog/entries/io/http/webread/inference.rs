use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-WEBREAD-ARITY",
            "webread requires a URL",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|url| !possible_text(url))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-WEBREAD-URL",
            "URL must be a character row or string scalar",
            0,
        ));
    }
    if request
        .literals
        .literal_string_at(0)
        .is_some_and(|url| url.trim().is_empty())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-WEBREAD-URL",
            "URL must not be empty",
            0,
        ));
    }
    let mut output = match response_mode(request, 1) {
        Some("text") => ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
        Some("binary") => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt8,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
        _ => ValueFact::unknown(DynamicReason::ExternalData),
    };
    output.residency = ResidencyFact::Host;
    finish_fixed(entry, request, output, diagnostics)
}

fn response_mode(request: &CallRequest, first_rest: usize) -> Option<&'static str> {
    let first = request.arguments.get(first_rest)?;
    let mut index = if matches!(
        first.kind,
        ValueKindFact::Struct(_) | ValueKindFact::Cell(_)
    ) {
        first_rest + 1
    } else if matches!(first.kind, ValueKindFact::Unknown)
        && request.literals.literal_string_at(first_rest).is_none()
    {
        return None;
    } else {
        first_rest
    };
    let mut mode = None;
    while index + 1 < request.arguments.len() {
        match request.literals.literal_string_at(index) {
            Some(name) if name.eq_ignore_ascii_case("contenttype") => {
                mode = request
                    .literals
                    .literal_string_at(index + 1)
                    .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
                        "text" | "char" | "string" => Some("text"),
                        "binary" | "octet-stream" | "raw" => Some("binary"),
                        _ => None,
                    });
            }
            None => mode = None,
            _ => {}
        }
        index += 2;
    }
    mode
}

fn possible_text(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Unknown)
        || super::super::super::repl_fs::text_input::scalar(value)
}
