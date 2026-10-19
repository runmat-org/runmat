use super::super::response_mode::{literal_response_mode, HttpResponseDialect, HttpResponseMode};
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::parse_http_url;
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
    if let Some(url) = request.literals.literal_string_at(0) {
        if !url.trim().is_empty() && parse_http_url(&url).is_err() {
            diagnostics.push(argument_error(
                "RM-CATALOG-WEBREAD-URL",
                "URL must be an absolute HTTP or HTTPS URL",
                0,
            ));
        }
    }
    let mut output = match literal_response_mode(request, 1, HttpResponseDialect::WebRead) {
        Some(HttpResponseMode::Text) => ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
        Some(HttpResponseMode::Binary) => ValueFact::proven(
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

fn possible_text(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Unknown)
        || super::super::super::repl_fs::text_input::scalar(value)
}
