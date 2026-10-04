use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

use super::JsonEncodeOptionName;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if count == 0 || (count > 2 && count.is_multiple_of(2)) {
        diagnostics.push(argument_error(
            "RM-CATALOG-JSONENCODE-ARITY",
            "jsonencode requires a value, optionally followed by a struct or complete name-value pairs",
            0,
        ));
    }
    if count == 2
        && (!matches!(
            request.arguments[1].kind,
            ValueKindFact::Struct(_) | ValueKindFact::Unknown
        ) || (matches!(request.arguments[1].kind, ValueKindFact::Struct(_))
            && request.arguments[1].shape.is_definitely_nonscalar()))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-JSONENCODE-OPTIONS",
            "a single options argument must be a scalar struct",
            1,
        ));
    }
    for index in (1..count).step_by(2) {
        if count == 2 {
            break;
        }
        let name = &request.arguments[index];
        if definitely_invalid_option_name(name) {
            diagnostics.push(argument_error(
                "RM-CATALOG-JSONENCODE-OPTION-NAME",
                "option names must be scalar text",
                index,
            ));
        }
        if request
            .literals
            .literal_string_at(index)
            .is_some_and(|name| JsonEncodeOptionName::from_public_name(&name).is_none())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-JSONENCODE-UNKNOWN-OPTION",
                "only PrettyPrint and ConvertInfAndNaN are supported",
                index,
            ));
        }
        if let Some(value) = request.arguments.get(index + 1) {
            if definitely_invalid_option_value(value) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-JSONENCODE-OPTION-VALUE",
                    "option values must be scalar logical, numeric, or boolean-like text",
                    index + 1,
                ));
            }
        }
    }
    let output = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    finish_fixed(entry, request, output, diagnostics)
}

fn definitely_invalid_option_name(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .is_some_and(|rows| rows.is_some_and(|rows| rows != 1))
        }),
        _ => true,
    }
}

fn definitely_invalid_option_value(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::String => {
            value.shape.is_definitely_nonscalar()
        }
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .is_some_and(|rows| rows.is_some_and(|rows| rows != 1))
        }),
        _ => true,
    }
}
