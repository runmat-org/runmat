use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-JSONDECODE-ARITY",
            "jsondecode requires exactly one text input",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(definitely_invalid_text)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-JSONDECODE-TEXT",
            "JSON input must be scalar text",
            0,
        ));
    }
    let output = if let Some(text) = request.literals.literal_string_at(0) {
        match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(serde_json::Value::Bool(_)) => ValueFact::scalar(ValueKindFact::Logical),
            Ok(serde_json::Value::Number(_)) => {
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }))
            }
            Ok(serde_json::Value::String(value)) => ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(1), Some(value.chars().count())]),
                StorageFact::Dense,
            ),
            Ok(serde_json::Value::Object(_)) => ValueFact::scalar(ValueKindFact::Struct(
                StructFact::scalar(std::collections::BTreeMap::new(), false),
            )),
            Ok(serde_json::Value::Null) => ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(0), Some(0)]),
                StorageFact::Dense,
            ),
            Ok(_) => ValueFact::unknown(DynamicReason::DynamicDispatch),
            Err(_) => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-JSONDECODE-PARSE",
                    "literal input is not valid JSON",
                    0,
                ));
                ValueFact::unknown(DynamicReason::DynamicDispatch)
            }
        }
    } else {
        ValueFact::unknown(DynamicReason::DynamicDispatch)
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn definitely_invalid_text(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .is_some_and(|rows| rows.is_some_and(|rows| rows > 1))
        }),
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        _ => true,
    }
}
