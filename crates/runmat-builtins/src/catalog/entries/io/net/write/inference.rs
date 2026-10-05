use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

use super::super::NetDataType;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::fixed(vec![ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);

    if !(2..=3).contains(&request.arguments.len()) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-NET-WRITE-ARITY",
            "write requires a client, payload, and optional datatype",
            0,
        ));
    }
    if request.arguments.first().is_some_and(|input| {
        !super::super::receiver_inference::may_be_receiver(input, super::super::TCPCLIENT_CLASS)
    }) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-NET-WRITE-CLIENT",
            "write requires a tcpclient value",
            0,
        ));
    }
    if request.arguments.get(1).is_some_and(|input| {
        !matches!(
            input.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Character
                | ValueKindFact::String
                | ValueKindFact::Unknown
        )
    }) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-NET-WRITE-DATA",
            "write payload must be real numeric, logical, or text data",
            1,
        ));
    }
    if let Some(input) = request.arguments.get(2) {
        if crate::catalog::inference::known_invalid_text_scalar(input)
            || !matches!(
                input.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
            || request
                .literals
                .literal_string_at(2)
                .is_some_and(|text| NetDataType::parse(&text).is_none())
        {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-NET-WRITE-DATATYPE",
                "write datatype must be a supported text scalar",
                2,
            ));
        }
    }
    result
}
