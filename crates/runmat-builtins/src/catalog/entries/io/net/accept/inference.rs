use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut fields = BTreeMap::new();
    fields.insert("Address".into(), ValueFact::scalar(ValueKindFact::String));
    fields.insert(
        "Port".into(),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real,
        })),
    );
    fields.insert(
        "Connected".into(),
        ValueFact::scalar(ValueKindFact::Logical),
    );
    let output = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(super::super::TCPCLIENT_CLASS.owned()),
        properties: fields,
        properties_complete: false,
        handle_semantics: Some(true),
    }));
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.is_empty() || !(request.arguments.len() - 1).is_multiple_of(2) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-NET-ACCEPT-ARITY",
            "accept requires a server and complete Timeout Name/Value pairs",
            0,
        ));
    }
    if request.arguments.first().is_some_and(|server| {
        !super::super::receiver_inference::may_be_receiver(server, super::super::TCPSERVER_CLASS)
    }) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-NET-ACCEPT-SERVER",
            "accept requires a tcpserver value",
            0,
        ));
    }
    for index in (1..request.arguments.len()).step_by(2) {
        let name = &request.arguments[index];
        if crate::catalog::inference::known_invalid_text_scalar(name)
            || !matches!(
                name.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
            || request
                .literals
                .literal_string_at(index)
                .is_some_and(|literal| {
                    super::super::NetConstructorOption::accept(&literal).is_none()
                })
        {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-NET-ACCEPT-OPTION",
                "accept supports only the Timeout option",
                index,
            ));
        }
        if let Some(value) = request.arguments.get(index + 1) {
            if value.shape.is_definitely_nonscalar()
                || !matches!(
                    value.kind,
                    ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    }) | ValueKindFact::Unknown
                )
                || request
                    .literals
                    .literal_args
                    .get(index + 1)
                    .is_some_and(|literal| {
                        super::super::constructor_inference::known_invalid_timeout(literal, true)
                    })
            {
                result.diagnostics.push(argument_error(
                    "RM-CATALOG-NET-ACCEPT-TIMEOUT",
                    "accept timeout must be a nonnegative numeric scalar or positive Inf",
                    index + 1,
                ));
            }
        }
    }
    result
}
