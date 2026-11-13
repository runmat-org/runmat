use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut fields = BTreeMap::new();
    fields.insert("Address".into(), ValueFact::scalar(ValueKindFact::String));
    fields.insert("Port".into(), real_double());
    fields.insert("Timeout".into(), real_double());
    fields.insert("ConnectTimeout".into(), real_double());
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
    if request.arguments.len() < 2 || !(request.arguments.len() - 2).is_multiple_of(2) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-TCPCLIENT-ARITY",
            "tcpclient requires host, port, then complete Name/Value pairs",
            0,
        ));
    }
    if request.arguments.first().is_some_and(|arg| {
        crate::catalog::inference::known_invalid_text_scalar(arg)
            || !matches!(
                arg.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
    }) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-TCPCLIENT-HOST",
            "tcpclient host must be scalar text",
            0,
        ));
    }
    if request.arguments.get(1).is_some_and(|arg| {
        arg.shape.is_definitely_nonscalar()
            || !matches!(
                arg.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Unknown
            )
            || request
                .literals
                .literal_args
                .get(1)
                .is_some_and(super::super::constructor_inference::known_invalid_port)
    }) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-TCPCLIENT-PORT",
            "tcpclient port must be an integer scalar in 0..65535 (zero is a RunMat extension)",
            1,
        ));
    }
    for index in (2..request.arguments.len()).step_by(2) {
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
                    super::super::NetConstructorOption::tcpclient(&literal).is_none()
                })
        {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-TCPCLIENT-OPTION",
                "unsupported tcpclient option name",
                index,
            ));
        }
        if request
            .literals
            .literal_string_at(index)
            .is_some_and(|literal| {
                matches!(
                    super::super::NetConstructorOption::tcpclient(&literal),
                    Some(
                        super::super::NetConstructorOption::Timeout
                            | super::super::NetConstructorOption::ConnectTimeout
                    )
                )
            })
            && request.arguments.get(index + 1).is_some_and(|value| {
                value.shape.is_definitely_nonscalar()
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
                            super::super::constructor_inference::known_invalid_timeout(
                                literal, true,
                            )
                        })
            })
        {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-TCPCLIENT-TIMEOUT",
                "tcpclient timeout controls require nonnegative numeric scalars or positive Inf",
                index + 1,
            ));
        }
    }
    result
}

fn real_double() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}
