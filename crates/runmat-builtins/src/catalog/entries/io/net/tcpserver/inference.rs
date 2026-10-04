use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut fields = BTreeMap::new();
    fields.insert(
        "ServerAddress".into(),
        ValueFact::scalar(ValueKindFact::String),
    );
    fields.insert("ServerPort".into(), real_double());
    fields.insert("Timeout".into(), real_double());
    fields.insert(
        "Connected".into(),
        ValueFact::scalar(ValueKindFact::Logical),
    );
    let output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, false)));
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() < 2 || !(request.arguments.len() - 2).is_multiple_of(2) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-TCPSERVER-ARITY",
            "tcpserver requires address, port, then complete Name/Value pairs",
            0,
        ));
    }
    if request.arguments.first().is_some_and(|arg| {
        arg.shape.is_definitely_nonscalar()
            || !matches!(
                arg.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
    }) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-TCPSERVER-ADDRESS",
            "tcpserver address must be scalar text",
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
            "RM-CATALOG-TCPSERVER-PORT",
            "tcpserver port must be an integer scalar in 0..65535 (zero is a RunMat extension)",
            1,
        ));
    }
    for index in (2..request.arguments.len()).step_by(2) {
        let name = &request.arguments[index];
        if name.shape.is_definitely_nonscalar()
            || !matches!(
                name.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
            || request
                .literals
                .literal_string_at(index)
                .is_some_and(|literal| !known_option(&literal))
        {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-TCPSERVER-OPTION",
                "unsupported tcpserver option name",
                index,
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

fn known_option(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "timeout" | "userdata" | "name" | "byteorder"
    )
}
