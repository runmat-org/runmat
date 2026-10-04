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
    let output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, false)));
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
        arg.shape.is_definitely_nonscalar()
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
                .is_some_and(known_invalid_port)
    }) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-TCPCLIENT-PORT",
            "tcpclient port must be an integer scalar in 0..65535 (zero is a RunMat extension)",
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
                "RM-CATALOG-TCPCLIENT-OPTION",
                "unsupported tcpclient option name",
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

fn known_invalid_port(literal: &LiteralValue) -> bool {
    let value = match literal {
        LiteralValue::Integer { .. } => {
            return literal
                .integer_value_i128()
                .is_none_or(|n| !(0..=65_535).contains(&n))
        }
        LiteralValue::Number(n) => *n,
        LiteralValue::Real {
            text,
            class: NumericClass::Double,
        } => return text.parse::<f64>().ok().is_none_or(invalid_float_port),
        _ => return false,
    };
    invalid_float_port(value)
}

fn invalid_float_port(value: f64) -> bool {
    !value.is_finite() || value.fract() != 0.0 || !(0.0..=65_535.0).contains(&value)
}

fn known_option(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "timeout"
            | "connecttimeout"
            | "byteorder"
            | "userdata"
            | "name"
            | "inputbuffersize"
            | "outputbuffersize"
    )
}
