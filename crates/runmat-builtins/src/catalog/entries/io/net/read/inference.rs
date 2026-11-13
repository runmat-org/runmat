use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

use super::super::NetDataType;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let datatype = if request.arguments.len() < 3 {
        Some(NetDataType::UInt8)
    } else {
        request
            .literals
            .literal_string_at(2)
            .and_then(|text| NetDataType::parse(&text))
    };
    let count = (request.arguments.len() >= 2)
        .then(|| {
            request
                .literals
                .literal_args
                .get(1)
                .and_then(positive_count)
        })
        .flatten();
    let output = datatype
        .map(|datatype| output_fact(datatype, count))
        .unwrap_or_else(|| {
            let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
            fact.residency = ResidencyFact::Host;
            fact
        });
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if !(1..=3).contains(&request.arguments.len()) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-NET-READ-ARITY",
            "read requires a client and at most count and datatype controls",
            0,
        ));
    }
    if request.arguments.first().is_some_and(|input| {
        !super::super::receiver_inference::may_be_receiver(input, super::super::TCPCLIENT_CLASS)
    }) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-NET-READ-CLIENT",
            "read requires a tcpclient value",
            0,
        ));
    }
    if let Some(input) = request.arguments.get(1) {
        if input.shape.is_definitely_nonscalar()
            || !matches!(
                input.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Unknown
            )
            || request
                .literals
                .literal_args
                .get(1)
                .is_some_and(known_invalid_count)
        {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-NET-READ-COUNT",
                "read count must be a positive finite integer scalar",
                1,
            ));
        }
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
                "RM-CATALOG-NET-READ-DATATYPE",
                "read datatype must be a supported text scalar",
                2,
            ));
        }
    }
    result
}

fn output_fact(datatype: NetDataType, count: Option<usize>) -> ValueFact {
    if datatype == NetDataType::String {
        return ValueFact::scalar(ValueKindFact::String);
    }
    let shape = match count {
        Some(count) => ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(count)],
        },
        None => ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Unknown],
        },
    };
    let kind = match datatype {
        NetDataType::Char => ValueKindFact::Character,
        NetDataType::UInt8 => numeric(NumericClass::UInt8),
        NetDataType::Int8 => numeric(NumericClass::Int8),
        NetDataType::UInt16 => numeric(NumericClass::UInt16),
        NetDataType::Int16 => numeric(NumericClass::Int16),
        NetDataType::UInt32 => numeric(NumericClass::UInt32),
        NetDataType::Int32 => numeric(NumericClass::Int32),
        NetDataType::UInt64 => numeric(NumericClass::UInt64),
        NetDataType::Int64 => numeric(NumericClass::Int64),
        NetDataType::Single => numeric(NumericClass::Single),
        NetDataType::Double => numeric(NumericClass::Double),
        NetDataType::String => unreachable!("string scalar handled above"),
    };
    ValueFact::proven(kind, shape, StorageFact::Dense)
}

fn numeric(class: NumericClass) -> ValueKindFact {
    ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    })
}

fn positive_count(literal: &LiteralValue) -> Option<usize> {
    match literal {
        LiteralValue::Integer { .. } => usize::try_from(literal.integer_value_i128()?)
            .ok()
            .filter(|n| *n > 0),
        LiteralValue::Number(value) => positive_float_count(*value),
        LiteralValue::Real {
            text,
            class: NumericClass::Double,
        } => positive_float_count(text.parse().ok()?),
        _ => None,
    }
}

fn positive_float_count(value: f64) -> Option<usize> {
    (value.is_finite() && value > 0.0 && value.fract() == 0.0 && value < usize::MAX as f64)
        .then_some(value as usize)
}

fn known_invalid_count(literal: &LiteralValue) -> bool {
    matches!(
        literal,
        LiteralValue::Integer { .. }
            | LiteralValue::Number(_)
            | LiteralValue::Real {
                class: NumericClass::Double,
                ..
            }
    ) && positive_count(literal).is_none()
}
