use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(4..=5).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CHEB2ORD-ARITY",
            "cheb2ord requires four design arguments and optional 's'",
            request.arguments.len().saturating_sub(1),
        ));
    }
    for (index, value) in request.arguments.iter().take(4).enumerate() {
        let edge = index < 2;
        let count = value.shape.element_count();
        if matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
            || !matches!(
                value.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            )
            || if edge {
                count.is_some_and(|n| n != 1 && n != 2)
            } else {
                value.shape.is_definitely_nonscalar()
            }
            || request
                .literals
                .literal_args
                .get(index)
                .is_some_and(proven_invalid_positive)
        {
            diagnostics.push(argument_error(
                if edge {
                    "RM-CATALOG-CHEB2ORD-FREQUENCY"
                } else {
                    "RM-CATALOG-CHEB2ORD-ATTENUATION"
                },
                if edge {
                    "frequency must be a positive finite real edge or pair"
                } else {
                    "attenuation must be a positive finite real scalar"
                },
                index,
            ));
        }
    }
    if let (Some(wp), Some(ws)) = (request.arguments.first(), request.arguments.get(1)) {
        if wp
            .shape
            .element_count()
            .zip(ws.shape.element_count())
            .is_some_and(|(a, b)| a != b)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-CHEB2ORD-FREQUENCY",
                "passband and stopband edge counts must agree",
                1,
            ));
        }
    }
    if let Some(option) = request.arguments.get(4) {
        if !matches!(
            option.kind,
            ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Unknown
        ) || request
            .literals
            .literal_args
            .get(4)
            .is_some_and(proven_invalid_option)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-CHEB2ORD-OPTION",
                "the optional analog selector must be 's'",
                4,
            ));
        }
    }
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let cutoff_shape = match request
        .arguments
        .first()
        .and_then(|value| value.shape.element_count())
    {
        Some(1) => ShapeFact::Scalar,
        Some(2) => ShapeFact::from(vec![Some(1), Some(2)]),
        _ => ShapeFact::Unknown,
    };
    let mut cutoff = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        cutoff_shape.clone(),
        match &cutoff_shape {
            ShapeFact::Scalar => StorageFact::Scalar,
            ShapeFact::Unknown => StorageFact::Unknown,
            _ => StorageFact::Dense,
        },
    );
    if cutoff_shape == ShapeFact::Unknown {
        cutoff.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    let outputs = if diagnostics.is_empty() {
        vec![scalar, cutoff]
    } else {
        vec![ValueFact::unknown(DynamicReason::UnsupportedRepresentation); 2]
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn proven_invalid_positive(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Vector(values) => return values.iter().any(proven_invalid_positive),
        LiteralValue::Complex {
            real, imaginary, ..
        } => {
            return real
                .parse::<f64>()
                .is_ok_and(|n| !n.is_finite() || n <= 0.0)
                || imaginary
                    .parse::<f64>()
                    .is_ok_and(|n| !n.is_finite() || n.abs() > 1.0e-12);
        }
        LiteralValue::Integer { .. } if value.integer_is_exact_binary64() == Some(false) => {
            return true;
        }
        _ => {}
    }
    LiteralContext::numeric_from_literal(value).is_some_and(|n| !n.is_finite() || n <= 0.0)
}

fn proven_invalid_option(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Character(text)
        | LiteralValue::String(text)
        | LiteralValue::Keyword(text) => !text.trim().eq_ignore_ascii_case("s"),
        LiteralValue::Unknown | LiteralValue::Symbolic(_) => false,
        _ => true,
    }
}
