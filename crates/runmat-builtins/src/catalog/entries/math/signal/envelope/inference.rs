use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ENVELOPE-ARITY",
            "envelope expects x and optional length and method",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(signal) = request.arguments.first() {
        if matches!(signal.storage, StorageFact::Sparse | StorageFact::Opaque)
            || !matches!(
                signal.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            )
            || signal.shape.rank().is_some_and(|rank| rank > 2)
            || signal.shape.element_count() == Some(0)
            || request
                .literals
                .literal_args
                .first()
                .is_some_and(nonfinite_signal_literal)
            || request
                .literals
                .literal_args
                .first()
                .is_some_and(inexact_integer_literal)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ENVELOPE-SIGNAL",
                "envelope requires a nonempty dense real numeric vector or matrix",
                0,
            ));
        }
    }
    if let Some(length) = request.arguments.get(1) {
        if length.shape.is_definitely_nonscalar()
            || !matches!(
                length.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            )
            || matches!(length.storage, StorageFact::Sparse | StorageFact::Opaque)
            || request
                .literals
                .literal_args
                .get(1)
                .is_some_and(invalid_length)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ENVELOPE-LENGTH",
                "n must be a positive integer scalar",
                1,
            ));
        }
    }
    if let Some(method) = request.arguments.get(2) {
        if !matches!(
            method.kind,
            ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Unknown
        ) || request
            .literals
            .literal_args
            .get(2)
            .is_some_and(invalid_method)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ENVELOPE-METHOD",
                "method must be analytic, rms, or peak",
                2,
            ));
        }
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 2)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-ENVELOPE-OUTPUTS",
            "envelope has at most two numerical outputs",
            request.arguments.len(),
        ));
    }
    let output = match request.arguments.first() {
        Some(signal) if diagnostics.is_empty() => output_fact(signal),
        _ => ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
    };
    let mut contract = CallContract::fixed(vec![output.clone(), output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn nonfinite_signal_literal(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Number(value) => !value.is_finite(),
        LiteralValue::Real { .. } => {
            LiteralContext::numeric_from_literal(literal).is_some_and(|value| !value.is_finite())
        }
        LiteralValue::Vector(values) => values.iter().any(nonfinite_signal_literal),
        LiteralValue::Matrix(rows) => rows.iter().flatten().any(nonfinite_signal_literal),
        _ => false,
    }
}

fn inexact_integer_literal(literal: &LiteralValue) -> bool {
    if literal.integer_is_exact_binary64() == Some(false) {
        return true;
    }
    match literal {
        LiteralValue::Vector(values) => values.iter().any(inexact_integer_literal),
        LiteralValue::Matrix(rows) => rows.iter().flatten().any(inexact_integer_literal),
        _ => false,
    }
}

fn output_fact(signal: &ValueFact) -> ValueFact {
    let class = match signal.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            ..
        }) => NumericClass::Single,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical => NumericClass::Double,
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        signal.shape.clone(),
        StorageFact::Unknown,
    );
    output.residency = ResidencyFact::Unknown;
    output.layout = LayoutFact::Unknown;
    output.contiguity = ContiguityFact::Unknown;
    output.alias = AliasFact::Unknown;
    output.view = ViewFact::Unknown;
    output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    output
}

fn invalid_length(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Bool(value) => !value,
        LiteralValue::Integer { .. } => literal
            .integer_value_i128()
            .is_some_and(|n| n <= 0 || usize::try_from(n).is_err()),
        _ => match LiteralContext::numeric_from_literal(literal) {
            Some(n) => {
                let rounded = n.round();
                !n.is_finite()
                    || rounded < 1.0
                    || (rounded - n).abs() > 1e-9
                    || rounded > usize::MAX as f64
                    || (usize::BITS == 64 && rounded == usize::MAX as f64)
            }
            None => !matches!(literal, LiteralValue::Unknown),
        },
    }
}

fn invalid_method(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Character(value)
        | LiteralValue::String(value)
        | LiteralValue::Keyword(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "analytic" | "rms" | "peak"
        ),
        LiteralValue::Unknown => false,
        _ => true,
    }
}
