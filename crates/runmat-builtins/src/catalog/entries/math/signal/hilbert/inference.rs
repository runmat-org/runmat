use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-HILBERT-ARITY",
            "hilbert expects a signal and optional FFT length",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let signal = request.arguments.first();
    if let Some(signal) = signal {
        if matches!(signal.storage, StorageFact::Sparse | StorageFact::Opaque)
            || !matches!(
                signal.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            )
            || matches!(signal.kind, ValueKindFact::Numeric(NumericFact { domain: NumericDomain::Complex, class }) if class.integer_class().is_some())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-HILBERT-SIGNAL",
                "hilbert requires a dense real or complex floating signal",
                0,
            ));
        }
    }
    let length = request.arguments.get(1);
    if let Some(length) = length {
        let empty_control = matches!(
            request.literals.literal_args.get(1),
            Some(LiteralValue::Empty)
        ) || length.shape.element_count() == Some(0);
        if !empty_control
            && (length.shape.is_definitely_nonscalar()
                || !matches!(
                    length.kind,
                    ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    }) | ValueKindFact::Logical
                        | ValueKindFact::Unknown
                )
                || matches!(length.storage, StorageFact::Sparse | StorageFact::Opaque)
                || literal_length(request, 1) == Length::Invalid)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-HILBERT-LENGTH",
                "FFT length must be a nonnegative integer scalar or []",
                1,
            ));
        }
    }

    let output = match signal {
        Some(signal) if diagnostics.is_empty() => output_fact(signal, length.is_some(), request),
        _ => ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Length {
    Empty,
    Known(usize),
    Unknown,
    Invalid,
}

fn literal_length(request: &CallRequest, index: usize) -> Length {
    match request.literals.literal_args.get(index) {
        Some(LiteralValue::Empty) => Length::Empty,
        Some(LiteralValue::Number(value)) => nonnegative_length(*value),
        Some(LiteralValue::Integer { .. }) => request.literals.literal_args[index]
            .integer_value_i128()
            .and_then(|value| usize::try_from(value).ok())
            .map_or(Length::Invalid, Length::Known),
        Some(LiteralValue::Bool(value)) => Length::Known(usize::from(*value)),
        Some(literal @ LiteralValue::Real { .. }) => LiteralContext::numeric_from_literal(literal)
            .map_or(Length::Invalid, nonnegative_length),
        _ => Length::Unknown,
    }
}

fn nonnegative_length(value: f64) -> Length {
    if !value.is_finite() || value < 0.0 {
        Length::Invalid
    } else {
        // Match the runtime FFT length parser after source-class rounding.
        let rounded = value.round();
        if (rounded - value).abs() > f64::EPSILON
            || rounded > usize::MAX as f64
            || (usize::BITS == 64 && rounded == usize::MAX as f64)
        {
            Length::Invalid
        } else {
            Length::Known(rounded as usize)
        }
    }
}

fn output_fact(signal: &ValueFact, has_length: bool, request: &CallRequest) -> ValueFact {
    let class = match signal.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            ..
        }) => NumericClass::Single,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical => NumericClass::Double,
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let empty_control = request
        .arguments
        .get(1)
        .is_some_and(|value| value.shape.element_count() == Some(0));
    let shape =
        if !has_length || empty_control || matches!(literal_length(request, 1), Length::Empty) {
            signal.shape.clone()
        } else {
            transformed_shape(&signal.shape, literal_length(request, 1))
        };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Complex,
        }),
        shape,
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

fn transformed_shape(shape: &ShapeFact, length: Length) -> ShapeFact {
    let Some(mut dims) = shape.known_dims() else {
        return ShapeFact::Unknown;
    };
    if dims.is_empty() {
        dims = vec![Some(1), Some(1)];
    }
    let axis = dims
        .iter()
        .position(|dimension| *dimension != Some(1))
        .unwrap_or(0);
    if dims[axis].is_none() {
        return ShapeFact::Unknown;
    }
    dims[axis] = match length {
        Length::Known(value) => Some(value),
        Length::Unknown => None,
        Length::Empty | Length::Invalid => dims[axis],
    };
    ShapeFact::from(dims)
}
