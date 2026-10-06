use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

fn invalid_bound_literal(literal: &LiteralValue, maximum: bool) -> bool {
    match literal {
        LiteralValue::Number(_) | LiteralValue::Real { .. } => {
            LiteralContext::numeric_from_literal(literal).is_some_and(|value| {
                value < 0.0
                    || !(value.is_finite() || maximum && value == f64::INFINITY)
                    || (value.is_finite()
                        && (value.fract() != 0.0
                            || value > usize::MAX as f64
                            || (usize::BITS == 64 && value == usize::MAX as f64)))
            })
        }
        LiteralValue::Integer { .. } => literal
            .integer_value_i128()
            .is_some_and(|value| usize::try_from(value).is_err()),
        LiteralValue::Bool(_)
        | LiteralValue::Character(_)
        | LiteralValue::String(_)
        | LiteralValue::Complex { .. }
        | LiteralValue::Vector(_)
        | LiteralValue::Matrix(_)
        | LiteralValue::Empty => true,
        _ => false,
    }
}

fn known_min_exceeds_max(min: &LiteralValue, max: &LiteralValue) -> bool {
    let exact_min = min.integer_value_i128();
    let exact_max = max.integer_value_i128();
    if let (Some(min), Some(max)) = (exact_min, exact_max) {
        return min > max;
    }
    let (Some(min_value), Some(max_value)) = (
        LiteralContext::numeric_from_literal(min),
        LiteralContext::numeric_from_literal(max),
    ) else {
        return false;
    };
    const EXACT_BINARY64_INTEGER: i128 = 1i128 << 53;
    if exact_min.is_some_and(|value| value.unsigned_abs() > EXACT_BINARY64_INTEGER as u128)
        || exact_max.is_some_and(|value| value.unsigned_abs() > EXACT_BINARY64_INTEGER as u128)
    {
        return false;
    }
    min_value.is_finite() && max_value.is_finite() && min_value > max_value
}

fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    output: Option<ValueFact>,
    min_args: usize,
    max_args: usize,
    code: &'static str,
) -> CallInference {
    let mut contract = CallContract::fixed(output.into_iter().collect());
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if !(min_args..=max_args).contains(&request.arguments.len()) {
        result
            .diagnostics
            .push(argument_error(code, "unsupported arity-check form", 0));
        return result;
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        let selector = index == 3;
        let known_bad = if selector {
            !matches!(
                argument.kind,
                ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Unknown
            )
        } else {
            !matches!(
                argument.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Unknown
            )
        };
        let invalid_shape = if selector {
            match argument.kind {
                ValueKindFact::String => argument.shape.is_definitely_nonscalar(),
                ValueKindFact::Character => argument
                    .shape
                    .known_dims()
                    .and_then(|dims| dims.first().copied().flatten())
                    .is_some_and(|rows| rows != 1),
                _ => false,
            }
        } else {
            argument.shape.is_definitely_nonscalar()
        };
        if known_bad || invalid_shape {
            result.diagnostics.push(argument_error(
                code,
                "expected a scalar arity bound or selector",
                index,
            ));
        }
        if matches!(
            argument.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) {
            result.diagnostics.push(argument_error(
                code,
                "arity controls must be host-resident",
                index,
            ));
        }
        if !selector
            && request
                .literals
                .literal_args
                .get(index)
                .is_some_and(|literal| invalid_bound_literal(literal, index == 1))
        {
            result.diagnostics.push(argument_error(
                code,
                "invalid nonnegative integer bound",
                index,
            ));
        }
    }
    if request.arguments.len() == 4
        && request
            .literals
            .literal_string_at(3)
            .is_some_and(|selector| selector.trim() != "struct")
    {
        result
            .diagnostics
            .push(argument_error(code, "expected the struct selector", 3));
    }
    if let (Some(min), Some(max)) = (
        request.literals.literal_args.first(),
        request.literals.literal_args.get(1),
    ) {
        if known_min_exceeds_max(min, max) {
            result
                .diagnostics
                .push(argument_error(code, "minimum exceeds maximum", 1));
        }
    }
    result
}

pub(super) fn infer_narginchk(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, None, 2, 2, "RM-NARGINCHK-INPUT")
}

pub(super) fn infer_nargoutchk(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let output = match request.arguments.len() {
        3 => Some(ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![None, None]),
            StorageFact::Dense,
        )),
        4 => Some(ValueFact::proven(
            ValueKindFact::Struct(StructFact::scalar(Default::default(), false)),
            ShapeFact::Scalar,
            StorageFact::Dense,
        )),
        _ => None,
    };
    infer(request, entry, output, 2, 4, "RM-NARGOUTCHK-INPUT")
}
