use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, reduction_shape, CallContract, CallInference, CallRequest, DynamicReason,
    InferenceDiagnostic, LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy)]
pub(super) struct Alias {
    pub(super) rejects_integer_data: bool,
    pub(super) defaults_integer_to_double: bool,
    pub(super) normalization_before_dimension: bool,
    pub(super) pairwise_min: bool,
    pub(super) single_output_dynamic: bool,
    pub(super) output_class_control: bool,
    pub(super) complex_error_code: Option<&'static str>,
}

fn is_integer(class: NumericClass) -> bool {
    class.integer_class().is_some()
}

fn keyword(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::String(text)
        | LiteralValue::Character(text)
        | LiteralValue::Keyword(text) => Some(text),
        _ => None,
    }
}

fn literal_is(value: &LiteralValue, searched: &str) -> bool {
    keyword(value).is_some_and(|value| value.eq_ignore_ascii_case(searched))
}

fn dimension(value: &LiteralValue) -> Option<usize> {
    match value {
        LiteralValue::Integer { text, .. } => text.parse::<usize>().ok().filter(|dim| *dim > 0),
        LiteralValue::Number(value) => (value.is_finite()
            && *value >= 1.0
            && value.fract() == 0.0
            && *value < usize::MAX as f64)
            .then_some(*value as usize),
        LiteralValue::Real { text, .. } => {
            let value = text.parse::<f64>().ok()?;
            dimension(&LiteralValue::Number(value))
        }
        _ => None,
    }
}

fn default_reduction_shape(input: &ShapeFact) -> ShapeFact {
    if matches!(input, ShapeFact::Ranked { .. }) {
        return input.clone();
    }
    let Some(dims) = input.known_dims() else {
        return ShapeFact::Unknown;
    };
    let Some(index) = dims.iter().position(|dim| *dim != Some(1)) else {
        return ShapeFact::Scalar;
    };
    if dims[index].is_none() {
        return input
            .rank()
            .map_or(ShapeFact::Unknown, |rank| ShapeFact::Ranked { rank });
    }
    reduction_shape(input, Some(index + 1))
}

fn normalization(value: &LiteralValue) -> bool {
    matches!(value, LiteralValue::Empty)
        || matches!(dimension(value), Some(1))
        || matches!(value, LiteralValue::Number(0.0))
        || matches!(value, LiteralValue::Integer { text, .. } if text == "0")
        || matches!(value, LiteralValue::Real { text, .. } if text.parse::<f64>() == Ok(0.0))
}

// This is a proof boundary, not a second implementation of the reductions'
// option grammar. Only complete, positionally unambiguous literal forms are
// established here. The owning reductions must eventually expose typed
// options to both runtime admission and catalog inference.
fn simple_reduction_proof(
    request: &CallRequest,
    input: &ShapeFact,
    input_class: NumericClass,
    alias: Alias,
) -> Option<(ShapeFact, Option<NumericClass>)> {
    let args = &request.arguments;
    let literals = &request.literals.literal_args;
    if args.len() != literals.len() {
        return None;
    }
    match literals.get(1..)? {
        [] => Some((default_reduction_shape(input), None)),
        [option] if alias.normalization_before_dimension => {
            normalization(option).then(|| (default_reduction_shape(input), None))
        }
        [option] if literal_is(option, "all") => Some((ShapeFact::Scalar, None)),
        [option] if alias.output_class_control && literal_is(option, "native") => {
            Some((default_reduction_shape(input), Some(input_class)))
        }
        [option] if alias.output_class_control && literal_is(option, "double") => {
            Some((default_reduction_shape(input), Some(NumericClass::Double)))
        }
        [LiteralValue::Empty] if alias.pairwise_min => Some((default_reduction_shape(input), None)),
        [option] if !alias.pairwise_min => {
            dimension(option).map(|dim| (axis_reduction_shape(input, dim), None))
        }
        [norm, axis] if alias.normalization_before_dimension && normalization(norm) => {
            if literal_is(axis, "all") {
                Some((ShapeFact::Scalar, None))
            } else {
                dimension(axis).map(|dim| (axis_reduction_shape(input, dim), None))
            }
        }
        [LiteralValue::Empty, axis] if alias.pairwise_min => {
            dimension(axis).map(|dim| (axis_reduction_shape(input, dim), None))
        }
        _ => None,
    }
}

fn axis_reduction_shape(input: &ShapeFact, dim: usize) -> ShapeFact {
    if matches!(input, ShapeFact::Ranked { .. }) || input.rank().is_some_and(|rank| dim > rank) {
        input.clone()
    } else {
        reduction_shape(input, Some(dim))
    }
}

fn pairwise_min_fact(request: &CallRequest, left: &ValueFact) -> Option<ValueFact> {
    let right = request.arguments.get(1)?;
    if !matches!(left.residency, ResidencyFact::Host)
        || !matches!(right.residency, ResidencyFact::Host)
    {
        return None;
    }
    let (Some(a), Some(b)) = (left.numeric(), right.numeric()) else {
        return None;
    };
    if a.domain != NumericDomain::Real || b.domain != NumericDomain::Real {
        return None;
    }
    // Mixed integer/floating operands need a value-level exactness check, and
    // floating scalar pairs use a scalar-double path. Neither has a proven
    // class solely from these argument facts.
    if a.class != b.class || (left.is_scalar() && right.is_scalar() && !is_integer(a.class)) {
        return None;
    }
    let shape = if left.is_scalar() {
        right.shape.clone()
    } else if right.is_scalar() || left.shape.is_proven_equivalent(&right.shape) {
        left.shape.clone()
    } else {
        return None;
    };
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut result = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: a.class,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    );
    result.residency = runmat_types::ResidencyFact::Unknown;
    Some(result)
}

fn pairwise_min_form(request: &CallRequest, alias: Alias) -> Option<bool> {
    if !alias.pairwise_min || request.arguments.len() != 2 {
        return Some(false);
    }
    let second = &request.arguments[1];
    if second.numeric().is_none()
        || matches!(
            request.literals.literal_args.get(1),
            Some(LiteralValue::Empty)
        )
    {
        return Some(false);
    }
    if second.numeric()?.domain != NumericDomain::Real
        || !matches!(second.residency, ResidencyFact::Host)
    {
        return None;
    }
    second.shape.element_count().map(|count| count > 0)
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    alias: Alias,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let min_form = pairwise_min_form(request, alias);
    let output = match request.arguments.first() {
        None => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.MISSING.INVALID_ARGUMENT",
                format!("{} requires input data", entry.identity.name),
            ));
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
        Some(input) => {
            if min_form == Some(true) {
                pairwise_min_fact(request, input)
                    .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
            } else if min_form.is_none() {
                ValueFact::unknown(DynamicReason::RuntimeValue)
            } else if let Some(numeric) = input.numeric() {
                if alias.rejects_integer_data && is_integer(numeric.class) {
                    diagnostics.push(InferenceDiagnostic::error(
                        "RM.MISSING.UNSUPPORTED_TYPE",
                        format!("{} rejects typed-integer data", entry.identity.name),
                    ));
                    ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
                } else if alias.complex_error_code.is_some()
                    && numeric.domain != NumericDomain::Real
                {
                    diagnostics.push(InferenceDiagnostic::error(
                        alias.complex_error_code.expect("checked above"),
                        format!("{} rejects complex data", entry.identity.name),
                    ));
                    ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
                } else if let Some((shape, output_class)) =
                    simple_reduction_proof(request, &input.shape, numeric.class, alias)
                {
                    let explicit_output_class = output_class.is_some();
                    let class = match output_class {
                        Some(class) => class,
                        None if alias.defaults_integer_to_double && is_integer(numeric.class) => {
                            NumericClass::Double
                        }
                        None => numeric.class,
                    };
                    if numeric.class == NumericClass::Single
                        && alias.single_output_dynamic
                        && (!alias.output_class_control || !explicit_output_class)
                    {
                        // The owning host reduction and some resident routes do
                        // not currently agree on every Single result class.
                        let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
                        result.shape = shape;
                        result
                    } else {
                        let storage = match shape.element_count() {
                            Some(1) => StorageFact::Scalar,
                            Some(_) => StorageFact::Dense,
                            None => StorageFact::Unknown,
                        };
                        let mut result = ValueFact::proven(
                            ValueKindFact::Numeric(NumericFact {
                                class,
                                domain: if alias.complex_error_code.is_some() {
                                    NumericDomain::Real
                                } else {
                                    numeric.domain
                                },
                            }),
                            shape,
                            storage,
                        );
                        result.residency = ResidencyFact::Unknown;
                        result
                    }
                } else {
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                }
            } else {
                if !matches!(input.kind, ValueKindFact::Unknown) {
                    diagnostics.push(InferenceDiagnostic::error(
                        "RM.MISSING.UNSUPPORTED_TYPE",
                        format!("{} requires numeric input data", entry.identity.name),
                    ));
                }
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
        }
    };
    let outputs = if alias.pairwise_min && min_form != Some(true) {
        let index = if min_form.is_none() || output.numeric().is_none() {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        } else {
            let mut index = ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                output.shape.clone(),
                output.storage,
            );
            index.residency = runmat_types::ResidencyFact::Unknown;
            index
        };
        vec![output, index]
    } else {
        vec![output]
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}
