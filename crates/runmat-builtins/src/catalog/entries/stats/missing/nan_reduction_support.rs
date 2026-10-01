use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, reduction_shape, CallContract, CallInference, CallRequest, DynamicReason,
    InferenceDiagnostic, LiteralValue, NumericClass, NumericDomain, NumericFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
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

fn has_keyword(request: &CallRequest, searched: &str) -> bool {
    request
        .literals
        .literal_args
        .iter()
        .filter_map(keyword)
        .any(|value| value.eq_ignore_ascii_case(searched))
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

fn unknown_reduction_shape(input: &ShapeFact) -> ShapeFact {
    input
        .rank()
        .map_or(ShapeFact::Unknown, |rank| ShapeFact::Ranked { rank })
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
        return unknown_reduction_shape(input);
    }
    reduction_shape(input, Some(index + 1))
}

fn selected_reduction_shape(request: &CallRequest, input: &ShapeFact, alias: Alias) -> ShapeFact {
    if has_keyword(request, "all") {
        return ShapeFact::Scalar;
    }
    // std and var place normalization before the optional dimension. Other
    // aliases may take a dimension immediately after the data.
    let dimension_index = if alias.normalization_before_dimension
        || (alias.pairwise_min
            && matches!(
                request.literals.literal_args.get(1),
                Some(LiteralValue::Empty)
            )) {
        2
    } else {
        1
    };
    if let Some(literal) = request.literals.literal_args.get(dimension_index) {
        if let Some(dim) = dimension(literal) {
            if matches!(input, ShapeFact::Ranked { .. })
                || input.rank().is_some_and(|rank| dim > rank)
            {
                return input.clone();
            }
            return reduction_shape(input, Some(dim));
        }
        if keyword(literal).is_none() {
            return unknown_reduction_shape(input);
        }
    }
    default_reduction_shape(input)
}

fn pairwise_min_fact(request: &CallRequest, left: &ValueFact) -> Option<ValueFact> {
    let right = request.arguments.get(1)?;
    let (Some(a), Some(b)) = (left.numeric(), right.numeric()) else {
        return None;
    };
    if a.domain != NumericDomain::Real || b.domain != NumericDomain::Real {
        return Some(ValueFact::unknown(DynamicReason::UnsupportedRepresentation));
    }
    let class = if a.class == b.class {
        a.class
    } else if left.is_scalar() && a.class == NumericClass::Double {
        b.class
    } else if right.is_scalar() && b.class == NumericClass::Double {
        a.class
    } else {
        return Some(ValueFact::unknown(DynamicReason::RuntimeValue));
    };
    let shape = if left.is_scalar() {
        right.shape.clone()
    } else if right.is_scalar() || left.shape.is_proven_equivalent(&right.shape) {
        left.shape.clone()
    } else {
        ShapeFact::Unknown
    };
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut result = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
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
                } else if has_keyword(request, "like") {
                    // A runtime prototype can select an output class not implied
                    // by the data class. Keep that class dynamic.
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                } else if numeric.class == NumericClass::Single
                    && alias.single_output_dynamic
                    && (!alias.output_class_control
                        || (!has_keyword(request, "native") && !has_keyword(request, "double")))
                {
                    // The owning host reduction and some resident routes do
                    // not currently agree on every Single result class.
                    // Preserve the provable geometry without claiming a class.
                    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
                    result.shape = selected_reduction_shape(request, &input.shape, alias);
                    result
                } else {
                    let class = if has_keyword(request, "double")
                        || (alias.defaults_integer_to_double
                            && is_integer(numeric.class)
                            && !has_keyword(request, "native"))
                    {
                        NumericClass::Double
                    } else {
                        numeric.class
                    };
                    let shape = selected_reduction_shape(request, &input.shape, alias);
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
                    result.residency = runmat_types::ResidencyFact::Unknown;
                    result
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
