use super::super::support::{
    admitted_max_pairwise_options, admitted_max_reduction_controls, has_literal_zero_axis,
    inferred_empty_max_shape, inferred_max_shape, is_empty_placeholder,
};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    broadcast_shape_checked, infer_call, infer_reduction, BroadcastShapeError, CallContract,
    CallInference, CallRequest, DynamicReason, InferenceDiagnostic, LiteralValue, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum MaxCallFamily {
    Reduction { option_start: usize },
    Pairwise,
    Unresolved,
}

fn call_family(request: &CallRequest) -> MaxCallFamily {
    match request.arguments.len() {
        0 | 1 => MaxCallFamily::Reduction { option_start: 1 },
        _ => match request.literals.literal_args.get(1) {
            Some(literal) if is_empty_placeholder(literal) => {
                MaxCallFamily::Reduction { option_start: 2 }
            }
            Some(LiteralValue::Unknown | LiteralValue::Symbolic(_)) | None => {
                match request.arguments.get(1) {
                    Some(value) if known_runtime_empty_placeholder(value) => {
                        MaxCallFamily::Reduction { option_start: 2 }
                    }
                    Some(value) if value.shape.element_count() == Some(0) => {
                        MaxCallFamily::Unresolved
                    }
                    Some(value) if value.shape.element_count().is_some() => MaxCallFamily::Pairwise,
                    _ => MaxCallFamily::Unresolved,
                }
            }
            Some(_) => MaxCallFamily::Pairwise,
        },
    }
}

fn known_runtime_empty_placeholder(value: &ValueFact) -> bool {
    if value.shape.element_count() != Some(0) || value.residency != ResidencyFact::Host {
        return false;
    }
    match &value.kind {
        ValueKindFact::Numeric(numeric) => {
            numeric.domain == NumericDomain::Real && value.storage == StorageFact::Dense
        }
        ValueKindFact::Logical => value.storage == StorageFact::Dense,
        ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Cell(_) => {
            value.storage != StorageFact::Sparse
        }
        _ => false,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let family = call_family(request);
    let outputs = match request.arguments.first() {
        None => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.MAX.INVALID_ARGUMENT",
                "max requires an input array",
            ));
            vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2]
        }
        Some(input) => match family {
            MaxCallFamily::Reduction { option_start } => {
                reduction_outputs(input, request, option_start, &mut diagnostics)
            }
            // A pairwise call has no index output. Its own option grammar and
            // operand classes must be admitted independently of reduction.
            MaxCallFamily::Pairwise => vec![pairwise_output(input, request, &mut diagnostics)],
            // The second value may be an empty reduction placeholder or a
            // pairwise operand. Do not prove either family's output contract.
            MaxCallFamily::Unresolved => {
                vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2]
            }
        },
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub(super) fn infer_fusion_fact(request: &CallRequest) -> Option<ValueFact> {
    if !matches!(call_family(request), MaxCallFamily::Pairwise) {
        return None;
    }
    let left = request.arguments.first()?;
    pairwise_output_checked(left, request).ok()
}

fn reduction_outputs(
    input: &ValueFact,
    request: &CallRequest,
    option_start: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Vec<ValueFact> {
    if request
        .literals
        .literal_args
        .get(option_start)
        .is_some_and(has_literal_zero_axis)
    {
        diagnostics.extend(infer_reduction(input, Some(0)).diagnostics);
        return dynamic_pair();
    }
    if input.storage == StorageFact::Sparse {
        return dynamic_pair();
    }
    let output_kind = match &input.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.class.integer_class().is_some()
                && numeric.domain == NumericDomain::Complex =>
        {
            return dynamic_pair();
        }
        ValueKindFact::Numeric(numeric) => ValueKindFact::Numeric(*numeric),
        ValueKindFact::Logical => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Unknown | ValueKindFact::Object(_) => return dynamic_pair(),
        _ => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.MAX.INVALID_INPUT",
                "max requires numeric, logical, or supported ordinal categorical input",
            ));
            return dynamic_pair();
        }
    };
    let Some(controls) =
        admitted_max_reduction_controls(request.arguments.len(), &request.literals, option_start)
    else {
        return dynamic_pair();
    };
    let shape = if input.shape.element_count() == Some(0) {
        let native_integer = matches!(&input.kind, ValueKindFact::Numeric(numeric) if numeric.class.integer_class().is_some());
        inferred_empty_max_shape(&input.shape, &controls, native_integer)
    } else {
        inferred_max_shape(&input.shape, &controls)
    };
    let mut maximum = ValueFact::proven(output_kind, shape.clone(), StorageFact::Unknown);
    let mut index = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Unknown,
    );
    if input.residency != ResidencyFact::Host {
        maximum.residency = ResidencyFact::Unknown;
        index.residency = ResidencyFact::Unknown;
    }
    vec![maximum, index]
}

fn dynamic_pair() -> Vec<ValueFact> {
    vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2]
}

fn pairwise_output(
    left: &ValueFact,
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    match pairwise_output_checked(left, request) {
        Ok(fact) => fact,
        Err(error) => {
            diagnostics.push(error.into_diagnostic());
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
    }
}

fn pairwise_output_checked(
    left: &ValueFact,
    request: &CallRequest,
) -> Result<ValueFact, BroadcastShapeError> {
    let Some(right) = request.arguments.get(1) else {
        return Ok(ValueFact::unknown(DynamicReason::RuntimeValue));
    };
    if !admitted_max_pairwise_options(request.arguments.len(), &request.literals)
        || left.storage == StorageFact::Sparse
        || right.storage == StorageFact::Sparse
    {
        return Ok(ValueFact::unknown(DynamicReason::RuntimeValue));
    }
    let kind = match (&left.kind, &right.kind) {
        (ValueKindFact::Numeric(a), ValueKindFact::Numeric(b)) if a == b => {
            if a.domain == NumericDomain::Complex && a.class.integer_class().is_some() {
                return Ok(ValueFact::unknown(DynamicReason::RuntimeValue));
            }
            let known_array_operand = left.shape.element_count().is_some_and(|count| count > 1)
                || right.shape.element_count().is_some_and(|count| count > 1);
            if a.domain == NumericDomain::Complex && !known_array_operand {
                // Scalar complex comparison can return a real scalar when
                // both imaginary components are zero. Runtime values decide.
                return Ok(ValueFact::unknown(DynamicReason::RuntimeValue));
            }
            ValueKindFact::Numeric(*a)
        }
        (ValueKindFact::Logical, ValueKindFact::Logical) => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => return Ok(ValueFact::unknown(DynamicReason::RuntimeValue)),
    };
    let shape = broadcast_shape_checked(&left.shape, &right.shape)?;
    let shape = if shape.element_count() == Some(1) {
        ShapeFact::Scalar
    } else {
        shape
    };
    let mut result = ValueFact::proven(kind, shape, StorageFact::Unknown);
    if left.residency != ResidencyFact::Host || right.residency != ResidencyFact::Host {
        result.residency = ResidencyFact::Unknown;
    }
    Ok(result)
}
