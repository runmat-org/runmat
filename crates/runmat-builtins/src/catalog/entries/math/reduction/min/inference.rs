use super::super::support::{
    admitted_min_pairwise_options, admitted_min_reduction_controls, has_literal_zero_axis,
    inferred_min_shape, is_empty_placeholder,
};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    broadcast_shape, infer_call, infer_reduction, CallContract, CallInference, CallRequest,
    DynamicReason, InferenceDiagnostic, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum MinCallFamily {
    Reduction { option_start: usize },
    Pairwise,
    Unresolved,
}

fn call_family(request: &CallRequest) -> MinCallFamily {
    match request.arguments.len() {
        0 | 1 => MinCallFamily::Reduction { option_start: 1 },
        _ => match request.literals.literal_args.get(1) {
            Some(literal) if is_empty_placeholder(literal) => {
                MinCallFamily::Reduction { option_start: 2 }
            }
            Some(LiteralValue::Unknown | LiteralValue::Symbolic(_)) | None => {
                match request
                    .arguments
                    .get(1)
                    .and_then(|value| value.shape.element_count())
                {
                    Some(0) => MinCallFamily::Reduction { option_start: 2 },
                    Some(_) => MinCallFamily::Pairwise,
                    None => MinCallFamily::Unresolved,
                }
            }
            Some(_) => MinCallFamily::Pairwise,
        },
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let family = call_family(request);
    let outputs = match request.arguments.first() {
        None => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.MIN.INVALID_ARGUMENT",
                "min requires an input array",
            ));
            vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2]
        }
        Some(input) => match family {
            MinCallFamily::Reduction { option_start } => {
                reduction_outputs(input, request, option_start, &mut diagnostics)
            }
            // A pairwise call has no index output. Its own option grammar and
            // operand classes must be admitted independently of reduction.
            MinCallFamily::Pairwise => vec![pairwise_output(input, request, &mut diagnostics)],
            // The second value may be an empty reduction placeholder or a
            // pairwise operand. Do not prove either family's output contract.
            MinCallFamily::Unresolved => {
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
        ValueKindFact::Unknown => return dynamic_pair(),
        _ => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.MIN.INVALID_INPUT",
                "min requires numeric or logical input",
            ));
            return dynamic_pair();
        }
    };
    let Some(controls) =
        admitted_min_reduction_controls(request.arguments.len(), &request.literals, option_start)
    else {
        return dynamic_pair();
    };
    // For empty all-dimensional reductions, the runtime returns an empty
    // array, not a scalar. A selector-dependent output cannot be certified
    // from the generic nonempty reduction rule.
    if input.shape.element_count() == Some(0) {
        return dynamic_pair();
    }
    let shape = inferred_min_shape(&input.shape, &controls);
    let mut minimum = ValueFact::proven(output_kind, shape.clone(), StorageFact::Unknown);
    let mut index = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Unknown,
    );
    if input.residency != ResidencyFact::Host {
        minimum.residency = ResidencyFact::Unknown;
        index.residency = ResidencyFact::Unknown;
    }
    vec![minimum, index]
}

fn dynamic_pair() -> Vec<ValueFact> {
    vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2]
}

fn pairwise_output(
    left: &ValueFact,
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    let Some(right) = request.arguments.get(1) else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if !admitted_min_pairwise_options(request.arguments.len(), &request.literals)
        || left.storage == StorageFact::Sparse
        || right.storage == StorageFact::Sparse
    {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    let kind = match (&left.kind, &right.kind) {
        (ValueKindFact::Numeric(a), ValueKindFact::Numeric(b)) if a == b => {
            if a.domain == NumericDomain::Complex && a.class.integer_class().is_some() {
                return ValueFact::unknown(DynamicReason::RuntimeValue);
            }
            let known_array_operand = left.shape.element_count().is_some_and(|count| count > 1)
                || right.shape.element_count().is_some_and(|count| count > 1);
            if a.domain == NumericDomain::Complex && !known_array_operand {
                // Scalar complex comparison can return a real scalar when
                // both imaginary components are zero. Runtime values decide.
                return ValueFact::unknown(DynamicReason::RuntimeValue);
            }
            ValueKindFact::Numeric(*a)
        }
        (ValueKindFact::Logical, ValueKindFact::Logical) => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let shape = match broadcast_shape(&left.shape, &right.shape) {
        Ok(shape) => shape,
        Err(error) => {
            diagnostics.push(error);
            return ValueFact::unknown(DynamicReason::RuntimeValue);
        }
    };
    let shape = if shape.element_count() == Some(1) {
        ShapeFact::Scalar
    } else {
        shape
    };
    let mut result = ValueFact::proven(kind, shape, StorageFact::Unknown);
    if left.residency != ResidencyFact::Host || right.residency != ResidencyFact::Host {
        result.residency = ResidencyFact::Unknown;
    }
    result
}
