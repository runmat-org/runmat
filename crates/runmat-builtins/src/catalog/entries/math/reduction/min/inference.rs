use super::super::support::{
    admitted_min_reduction_controls, inferred_min_shape, is_empty_placeholder,
};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact, ValueFact,
    ValueKindFact,
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
            Some(LiteralValue::Unknown) | None => MinCallFamily::Unresolved,
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
            // A pairwise call has no index output; the owning grammar still
            // needs to admit its operands/options before any class, shape, or
            // residency fact can be proven.
            MinCallFamily::Pairwise => vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
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
