use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

fn result_fact(input: &ValueFact) -> ValueFact {
    match &input.kind {
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::String => {
            let mut output = input.clone();
            output.residency = ResidencyFact::Host;
            output.alias = runmat_types::AliasFact::Unique;
            output
        }
        // Tables can contain variables with different classes and masks.
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

fn mask_fact(input: &ValueFact) -> ValueFact {
    match &input.kind {
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::String => {
            let storage = if input.storage == StorageFact::Scalar {
                StorageFact::Scalar
            } else {
                StorageFact::Dense
            };
            ValueFact::proven(ValueKindFact::Logical, input.shape.clone(), storage)
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let (result, mask) = request.arguments.first().map_or_else(
        || {
            (
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            )
        },
        |input| (result_fact(input), mask_fact(input)),
    );
    let mut contract = CallContract::fixed(vec![result, mask]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    if request.arguments.len() < 2 {
        inferred.diagnostics.push(InferenceDiagnostic::error(
            "RM.MISSING.INVALID_ARGUMENT",
            "fillmissing requires data and a fill method",
        ));
    }
    inferred
}
