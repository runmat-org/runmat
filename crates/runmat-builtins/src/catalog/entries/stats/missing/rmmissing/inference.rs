use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    NumericClass, ShapeFact, ValueFact, ValueKindFact,
};

fn cannot_contain_standard_missing(input: &ValueFact) -> bool {
    matches!(
        input.numeric().map(|numeric| numeric.class),
        Some(
            NumericClass::Int8
                | NumericClass::Int16
                | NumericClass::Int32
                | NumericClass::Int64
                | NumericClass::UInt8
                | NumericClass::UInt16
                | NumericClass::UInt32
                | NumericClass::UInt64
        )
    )
}

fn result_fact(input: &ValueFact) -> ValueFact {
    match &input.kind {
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::String => {
            let mut output = input.clone();
            if !cannot_contain_standard_missing(input) {
                output.shape = input
                    .shape
                    .rank()
                    .map_or(ShapeFact::Unknown, |rank| ShapeFact::from(vec![None; rank]));
                output.certainty =
                    runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
            }
            // An admitted resident input may be restored to its owning provider,
            // while an input that cannot be uploaded remains on the host.
            output.residency = runmat_types::ResidencyFact::Unknown;
            output.alias = runmat_types::AliasFact::Unique;
            output
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let result = request.arguments.first().map_or_else(
        || ValueFact::unknown(DynamicReason::RuntimeValue),
        result_fact,
    );
    let mask = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::Unknown,
        runmat_types::StorageFact::Dense,
    );
    let mut contract = CallContract::fixed(vec![result, mask]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    if request.arguments.is_empty() {
        inferred.diagnostics.push(InferenceDiagnostic::error(
            "RM.MISSING.INVALID_ARGUMENT",
            "rmmissing requires an input",
        ));
    }
    inferred
}
