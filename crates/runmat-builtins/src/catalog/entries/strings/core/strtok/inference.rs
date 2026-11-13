use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn infer_outputs(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    outputs: Vec<ValueFact>,
    diagnostics: Vec<runmat_types::InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRTOK-ARITY",
            "strtok accepts text and optional delimiters",
            0,
        ));
    }
    let primary = match request.arguments.first() {
        Some(input) if matches!(input.kind, ValueKindFact::String | ValueKindFact::Cell(_)) => {
            ValueFact::proven(input.kind.clone(), input.shape.clone(), StorageFact::Dense)
        }
        Some(input) if matches!(input.kind, ValueKindFact::Character) => ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::Unknown,
            StorageFact::Dense,
        ),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    infer_outputs(request, entry, vec![primary.clone(), primary], diagnostics)
}
