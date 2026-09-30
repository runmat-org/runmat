use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, InferenceDiagnostic, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::fixed(vec![ValueFact::scalar(ValueKindFact::Logical)]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() != 1 {
        result.diagnostics.push(InferenceDiagnostic::error(
            "RM.MISSING.INVALID_ARGUMENT",
            "anymissing requires one input",
        ));
    }
    result
}
