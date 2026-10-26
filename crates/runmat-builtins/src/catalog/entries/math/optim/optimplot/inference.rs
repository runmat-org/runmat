use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = ValueFact::scalar(ValueKindFact::Logical);
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() != 3 {
        result.diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "optimization plot callbacks require x, optimValues and state",
        ));
    }
    if request.arguments.get(1).is_some_and(|value| {
        !matches!(
            value.kind,
            ValueKindFact::Struct(_) | ValueKindFact::Unknown
        ) || value.shape.element_count().is_some_and(|count| count != 1)
    }) {
        result.diagnostics.push(InferenceDiagnostic::error(
            "VALUES",
            "optimValues must be a scalar struct",
        ));
    }
    if request.arguments.get(2).is_some_and(|value| {
        !matches!(
            value.kind,
            ValueKindFact::Character | ValueKindFact::String | ValueKindFact::Unknown
        )
    }) {
        result.diagnostics.push(InferenceDiagnostic::error(
            "STATE",
            "state must be scalar text",
        ));
    }
    result
}
