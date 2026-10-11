use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-FINDPROP-ARITY",
            "expected object and property name",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(receiver) = request.arguments.first() {
        if !matches!(
            receiver.kind,
            ValueKindFact::Unknown | ValueKindFact::Object(_)
        ) {
            diagnostics.push(argument_error(
                "RM-FINDPROP-TARGET",
                "expected object or handle object",
                0,
            ));
        }
    }
    if let Some(name) = request.arguments.get(1) {
        if !matches!(
            name.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) || name.shape.is_definitely_nonscalar()
        {
            diagnostics.push(argument_error(
                "RM-FINDPROP-NAME",
                "expected scalar text property name",
                1,
            ));
        }
    }
    // Class declarations and per-instance mutations can change the selected
    // metadata class or return []. Object field facts are not metadata facts.
    let mut contract = CallContract::fixed(vec![ValueFact::unknown(DynamicReason::RuntimeValue)]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
