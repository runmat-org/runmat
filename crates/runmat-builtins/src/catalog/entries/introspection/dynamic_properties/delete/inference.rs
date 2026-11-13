use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-DYNAMIC-DELETE-ARITY",
            "expected one metadata handle",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(receiver) = request.arguments.first() {
        if !matches!(
            &receiver.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Object(ObjectFact {
                    handle_semantics: None | Some(true),
                    ..
                })
        ) {
            diagnostics.push(argument_error(
                "RM-DYNAMIC-DELETE-TARGET",
                "expected dynamic-property metadata handle",
                0,
            ));
        }
    }
    let mut contract = CallContract::fixed(vec![]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
