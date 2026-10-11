use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-ADDLISTENER-ARITY",
            "expected target, event name and callback",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(target) = request.arguments.first() {
        if !matches!(
            &target.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Object(ObjectFact {
                    handle_semantics: None | Some(true),
                    ..
                })
        ) {
            diagnostics.push(argument_error(
                "RM-ADDLISTENER-TARGET",
                "target must be a handle object",
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
                "RM-ADDLISTENER-EVENT",
                "event name must be scalar text",
                1,
            ));
        }
    }
    if let Some(callback) = request.arguments.get(2) {
        if !matches!(
            callback.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Callable(_)
                | ValueKindFact::String
                | ValueKindFact::Character
        ) {
            diagnostics.push(argument_error(
                "RM-ADDLISTENER-CALLBACK",
                "callback must be callable or scalar text",
                2,
            ));
        }
    }
    let mut contract = CallContract::fixed(vec![ValueFact::unknown(DynamicReason::RuntimeValue)]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
