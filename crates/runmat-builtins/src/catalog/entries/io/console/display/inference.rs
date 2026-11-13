use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest, EffectKind, ValueKindFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = runmat_types::CallContract::fixed(vec![]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = runmat_types::infer_call(&contract, request);
    if request.arguments.len() != 1 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-DISPLAY-ARITY",
            "display requires one input",
            0,
        ));
    }
    if request.arguments.first().is_some_and(|value| {
        matches!(
            value.kind,
            ValueKindFact::Numeric(_)
                | ValueKindFact::Logical
                | ValueKindFact::Character
                | ValueKindFact::String
                | ValueKindFact::Symbolic
        )
    }) {
        result.effects.0.remove(&EffectKind::HostCallback);
        result.effects.0.remove(&EffectKind::Unknown);
    }
    result
}
