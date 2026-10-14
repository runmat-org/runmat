use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-ONCLEANUP-ARITY",
            "onCleanup requires exactly one callback",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(callback) = request.arguments.first() {
        if !matches!(
            callback.kind,
            ValueKindFact::Unknown | ValueKindFact::Callable(_)
        ) {
            diagnostics.push(argument_error(
                "RM-ONCLEANUP-CALLBACK",
                "cleanupFun must be a function handle or closure",
                0,
            ));
        }
    }
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("onCleanup").owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    match request.arguments.first().map(|callback| &callback.kind) {
        Some(ValueKindFact::Callable(callable)) => {
            result
                .capabilities
                .0
                .extend(callable.capabilities.0.iter().copied());
            if let Some(CallableIdentity::Builtin(identity)) = &callable.identity {
                if let Some(callback_entry) = crate::builtin_catalog_entry_by_name(&identity.0) {
                    result
                        .effects
                        .0
                        .extend(callback_entry.contract.effect_set().0);
                    result
                        .capabilities
                        .0
                        .extend(callback_entry.contract.capability_set().0);
                } else {
                    result.effects.0.insert(EffectKind::Unknown);
                }
            } else {
                result.effects.0.insert(EffectKind::Unknown);
            }
        }
        Some(ValueKindFact::Unknown) => {
            result.effects.0.insert(EffectKind::Unknown);
        }
        _ => {}
    }
    if !diagnostics.is_empty() {
        result.outputs.iter_mut().for_each(|output| {
            *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        });
    }
    result.diagnostics.extend(diagnostics);
    result
}
