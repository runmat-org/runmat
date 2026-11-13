use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-NOTIFY-ARITY",
            "expected target and event name",
            request.arguments.len(),
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
                "RM-NOTIFY-TARGET",
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
                "RM-NOTIFY-EVENT",
                "event name must be scalar text",
                1,
            ));
        }
    }
    let status = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut contract = CallContract::fixed(vec![status]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    // Callback registrations are live mutable state. Even a known event name
    // cannot establish the current callback identity or effects statically.
    result.effects.0.insert(EffectKind::Unknown);
    result
}
