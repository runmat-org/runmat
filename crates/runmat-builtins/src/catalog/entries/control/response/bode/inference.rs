use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, CapabilityRequirement,
    EffectKind, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BODE-ARITY",
            "bode expects a model and optional frequency vector",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let count = request.outputs.requested.known_count();
    if count.is_some_and(|n| n > 3) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BODE-OUTPUTS",
            "bode supports at most three outputs",
            0,
        ));
    }
    if let Some(model) = request.arguments.first() {
        let valid = match &model.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => {
                object
                    .runtime_class
                    .as_ref()
                    .is_some_and(|class| class == &standard::TRANSFER_FUNCTION.owned())
                    || (object.runtime_class.is_none() && object.class.is_none())
            }
            _ => false,
        };
        if !valid
            || model.shape.element_count().is_some_and(|n| n != 1)
            || matches!(model.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-BODE-MODEL",
                "sys must be a scalar SISO tf model",
                0,
            ));
        }
    }
    if let Some(w) = request.arguments.get(1) {
        if w.shape
            .known_dims()
            .is_some_and(|dims| dims.iter().filter(|dim| dim.is_some_and(|n| n > 1)).count() > 1)
            || w.shape.element_count() == Some(0)
            || matches!(w.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-BODE-W",
                "w must be a nonempty real host vector",
                1,
            ));
        }
    }
    let samples = request
        .arguments
        .get(1)
        .and_then(|w| w.shape.element_count())
        .or_else(|| (request.arguments.len() == 1).then_some(200));
    let array = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(1), samples]),
        StorageFact::Dense,
    );
    let frequency = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![samples, Some(1)]),
        StorageFact::Dense,
    );
    let mut contract = CallContract::fixed(if count == Some(0) {
        vec![]
    } else {
        vec![array.clone(), array, frequency]
    });
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if count.is_some_and(|n| n > 0) {
        contract.effects.0.remove(&EffectKind::UserInterface);
        contract
            .capabilities
            .0
            .remove(&CapabilityRequirement::UserInterface);
    }
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
