use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, DynamicReason,
    InferenceDiagnostic, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn result_fact(input: &ValueFact) -> ValueFact {
    match &input.kind {
        ValueKindFact::Numeric(_) | ValueKindFact::String | ValueKindFact::Logical => {
            let mut output = input.clone();
            output.residency = runmat_types::ResidencyFact::Host;
            output.alias = runmat_types::AliasFact::Unique;
            output
        }
        ValueKindFact::Character => {
            let shape = input.shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
                ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)])
            });
            ValueFact::proven(ValueKindFact::String, shape, StorageFact::Dense)
        }
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_some_and(|class| {
                class.is(standard::TABLE) || class.is(standard::TIMETABLE)
            }) =>
        {
            let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
            output.kind = ValueKindFact::Object(runmat_types::ObjectFact {
                class: object.class,
                runtime_class: object.runtime_class.clone(),
                properties: Default::default(),
                properties_complete: false,
                handle_semantics: object.handle_semantics,
            });
            output.residency = runmat_types::ResidencyFact::Host;
            output
        }
        _ => {
            let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
            output.residency = runmat_types::ResidencyFact::Host;
            output
        }
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = request.arguments.first().map_or_else(
        || ValueFact::unknown(DynamicReason::RuntimeValue),
        result_fact,
    );
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() != 2 {
        result.diagnostics.push(InferenceDiagnostic::error(
            "RM.MISSING.INVALID_ARGUMENT",
            "standardizeMissing requires data and indicators",
        ));
    }
    result
}
