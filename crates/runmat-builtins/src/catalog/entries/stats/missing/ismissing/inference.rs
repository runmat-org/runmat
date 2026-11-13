use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, DynamicReason,
    InferenceDiagnostic, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn mask_fact(input: &ValueFact) -> ValueFact {
    match &input.kind {
        ValueKindFact::Numeric(_)
        | ValueKindFact::Logical
        | ValueKindFact::String
        | ValueKindFact::Cell(_) => {
            let storage = if input.storage == StorageFact::Scalar {
                StorageFact::Scalar
            } else {
                StorageFact::Dense
            };
            ValueFact::proven(ValueKindFact::Logical, input.shape.clone(), storage)
        }
        ValueKindFact::Character => {
            let shape = input.shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
                ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)])
            });
            ValueFact::proven(ValueKindFact::Logical, shape, StorageFact::Dense)
        }
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_some_and(|class| {
                class.is(standard::TABLE) || class.is(standard::TIMETABLE)
            }) =>
        {
            ValueFact::proven(
                ValueKindFact::Logical,
                ShapeFact::Unknown,
                StorageFact::Dense,
            )
        }
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_some_and(|class| {
                class.is(standard::DATETIME) || class.is(standard::DURATION)
            }) =>
        {
            ValueFact::proven(
                ValueKindFact::Logical,
                ShapeFact::Unknown,
                StorageFact::Dense,
            )
        }
        // Structures map recursively to structures of masks; arbitrary object
        // or unknown values may return scalar false instead. Neither is a
        // logical-array guarantee.
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
        mask_fact,
    );
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() != 1 {
        result.diagnostics.push(InferenceDiagnostic::error(
            "RM.MISSING.INVALID_ARGUMENT",
            "ismissing requires one input",
        ));
    }
    result
}
