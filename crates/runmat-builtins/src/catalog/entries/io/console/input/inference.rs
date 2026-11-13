use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let raw_text = known_raw_text_mode(request);
    let output = if raw_text {
        let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
        fact.kind = ValueKindFact::Character;
        fact.shape = ShapeFact::Ranked { rank: 2 };
        fact.storage = StorageFact::Dense;
        fact.residency = ResidencyFact::Host;
        fact
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    let mut contract = runmat_types::CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = runmat_types::infer_call(&contract, request);
    if request.arguments.len() > 2 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-INPUT-ARITY",
            "input accepts at most two controls",
            2,
        ));
    }
    for (index, value) in request.arguments.iter().enumerate() {
        if matches!(value.residency, ResidencyFact::Device { .. }) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-INPUT-RESIDENT",
                "input prompt and flag cannot be provider-resident",
                index,
            ));
        }
    }
    let prompt_index = if request.arguments.len() == 2
        && request.literals.literal_string_at(0).as_deref() == Some("s")
        && request.literals.literal_string_at(1).as_deref() != Some("s")
    {
        1
    } else {
        0
    };
    if let Some(prompt) = request.arguments.get(prompt_index) {
        if definitely_nontext(prompt) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-INPUT-PROMPT",
                "input prompt must be a text scalar",
                prompt_index,
            ));
        }
    }
    if request.arguments.len() == 2 {
        if let (Some(first), Some(second)) = (
            request.literals.literal_string_at(0),
            request.literals.literal_string_at(1),
        ) {
            if super::InputStringMode::decode(&first).is_none()
                && super::InputStringMode::decode(&second).is_none()
            {
                result.diagnostics.push(argument_error(
                    "RM-CATALOG-INPUT-FLAG",
                    "input string flag must be 's'",
                    1,
                ));
            }
        }
    }
    result
}

fn known_raw_text_mode(request: &CallRequest) -> bool {
    if request.arguments.len() != 2 {
        return false;
    }
    request.literals.literal_string_at(1).as_deref() == Some("s")
        || (request.literals.literal_string_at(0).as_deref() == Some("s")
            && request.literals.literal_string_at(1).is_some())
}

fn definitely_nontext(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .copied()
                .flatten()
                .is_some_and(|rows| rows != 1)
                || dims
                    .iter()
                    .skip(2)
                    .any(|extent| extent.is_some_and(|n| n > 1))
        }),
        ValueKindFact::String => value.shape.element_count().is_some_and(|count| count != 1),
        _ => true,
    }
}
