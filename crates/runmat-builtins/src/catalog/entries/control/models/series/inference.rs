use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, DynamicReason, ObjectFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 2..=4) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SERIES-ARITY",
            "series expects two models, optionally with 'name' or two channel lists",
            0,
        ));
    }
    if request.arguments.len() == 3 {
        if let Some(selector) = request.literals.literal_string_at(2) {
            if !selector.eq_ignore_ascii_case("name") {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SERIES-SELECTOR",
                    "series selector must be 'name'",
                    2,
                ));
            }
        }
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SERIES-OUTPUTS",
            "series returns at most one model",
            0,
        ));
    }
    let mut saw_ss = false;
    let mut both_tf = request.arguments.len() == 2;
    for (index, value) in request.arguments.iter().take(2).enumerate() {
        match &value.kind {
            ValueKindFact::Unknown => both_tf = false,
            ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
                Some(class) if class == &standard::STATE_SPACE.owned() => {
                    saw_ss = true;
                    both_tf = false;
                }
                Some(class) if class == &standard::TRANSFER_FUNCTION.owned() => {}
                None if object.class.is_none() => both_tf = false,
                _ => diagnostics.push(argument_error(
                    "RM-CATALOG-SERIES-MODEL",
                    "series requires tf or ss model objects",
                    index,
                )),
            },
            ValueKindFact::Numeric(_) if value.shape.element_count() == Some(1) => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-SERIES-MODEL",
                "series requires tf or ss model objects or scalar gains",
                index,
            )),
        }
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    if saw_ss || both_tf {
        let output_shape = match (request.arguments.first(), request.arguments.get(1)) {
            (Some(first), Some(second))
                if first.shape.element_count() == Some(1)
                    && second.shape.element_count() == Some(1) =>
            {
                ShapeFact::Scalar
            }
            (Some(first), Some(second))
                if first.shape.is_definitely_nonscalar()
                    && (second.shape.element_count() == Some(1) || first.shape == second.shape) =>
            {
                first.shape.clone()
            }
            (Some(first), Some(second))
                if second.shape.is_definitely_nonscalar()
                    && first.shape.element_count() == Some(1) =>
            {
                second.shape.clone()
            }
            _ => ShapeFact::Unknown,
        };
        output = ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(if saw_ss {
                    standard::STATE_SPACE.owned()
                } else {
                    standard::TRANSFER_FUNCTION.owned()
                }),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            output_shape,
            StorageFact::Opaque,
        );
    }
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
