use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CellFact, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let converted: Vec<_> = request.arguments.iter().map(convert).collect();
    let mut contract = CallContract::fixed(converted);
    if let Some(requested) = request.outputs.requested.known_count() {
        contract.outputs.resize_with(requested, padded_zero);
        contract.maximum_outputs = Some(requested);
    }
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.is_empty() {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-CONVERT-STRINGS-ARITY",
            "convertStringsToChars requires at least one input",
            0,
        ));
    }
    result
}

fn convert(value: &ValueFact) -> ValueFact {
    match &value.kind {
        ValueKindFact::Unknown => return ValueFact::unknown(DynamicReason::RuntimeValue),
        ValueKindFact::String => {}
        _ => return value.clone(),
    }
    match value.shape.element_count() {
        Some(1) => ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![None, None]),
            StorageFact::Dense,
        ),
        Some(_) => ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(character_element()),
                elements: Vec::new(),
                elements_complete: false,
            }),
            value.shape.clone(),
            StorageFact::Dense,
        ),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

fn character_element() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![None, None]),
        StorageFact::Dense,
    )
}

fn padded_zero() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection, RequestedOutputCount, ResidencyFact};

    #[test]
    fn string_shapes_passthrough_and_padding() {
        let string_array = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        );
        let mut resident = padded_zero();
        resident.residency = ResidencyFact::Device { provider: None };
        let request = CallRequest {
            arguments: vec![string_array.clone(), resident.clone()],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
        };
        let result = infer(
            &request,
            &super::super::CONVERT_STRINGS_TO_CHARS_CATALOG_ENTRY,
        );
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Cell(_)));
        assert_eq!(result.outputs[0].shape, string_array.shape);
        assert_eq!(result.outputs[1], resident);
        assert_eq!(result.outputs[2], padded_zero());
        assert!(result.diagnostics.is_empty());
    }

    #[test]
    fn unknown_input_does_not_claim_passthrough_or_character_output() {
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(
            &request,
            &super::super::CONVERT_STRINGS_TO_CHARS_CATALOG_ENTRY,
        );
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
}
