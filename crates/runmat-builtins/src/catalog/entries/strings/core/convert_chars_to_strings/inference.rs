use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, StorageFact, ValueFact, ValueKindFact,
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
            "RM-CATALOG-CONVERT-CHARS-ARITY",
            "convertCharsToStrings requires at least one input",
            0,
        ));
    }
    result
}

fn convert(value: &ValueFact) -> ValueFact {
    match &value.kind {
        ValueKindFact::Character => ValueFact::scalar(ValueKindFact::String),
        ValueKindFact::Cell(cell) => {
            if cell.elements_complete {
                if cell.elements.iter().all(proven_character_vector) {
                    return ValueFact::proven(
                        ValueKindFact::String,
                        value.shape.clone(),
                        StorageFact::Dense,
                    );
                }
                if cell.elements.iter().any(proven_non_character_vector) {
                    return value.clone();
                }
            } else if proven_non_character_vector(&cell.element) {
                return value.clone();
            }
            let mut uncertain = ValueFact::unknown(DynamicReason::RuntimeValue);
            // Both possible results (unchanged cell or converted string array)
            // retain the container's shape.
            uncertain.shape = value.shape.clone();
            uncertain.storage = StorageFact::Dense;
            uncertain
        }
        ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
        _ => value.clone(),
    }
}

fn proven_character_vector(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Character)
        && value
            .shape
            .known_dims()
            .and_then(|dims| dims.first().copied().flatten())
            .is_some_and(|rows| rows <= 1)
}

fn proven_non_character_vector(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Character => value
            .shape
            .known_dims()
            .and_then(|dims| dims.first().copied().flatten())
            .is_some_and(|rows| rows > 1),
        _ => true,
    }
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
    use runmat_types::{
        CellFact, LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact,
    };

    fn call(value: ValueFact) -> CallInference {
        infer(
            &CallRequest {
                arguments: vec![value],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            &super::super::CONVERT_CHARS_TO_STRINGS_CATALOG_ENTRY,
        )
    }

    #[test]
    fn character_and_known_cellstr_convert_while_mixed_cells_pass_through() {
        let chars = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(3)]),
            StorageFact::Dense,
        );
        assert_eq!(call(chars.clone()).outputs[0].kind, ValueKindFact::String);
        let known_cell = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(chars.clone()),
                elements: vec![chars.clone(), chars],
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        let converted = call(known_cell.clone());
        assert_eq!(converted.outputs[0].kind, ValueKindFact::String);
        assert_eq!(converted.outputs[0].shape, known_cell.shape);
        let ValueKindFact::Cell(mut cell) = known_cell.kind.clone() else {
            unreachable!()
        };
        cell.elements[1] = padded_zero();
        let mixed = ValueFact {
            kind: ValueKindFact::Cell(cell),
            ..known_cell
        };
        assert_eq!(call(mixed.clone()).outputs[0], mixed);
    }

    #[test]
    fn requested_extra_output_uses_current_zero_padding() {
        let input = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        let result = infer(
            &CallRequest {
                arguments: vec![input],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
            },
            &super::super::CONVERT_CHARS_TO_STRINGS_CATALOG_ENTRY,
        );
        assert_eq!(result.outputs[0].kind, ValueKindFact::String);
        assert_eq!(result.outputs[1], padded_zero());
    }
}
