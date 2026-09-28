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
            "RM-CATALOG-CONVERT-CONTAINED-ARITY",
            "convertContainedStringsToChars requires at least one input",
            0,
        ));
    }
    result
}

fn convert(value: &ValueFact) -> ValueFact {
    match &value.kind {
        ValueKindFact::String => convert_string(value),
        ValueKindFact::Cell(cell) => {
            let mut converted = value.clone();
            converted.kind = ValueKindFact::Cell(CellFact {
                element: Box::new(convert(&cell.element)),
                elements: cell.elements.iter().map(convert).collect(),
                elements_complete: cell.elements_complete,
            });
            converted
        }
        ValueKindFact::Struct(structure) => {
            let mut converted = value.clone();
            let mut fields = structure.clone();
            fields
                .fields
                .values_mut()
                .for_each(|member| *member = convert(member));
            for element in &mut fields.elements {
                element
                    .values_mut()
                    .for_each(|member| *member = convert(member));
            }
            converted.kind = ValueKindFact::Struct(fields);
            converted
        }
        ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
        _ => value.clone(),
    }
}

fn convert_string(value: &ValueFact) -> ValueFact {
    match value.shape.element_count() {
        Some(1) => ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![None, None]),
            StorageFact::Dense,
        ),
        Some(_) => ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::proven(
                    ValueKindFact::Character,
                    ShapeFact::from(vec![None, None]),
                    StorageFact::Dense,
                )),
                elements: Vec::new(),
                elements_complete: false,
            }),
            value.shape.clone(),
            StorageFact::Dense,
        ),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
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
    use runmat_types::{LiteralContext, OutputSelection, RequestedOutputCount, StructFact};
    use std::collections::BTreeMap;

    #[test]
    fn nested_struct_string_converts_but_resident_member_fact_is_preserved() {
        let string = ValueFact::scalar(ValueKindFact::String);
        let mut resident = padded_zero();
        resident.residency = runmat_types::ResidencyFact::Device { provider: None };
        let fields = BTreeMap::from([
            ("label".into(), string),
            ("payload".into(), resident.clone()),
        ]);
        let input = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
        let result = infer(
            &CallRequest {
                arguments: vec![input],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            &super::super::CONVERT_CONTAINED_STRINGS_TO_CHARS_CATALOG_ENTRY,
        );
        let ValueKindFact::Struct(output) = &result.outputs[0].kind else {
            panic!("struct fact")
        };
        assert_eq!(output.fields["label"].kind, ValueKindFact::Character);
        assert_eq!(output.fields["payload"], resident);
    }

    #[test]
    fn unknown_value_stays_dynamic_and_extra_slot_is_padded() {
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        };
        let result = infer(
            &request,
            &super::super::CONVERT_CONTAINED_STRINGS_TO_CHARS_CATALOG_ENTRY,
        );
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(result.outputs[1], padded_zero());
    }
}
