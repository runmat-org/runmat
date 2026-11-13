use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DEBLANK-ARITY",
            "deblank requires exactly one text input",
            request.arguments.len().min(1),
        ));
    }
    let output = match request.arguments.first() {
        Some(input) => match &input.kind {
            ValueKindFact::String => {
                ValueFact::proven(ValueKindFact::String, input.shape.clone(), input.storage)
            }
            ValueKindFact::Character => ValueFact::proven(
                ValueKindFact::Character,
                input.shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
                    ShapeFact::from(vec![dims.first().copied().flatten(), None])
                }),
                StorageFact::Dense,
            ),
            ValueKindFact::Cell(_) => ValueFact::proven(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                    elements: Vec::new(),
                    elements_complete: false,
                }),
                input.shape.clone(),
                input.storage,
            ),
            ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-DEBLANK-TEXT",
                    "deblank requires host text",
                    0,
                ));
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
        },
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};

    fn call(argument: ValueFact) -> CallInference {
        infer(
            &CallRequest {
                arguments: vec![argument],
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            },
            &super::super::DEBLANK_CATALOG_ENTRY,
        )
    }

    #[test]
    fn string_shape_preserved_character_width_unknown() {
        let string = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(string).outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        let chars = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(8)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(chars).outputs[0].shape,
            ShapeFact::from(vec![Some(2), None])
        );
    }

    #[test]
    fn numeric_input_rejects() {
        assert!(call(ValueFact::scalar(ValueKindFact::Logical))
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-DEBLANK-TEXT"));
    }
}
