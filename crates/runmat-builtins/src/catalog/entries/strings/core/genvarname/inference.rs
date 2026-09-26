use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

fn character_row() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}
fn invalid_text(input: &ValueFact, container: bool) -> bool {
    match &input.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Character => input.shape.known_dims().is_some_and(|dims| {
            dims.first().is_some_and(|d| d.is_some_and(|d| d > 1))
                || dims.iter().skip(2).any(|d| d.is_some_and(|d| d != 1))
        }),
        ValueKindFact::String => !container && input.shape.element_count().is_some_and(|n| n != 1),
        ValueKindFact::Cell(cell) if container => {
            cell.elements.iter().any(|v| invalid_text(v, false))
        }
        _ => true,
    }
}
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-GENVARNAME-ARITY",
            "genvarname requires one or two inputs",
            0,
        ));
    }
    for (index, input) in request.arguments.iter().enumerate() {
        if invalid_text(input, true) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GENVARNAME-TEXT",
                "names and exclusions must be text containers with scalar text elements",
                index,
            ));
        }
    }
    let output = match request.arguments.first() {
        Some(input) => match &input.kind {
            ValueKindFact::Character => character_row(),
            ValueKindFact::String => ValueFact::proven(
                ValueKindFact::String,
                input.shape.clone(),
                match input.storage {
                    StorageFact::Scalar => StorageFact::Scalar,
                    StorageFact::Dense => StorageFact::Dense,
                    _ => StorageFact::Unknown,
                },
            ),
            ValueKindFact::Cell(cell) => ValueFact::proven(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(character_row()),
                    elements: cell.elements.iter().map(|_| character_row()).collect(),
                    elements_complete: cell.elements_complete,
                }),
                input.shape.clone(),
                StorageFact::Dense,
            ),
            _ => ValueFact::unknown(DynamicReason::RuntimeValue),
        },
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn character_length_is_not_preserved_and_unknown_is_not_text() {
        for (input, expected) in [
            (
                ValueFact::proven(
                    ValueKindFact::Character,
                    ShapeFact::from(vec![Some(1), Some(4)]),
                    StorageFact::Dense,
                ),
                character_row(),
            ),
            (
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            ),
        ] {
            let request = CallRequest {
                arguments: vec![input],
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            };
            assert_eq!(
                infer(&request, &super::super::GENVARNAME_CATALOG_ENTRY).outputs[0],
                expected
            );
        }
    }
    #[test]
    fn preserves_nd_container_shape_but_replaces_element_width() {
        let input = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::scalar(ValueKindFact::String)),
                elements: vec![],
                elements_complete: false,
            }),
            ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
            StorageFact::Dense,
        );
        let request = CallRequest {
            arguments: vec![input.clone()],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::GENVARNAME_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].shape, input.shape);
        let ValueKindFact::Cell(cell) = &result.outputs[0].kind else {
            panic!("cell output")
        };
        assert_eq!(*cell.element, character_row());
    }
}
