use crate::{
    catalog::inference::{argument_error, finish_fixed, possible_character_row},
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
            "RM-CATALOG-LOWER-ARITY",
            "lower requires one text input",
            request.arguments.len().min(1),
        ));
    }
    let output = request.arguments.first().map_or_else(
        || ValueFact::unknown(DynamicReason::RuntimeValue),
        |input| match &input.kind {
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
            ValueKindFact::Cell(cell) => {
                if input.shape.element_count() != Some(0)
                    && cell
                        .elements
                        .iter()
                        .chain(std::iter::once(cell.element.as_ref()))
                        .any(invalid_cell_element)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-LOWER-CELL",
                        "lower cell elements must be string scalars or character vectors",
                        0,
                    ));
                }
                ValueFact::proven(
                    ValueKindFact::Cell(CellFact {
                        element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                        elements: Vec::new(),
                        elements_complete: false,
                    }),
                    input.shape.clone(),
                    input.storage,
                )
            }
            ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LOWER-TEXT",
                    "lower requires string, character or cell text",
                    0,
                ));
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
        },
    );
    finish_fixed(entry, request, output, diagnostics)
}

fn invalid_cell_element(element: &ValueFact) -> bool {
    match element.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::String => element
            .shape
            .element_count()
            .is_some_and(|count| count != 1),
        ValueKindFact::Character => !possible_character_row(&element.shape),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};

    fn call(arguments: Vec<ValueFact>) -> CallInference {
        infer(
            &CallRequest {
                arguments,
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            },
            &super::super::LOWER_CATALOG_ENTRY,
        )
    }
    #[test]
    fn string_shape_and_char_rows() {
        let scalar = call(vec![ValueFact::scalar(ValueKindFact::String)]);
        assert!(scalar.diagnostics.is_empty());
        assert_eq!(scalar.outputs[0].kind, ValueKindFact::String);
        let strings = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(vec![strings]).outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        let chars = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(4)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(vec![chars]).outputs[0].shape,
            ShapeFact::from(vec![Some(2), None])
        );
    }
    #[test]
    fn invalid_numeric_input_and_arity_are_diagnosed() {
        assert!(call(vec![ValueFact::scalar(ValueKindFact::Logical)])
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-LOWER-TEXT"));
        assert!(call(Vec::new())
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-LOWER-ARITY"));
    }
    #[test]
    fn empty_cell_does_not_invent_an_invalid_element() {
        let empty = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::never()),
                elements: Vec::new(),
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(0), Some(0)]),
            StorageFact::Dense,
        );
        assert!(call(vec![empty]).diagnostics.is_empty());
    }
}
