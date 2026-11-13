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
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ERASE-ARITY",
            "erase requires text and a pattern",
            request.arguments.len().min(1),
        ));
    }
    if let Some(pattern) = request.arguments.get(1) {
        match &pattern.kind {
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown => {}
            ValueKindFact::Cell(cell) => {
                if pattern.shape.element_count() != Some(0)
                    && cell
                        .elements
                        .iter()
                        .chain(std::iter::once(cell.element.as_ref()))
                        .any(invalid_cell_element)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-ERASE-PATTERN",
                        "erase pattern cells must contain text scalars or character vectors",
                        1,
                    ));
                }
            }
            ValueKindFact::Object(object)
                if object
                    .runtime_class
                    .as_ref()
                    .is_none_or(|class| class == &runmat_types::standard::PATTERN.owned()) => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-ERASE-PATTERN",
                "erase patterns must be text or pattern objects",
                1,
            )),
        }
    }
    let output = match request.arguments.first() {
        Some(text) => match &text.kind {
            ValueKindFact::String => {
                ValueFact::proven(ValueKindFact::String, text.shape.clone(), text.storage)
            }
            ValueKindFact::Character => ValueFact::proven(
                ValueKindFact::Character,
                text.shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
                    ShapeFact::from(vec![dims.first().copied().flatten(), None])
                }),
                StorageFact::Dense,
            ),
            ValueKindFact::Cell(cell) => {
                if text.shape.element_count() != Some(0)
                    && cell
                        .elements
                        .iter()
                        .chain(std::iter::once(cell.element.as_ref()))
                        .any(invalid_cell_element)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-ERASE-TEXT",
                        "erase text cells must contain text scalars or character vectors",
                        0,
                    ));
                }
                ValueFact::proven(
                    ValueKindFact::Cell(CellFact {
                        element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                        elements: Vec::new(),
                        elements_complete: false,
                    }),
                    text.shape.clone(),
                    text.storage,
                )
            }
            ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ERASE-TEXT",
                    "erase requires text input",
                    0,
                ));
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
        },
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn invalid_cell_element(element: &ValueFact) -> bool {
    match &element.kind {
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

    fn call(text: ValueFact, pattern: ValueFact) -> CallInference {
        infer(
            &CallRequest {
                arguments: vec![text, pattern],
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            },
            &super::super::ERASE_CATALOG_ENTRY,
        )
    }

    #[test]
    fn string_shape_retained_and_char_width_dynamic() {
        let scalar = ValueFact::scalar(ValueKindFact::String);
        let result = call(scalar.clone(), scalar);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::String);
        let strings = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let pattern = ValueFact::scalar(ValueKindFact::String);
        assert_eq!(
            call(strings, pattern.clone()).outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        let chars = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(8)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(chars, pattern).outputs[0].shape,
            ShapeFact::from(vec![Some(2), None])
        );
    }

    #[test]
    fn invalid_pattern_and_numeric_text_diagnosed() {
        let numeric = ValueFact::scalar(ValueKindFact::Logical);
        let result = call(numeric.clone(), numeric);
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-ERASE-TEXT"));
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-ERASE-PATTERN"));
    }
}
