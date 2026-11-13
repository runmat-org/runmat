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
    let arguments = &request.arguments;
    if !(1..=3).contains(&arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRIP-ARITY",
            "strip requires text and at most direction and strip-character controls",
            arguments.len().min(3),
        ));
    }
    let output = arguments.first().map_or_else(
        || ValueFact::unknown(DynamicReason::RuntimeValue),
        |input| match &input.kind {
            ValueKindFact::String => {
                ValueFact::proven(ValueKindFact::String, input.shape.clone(), input.storage)
            }
            ValueKindFact::Character => ValueFact::proven(
                ValueKindFact::Character,
                input
                    .shape
                    .known_dims()
                    .map_or(ShapeFact::Unknown, |dims| match dims.as_slice() {
                        [] => input.shape.clone(),
                        [_] => ShapeFact::from(vec![None]),
                        [rows, _] => ShapeFact::from(vec![*rows, None]),
                        _ => ShapeFact::Unknown,
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
                        "RM-CATALOG-STRIP-CELL",
                        "strip cell elements must be scalar text or character vectors",
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
                    "RM-CATALOG-STRIP-TEXT",
                    "strip requires string, character, or cell text",
                    0,
                ));
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
        },
    );
    for (index, control) in arguments.iter().enumerate().skip(1).take(2) {
        let known_bad = match &control.kind {
            ValueKindFact::String | ValueKindFact::Cell(_) | ValueKindFact::Unknown => false,
            ValueKindFact::Character => !possible_character_row(&control.shape),
            _ => true,
        };
        if known_bad {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRIP-CONTROL",
                "strip controls must be textual direction or strip characters",
                index,
            ));
        }
    }
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
            &super::super::STRIP_CATALOG_ENTRY,
        )
    }

    #[test]
    fn known_text_shape_and_invalid_controls() {
        let text = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let direction = ValueFact::scalar(ValueKindFact::String);
        assert_eq!(
            call(vec![text.clone(), direction]).outputs[0].shape,
            text.shape
        );
        let bad = call(vec![text, ValueFact::scalar(ValueKindFact::Logical)]);
        assert!(bad
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-STRIP-CONTROL"));
        assert!(call(Vec::new())
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-STRIP-ARITY"));

        let character = |dims| {
            ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(dims),
                StorageFact::Dense,
            )
        };
        assert_eq!(
            call(vec![character(vec![Some(5)])]).outputs[0].shape,
            ShapeFact::from(vec![None]),
        );
        assert_eq!(
            call(vec![character(vec![Some(2), Some(5)])]).outputs[0].shape,
            ShapeFact::from(vec![Some(2), None]),
        );
        assert_eq!(
            call(vec![character(vec![Some(2), Some(5), Some(3)])]).outputs[0].shape,
            ShapeFact::Unknown,
        );
    }
}
