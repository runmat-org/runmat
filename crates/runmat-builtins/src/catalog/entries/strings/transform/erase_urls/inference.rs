use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, ObjectFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ERASE-URLS-ARITY",
            "eraseURLs requires exactly one input",
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
                        .any(|value| {
                            !matches!(
                                value.kind,
                                ValueKindFact::Unknown
                                    | ValueKindFact::String
                                    | ValueKindFact::Character
                                    | ValueKindFact::Cell(_)
                            )
                        })
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-ERASE-URLS-CELL",
                        "eraseURLs cell elements must be text",
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
            ValueKindFact::Object(object)
                if object.runtime_class.as_ref()
                    == Some(&runmat_types::standard::TOKENIZED_DOCUMENT.owned()) =>
            {
                ValueFact::proven(
                    ValueKindFact::Object(ObjectFact {
                        class: object.class,
                        runtime_class: object.runtime_class.clone(),
                        properties: Default::default(),
                        properties_complete: false,
                        handle_semantics: Some(false),
                    }),
                    input.shape.clone(),
                    StorageFact::Opaque,
                )
            }
            ValueKindFact::Object(object) if object.runtime_class.is_none() => {
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
            ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ERASE-URLS-TEXT",
                    "eraseURLs requires host text or tokenizedDocument",
                    0,
                ));
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
        },
    );
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};

    #[test]
    fn preserves_string_shape_and_rejects_numeric() {
        let input = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let mut request = CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::ERASE_URLS_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        request.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
        assert!(infer(&request, &super::super::ERASE_URLS_CATALOG_ENTRY)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-ERASE-URLS-TEXT"));
    }
}
