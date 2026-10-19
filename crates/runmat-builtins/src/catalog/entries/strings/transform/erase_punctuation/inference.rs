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
    let count = request.arguments.len();
    if count != 1 && count != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ERASE-PUNCTUATION-ARITY",
            "erasePunctuation requires one input or a TokenTypes name-value pair",
            count.min(1),
        ));
    }
    if count == 3 {
        if !matches!(
            request.arguments.first().map(|value| &value.kind),
            Some(ValueKindFact::Object(_) | ValueKindFact::Unknown)
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ERASE-PUNCTUATION-OPTION",
                "TokenTypes applies only to tokenizedDocument",
                0,
            ));
        }
        if request
            .literals
            .literal_string_at(1)
            .is_some_and(|name| !name.eq_ignore_ascii_case("TokenTypes"))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ERASE-PUNCTUATION-OPTION",
                "erasePunctuation expects the TokenTypes option",
                1,
            ));
        }
        for index in 1..3 {
            if request.arguments.get(index).is_some_and(|value| {
                !matches!(
                    value.kind,
                    ValueKindFact::Unknown
                        | ValueKindFact::String
                        | ValueKindFact::Character
                        | ValueKindFact::Cell(_)
                )
            }) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ERASE-PUNCTUATION-OPTION",
                    "TokenTypes and its name must be text",
                    index,
                ));
            }
        }
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
                        "RM-CATALOG-ERASE-PUNCTUATION-CELL",
                        "erasePunctuation cell elements must be text",
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
                    "RM-CATALOG-ERASE-PUNCTUATION-TEXT",
                    "erasePunctuation requires host text or tokenizedDocument",
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
    use runmat_types::{LiteralContext, LiteralValue, OutputSelection};

    #[test]
    fn token_types_requires_document_and_correct_name() {
        let text = ValueFact::scalar(ValueKindFact::String);
        let request = CallRequest {
            arguments: vec![
                text,
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("bad".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::default(),
        };
        assert!(
            infer(&request, &super::super::ERASE_PUNCTUATION_CATALOG_ENTRY)
                .diagnostics
                .iter()
                .any(|d| d.code == "RM-CATALOG-ERASE-PUNCTUATION-OPTION")
        );
    }
}
