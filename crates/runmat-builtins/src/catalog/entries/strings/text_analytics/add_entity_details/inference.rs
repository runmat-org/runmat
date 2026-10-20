use super::super::scalar_text_control::known_invalid_scalar_text_shape;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ObjectFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if count == 0 || count.is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ADD-ENTITY-ARITY",
            "addEntityDetails requires tokenizedDocument and optional name-value pairs",
            count.min(1),
        ));
    }
    for index in (1..count).step_by(2) {
        if request.arguments.get(index).is_some_and(|value| {
            !matches!(
                &value.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
        }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ADD-ENTITY-OPTION-TEXT",
                "option names must be scalar text",
                index,
            ));
        }
        if request
            .arguments
            .get(index)
            .is_some_and(known_invalid_scalar_text_shape)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ADD-ENTITY-OPTION-TEXT",
                "option names must be scalar text",
                index,
            ));
        }
        if request
            .literals
            .literal_string_at(index)
            .is_some_and(|name| {
                !["RetokenizeMethod", "DiscardKnownValues", "Model"]
                    .iter()
                    .any(|option| name.eq_ignore_ascii_case(option))
            })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ADD-ENTITY-OPTION",
                "unsupported entity-detail option",
                index,
            ));
        }
    }
    let output = request.arguments.first().map_or_else(
        || ValueFact::unknown(DynamicReason::RuntimeValue),
        |input| match &input.kind {
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
                    "RM-CATALOG-ADD-ENTITY-DOCUMENT",
                    "addEntityDetails requires tokenizedDocument",
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
    fn document_plus_option_pairs_have_odd_arity() {
        for (count, valid) in [
            (0, false),
            (1, true),
            (2, false),
            (3, true),
            (4, false),
            (5, true),
        ] {
            let result = infer(
                &CallRequest {
                    arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); count],
                    literals: LiteralContext::default(),
                    outputs: OutputSelection::default(),
                },
                &super::super::ADD_ENTITY_DETAILS_CATALOG_ENTRY,
            );
            assert_eq!(
                result
                    .diagnostics
                    .iter()
                    .any(|d| d.code == "RM-CATALOG-ADD-ENTITY-ARITY"),
                !valid,
                "argument count {count}",
            );
        }
    }
}
