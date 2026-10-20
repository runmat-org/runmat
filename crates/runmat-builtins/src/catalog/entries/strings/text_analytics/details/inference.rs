use super::super::scalar_text_control::known_invalid_scalar_text_shape;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ObjectFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn document_input(
    request: &CallRequest,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
    code: &'static str,
) -> ValueFact {
    request.arguments.first().map_or_else(
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
                    code,
                    "expected a tokenizedDocument object",
                    0,
                ));
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
        },
    )
}

fn options(
    request: &CallRequest,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
    code: &'static str,
    names: &[&str],
) {
    let count = request.arguments.len();
    if count == 0 || count.is_multiple_of(2) {
        diagnostics.push(argument_error(
            code,
            "expected tokenizedDocument followed by name-value pairs",
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
                code,
                "option name must be scalar text",
                index,
            ));
        }
        if request
            .arguments
            .get(index)
            .is_some_and(known_invalid_scalar_text_shape)
        {
            diagnostics.push(argument_error(
                code,
                "option name must be scalar text",
                index,
            ));
        }
        if request
            .literals
            .literal_string_at(index)
            .is_some_and(|name| {
                !names
                    .iter()
                    .any(|accepted| name.eq_ignore_ascii_case(accepted))
            })
        {
            diagnostics.push(argument_error(code, "unsupported detail option", index));
        }
    }
}

pub(super) fn token_details(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-TOKEN-DETAILS-ARITY",
            "tokenDetails requires one tokenizedDocument",
            request.arguments.len().min(1),
        ));
    }
    let document = document_input(
        request,
        &mut diagnostics,
        "RM-CATALOG-TOKEN-DETAILS-DOCUMENT",
    );
    let output = match document.kind {
        ValueKindFact::Object(_) => ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(runmat_types::standard::TABLE.owned()),
                properties: Default::default(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Unknown,
            StorageFact::Opaque,
        ),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn add_type_details(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    options(
        request,
        &mut diagnostics,
        "RM-CATALOG-ADD-TYPE-OPTION",
        &["DiscardKnownValues", "TopLevelDomains"],
    );
    let output = document_input(request, &mut diagnostics, "RM-CATALOG-ADD-TYPE-DOCUMENT");
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn add_sentence_details(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    options(
        request,
        &mut diagnostics,
        "RM-CATALOG-ADD-SENTENCE-OPTION",
        &["Abbreviations", "Starters", "DiscardKnownValues"],
    );
    let output = document_input(
        request,
        &mut diagnostics,
        "RM-CATALOG-ADD-SENTENCE-DOCUMENT",
    );
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};

    #[test]
    fn detail_mutators_require_document_then_option_pairs() {
        for (count, valid) in [
            (0, false),
            (1, true),
            (2, false),
            (3, true),
            (4, false),
            (5, true),
        ] {
            let request = CallRequest {
                arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); count],
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            };
            for (entry, infer) in [
                (
                    &super::super::ADD_TYPE_DETAILS_CATALOG_ENTRY,
                    add_type_details as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
                ),
                (
                    &super::super::ADD_SENTENCE_DETAILS_CATALOG_ENTRY,
                    add_sentence_details as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
                ),
            ] {
                let result = infer(&request, entry);
                assert_eq!(
                    result
                        .diagnostics
                        .iter()
                        .any(|d| d.code == "RM-CATALOG-ADD-TYPE-OPTION"
                            || d.code == "RM-CATALOG-ADD-SENTENCE-OPTION"),
                    !valid,
                    "{} argument count {count}",
                    entry.identity.name,
                );
            }
        }
    }
}
