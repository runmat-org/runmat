use super::super::scalar_text_control::known_invalid_scalar_text_shape;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 0 | 2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-STOPWORDS-ARITY",
            "stopWords accepts no inputs or a Language name-value pair",
            request.arguments.len().min(2),
        ));
    }
    if request.arguments.len() == 2
        && request
            .literals
            .literal_string_at(0)
            .is_some_and(|name| !name.eq_ignore_ascii_case("Language"))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-STOPWORDS-OPTION",
            "stopWords expects the Language option",
            0,
        ));
    }
    if request.arguments.len() == 2
        && request
            .literals
            .literal_string_at(1)
            .is_some_and(|language| {
                !["en", "ja", "de", "ko"]
                    .iter()
                    .any(|accepted| language.eq_ignore_ascii_case(accepted))
            })
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-STOPWORDS-LANGUAGE",
            "stopWords language must be en, ja, de, or ko",
            1,
        ));
    }
    for (index, value) in request.arguments.iter().take(2).enumerate() {
        match &value.kind {
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-STOPWORDS-TEXT",
                "stopWords option name and language must be scalar text",
                index,
            )),
        }
        if known_invalid_scalar_text_shape(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STOPWORDS-SCALAR",
                "stopWords requires scalar text controls",
                index,
            ));
        }
    }
    let output = if diagnostics.is_empty() {
        ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![None, Some(1)]),
            StorageFact::Dense,
        )
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish_fixed(entry, request, output, diagnostics)
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
            &super::super::STOP_WORDS_CATALOG_ENTRY,
        )
    }

    #[test]
    fn column_result_and_known_invalid_control() {
        let result = call(vec![]);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![None, Some(1)])
        );
        let invalid = call(vec![
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::String),
        ]);
        assert!(invalid
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-STOPWORDS-TEXT"));
        let multirow_character = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Dense,
        );
        let invalid = call(vec![
            multirow_character,
            ValueFact::scalar(ValueKindFact::String),
        ]);
        assert!(invalid
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-STOPWORDS-SCALAR"));
    }
}
