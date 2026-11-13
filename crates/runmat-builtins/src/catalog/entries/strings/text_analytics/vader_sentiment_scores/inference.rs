use super::super::scalar_text_control::known_invalid_scalar_text_shape;
use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if count == 0 || count.is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-VADER-ARITY",
            "vaderSentimentScores requires documents and name-value pairs",
            count.min(1),
        ));
    }
    if let Some(document) = request.arguments.first() {
        match &document.kind {
            ValueKindFact::Object(object)
                if object.runtime_class.as_ref()
                    == Some(&runmat_types::standard::TOKENIZED_DOCUMENT.owned()) => {}
            ValueKindFact::Object(object) if object.runtime_class.is_none() => {}
            ValueKindFact::Unknown => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-VADER-DOCUMENT",
                "vaderSentimentScores requires tokenizedDocument",
                0,
            )),
        }
    }
    for index in (1..count).step_by(2) {
        if request.arguments.get(index).is_some_and(|value| {
            !matches!(
                &value.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
        }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-VADER-OPTION-TEXT",
                "option name must be text",
                index,
            ));
        }
        if request
            .arguments
            .get(index)
            .is_some_and(known_invalid_scalar_text_shape)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-VADER-OPTION-TEXT",
                "option name must be scalar text",
                index,
            ));
        }
        if request
            .literals
            .literal_string_at(index)
            .is_some_and(|name| {
                !["SentimentLexicon", "Boosters", "Dampeners", "Negations"]
                    .iter()
                    .any(|option| name.eq_ignore_ascii_case(option))
            })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-VADER-OPTION",
                "unsupported vaderSentimentScores option",
                index,
            ));
        }
    }
    let score = if diagnostics.is_empty() {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![None, Some(1)]),
            StorageFact::Dense,
        )
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish_fixed_outputs(
        entry,
        request,
        vec![score.clone(), score.clone(), score.clone(), score],
        diagnostics,
    )
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
                &super::super::VADER_SENTIMENT_SCORES_CATALOG_ENTRY,
            );
            assert_eq!(
                result
                    .diagnostics
                    .iter()
                    .any(|d| d.code == "RM-CATALOG-VADER-ARITY"),
                !valid,
                "argument count {count}",
            );
        }
    }
}
