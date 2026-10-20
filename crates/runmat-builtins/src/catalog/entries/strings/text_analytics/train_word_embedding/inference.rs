use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len().is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TRAINWORDEMBEDDING-ARITY",
            "trainWordEmbedding requires a source followed by name-value pairs",
            0,
        ));
    }
    if let Some(source) = request.arguments.first() {
        if !matches!(
            source.kind,
            ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
                | ValueKindFact::Object(_)
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRAINWORDEMBEDDING-SOURCE",
                "trainWordEmbedding source must be a host filename or tokenizedDocument object",
                0,
            ));
        }
    }
    finish_fixed(entry, request, embedding_fact(), diagnostics)
}

fn embedding_fact() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("wordEmbedding").owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}
