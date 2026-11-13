use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-READWORDEMBEDDING-ARITY",
            "readWordEmbedding requires exactly one filename",
            0,
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-READWORDEMBEDDING-FILENAME",
                "readWordEmbedding requires a host text filename",
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
