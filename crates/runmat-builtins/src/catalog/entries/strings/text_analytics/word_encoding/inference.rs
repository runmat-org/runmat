use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-WORDENCODING-ARITY",
            "wordEncoding requires documents or words",
            0,
        ));
    }
    let output = if diagnostics.is_empty() {
        ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(StaticClassIdentity::new("wordEncoding").owned()),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Scalar,
            StorageFact::Opaque,
        )
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish_fixed(entry, request, output, diagnostics)
}
