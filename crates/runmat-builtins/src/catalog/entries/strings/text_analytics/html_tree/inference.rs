use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-HTMLTREE-ARITY",
            "htmlTree requires exactly one input",
            request.arguments.len().min(1),
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
                | ValueKindFact::Object(_)
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-HTMLTREE-TEXT",
                "htmlTree input must be host text",
                0,
            ));
        }
    }
    let output = if diagnostics.is_empty() {
        ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(StaticClassIdentity::new("htmlTree").owned()),
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
