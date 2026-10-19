use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    class: StaticClassIdentity,
    assignment: bool,
) -> CallInference {
    let expected_count = if assignment { 3 } else { 2 };
    let mut diagnostics = Vec::new();
    if request.arguments.len() != expected_count {
        diagnostics.push(argument_error(
            "RM-TABULAR-SUBSCRIPT-ARITY",
            format!("expected {expected_count} inputs"),
            0,
        ));
    }
    let receiver = request.arguments.first();
    if let Some(receiver) = receiver {
        let admissible = match &receiver.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => object.runtime_class.as_ref().is_none_or(|actual| {
                actual == &class.owned()
                    || (class == standard::TABLE && actual == &standard::TIMETABLE.owned())
            }),
            _ => false,
        };
        if !admissible {
            diagnostics.push(argument_error(
                "RM-TABULAR-SUBSCRIPT-RECEIVER",
                format!("expected a {} object", class.display_name()),
                0,
            ));
        }
    }
    if let Some(path) = request.arguments.get(1) {
        if !matches!(path.kind, ValueKindFact::Unknown | ValueKindFact::Struct(_)) {
            diagnostics.push(argument_error(
                "RM-TABULAR-SUBSCRIPT-PATH",
                "expected a standard subscript structure or structure array",
                1,
            ));
        }
    }
    let output = if assignment {
        // Indexing can replace variables, dictionary entries, and nested values;
        // retaining pre-assignment property facts would be unsound.
        receiver
            .and_then(|receiver| match &receiver.kind {
                ValueKindFact::Object(object) => object.runtime_class.as_ref().and_then(|actual| {
                    (actual == &class.owned()
                        || (class == standard::TABLE && actual == &standard::TIMETABLE.owned()))
                        .then(|| {
                            ValueFact::proven(
                                ValueKindFact::Object(ObjectFact {
                                    class: None,
                                    runtime_class: Some(actual.clone()),
                                    properties: BTreeMap::new(),
                                    properties_complete: false,
                                    handle_semantics: Some(false),
                                }),
                                ShapeFact::Scalar,
                                StorageFact::Opaque,
                            )
                        })
                }),
                _ => None,
            })
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::DynamicDispatch))
    } else {
        // A runtime substruct can select a table, a column, an element, or an
        // arbitrary nested value, including a resident value.
        ValueFact::unknown(DynamicReason::DynamicDispatch)
    };
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn table_subsref(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, standard::TABLE, false)
}

pub(super) fn table_subsasgn(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, standard::TABLE, true)
}

pub(super) fn dictionary_subsref(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer(request, entry, standard::DICTIONARY, false)
}

pub(super) fn dictionary_subsasgn(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer(request, entry, standard::DICTIONARY, true)
}
