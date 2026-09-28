use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 4 {
        diagnostics.push(argument_error(
            "RM-PIVOT-ARITY",
            "expected a table and row, column, and data variable selectors",
            0,
        ));
    } else if !(request.arguments.len() - 4).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-PIVOT-OPTIONS",
            "name-value options require complete pairs",
            4,
        ));
    }
    if let Some(input) = request.arguments.first() {
        match &input.kind {
            ValueKindFact::Unknown => {}
            ValueKindFact::Object(object)
                if object.runtime_class.as_ref().is_none_or(|class| {
                    class == &standard::TABLE.owned() || class == &standard::TIMETABLE.owned()
                }) => {}
            _ => diagnostics.push(argument_error(
                "RM-PIVOT-TABLE",
                "expected a table or timetable input",
                0,
            )),
        }
    }
    // Keys and output variables depend on the table's runtime payload, not its
    // class alone. Preserve table identity without inventing a column schema.
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TABLE.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, output, diagnostics)
}
