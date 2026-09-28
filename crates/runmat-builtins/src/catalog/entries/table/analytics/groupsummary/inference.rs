use crate::catalog::inference::{argument_error, finish_fixed, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

fn table_fact() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TABLE.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 {
        diagnostics.push(argument_error(
            "RM-GROUPSUMMARY-ARITY",
            "expected data, grouping variables, and a method or bin specification",
            0,
        ));
    }
    let input = request.arguments.first();
    match input.map(|fact| &fact.kind) {
        Some(ValueKindFact::Object(object))
            if object.runtime_class.as_ref().is_some_and(|class| {
                class == &standard::TABLE.owned() || class == &standard::TIMETABLE.owned()
            }) =>
        {
            finish_fixed(entry, request, table_fact(), diagnostics)
        }
        Some(ValueKindFact::Numeric(_)) | Some(ValueKindFact::Cell(_)) => {
            // Method and group cardinality determine summary class and shape.
            // Integer min/max can retain class; floating statistics return double.
            let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
            let counts = ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![None, None]),
                StorageFact::Dense,
            );
            finish_fixed_outputs(
                entry,
                request,
                vec![unknown.clone(), unknown, counts],
                diagnostics,
            )
        }
        Some(ValueKindFact::Unknown) | None => {
            // The runtime may select the table or array overload after dispatch.
            let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
            finish_fixed_outputs(
                entry,
                request,
                vec![unknown.clone(), unknown.clone(), unknown],
                diagnostics,
            )
        }
        Some(ValueKindFact::Object(object)) if object.runtime_class.is_none() => {
            let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
            finish_fixed_outputs(
                entry,
                request,
                vec![unknown.clone(), unknown.clone(), unknown],
                diagnostics,
            )
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-GROUPSUMMARY-DATA",
                "expected a table, timetable, numeric array, or cell array of numeric arrays",
                0,
            ));
            let unknown = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            finish_fixed_outputs(
                entry,
                request,
                vec![unknown.clone(), unknown.clone(), unknown],
                diagnostics,
            )
        }
    }
}
