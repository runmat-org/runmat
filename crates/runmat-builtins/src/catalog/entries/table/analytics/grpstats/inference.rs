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

fn numeric_summary(interval: bool) -> ValueFact {
    let shape = if interval {
        ShapeFact::from(vec![None, None, Some(2)])
    } else {
        ShapeFact::from(vec![None, None])
    };
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn group_names() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![None, None]),
        StorageFact::Dense,
    )
}

fn statistic_fact(name: &str) -> Option<ValueFact> {
    match name.to_ascii_lowercase().as_str() {
        "gname" => Some(group_names()),
        "meanci" | "predci" => Some(numeric_summary(true)),
        "mean" | "sem" | "std" | "var" | "min" | "max" | "range" | "numel" | "count"
        | "groupcount" => Some(numeric_summary(false)),
        _ => None,
    }
}

fn known_statistics(request: &CallRequest) -> Option<Vec<ValueFact>> {
    if request.arguments.len() == 2 {
        return Some(vec![numeric_summary(false)]);
    }
    match request.literals.literal_args.get(2) {
        Some(
            LiteralValue::Character(name)
            | LiteralValue::String(name)
            | LiteralValue::Keyword(name),
        ) if matches!(
            name.to_ascii_lowercase().as_str(),
            "alpha" | "datavars" | "varnames"
        ) =>
        {
            Some(vec![numeric_summary(false)])
        }
        Some(
            LiteralValue::Character(name)
            | LiteralValue::String(name)
            | LiteralValue::Keyword(name),
        ) => statistic_fact(name).map(|fact| vec![fact]),
        Some(LiteralValue::Vector(names)) if !names.is_empty() => names
            .iter()
            .map(|name| match name {
                LiteralValue::Character(name)
                | LiteralValue::String(name)
                | LiteralValue::Keyword(name) => statistic_fact(name),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn unknown_outputs(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    mut diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::RuntimeValue);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    diagnostics.append(&mut result.diagnostics);
    result.diagnostics = diagnostics;
    result
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-GRPSTATS-ARITY",
            "expected data and grouping variables",
            0,
        ));
    }
    match request.arguments.first().map(|fact| &fact.kind) {
        Some(ValueKindFact::Object(object))
            if object.runtime_class.as_ref().is_some_and(|class| {
                class == &standard::TABLE.owned() || class == &standard::TIMETABLE.owned()
            }) =>
        {
            finish_fixed(entry, request, table_fact(), diagnostics)
        }
        Some(ValueKindFact::Numeric(_)) | Some(ValueKindFact::Logical) => {
            if let Some(outputs) = known_statistics(request) {
                if request
                    .outputs
                    .requested
                    .known_count()
                    .is_some_and(|count| count != 0 && count != outputs.len())
                {
                    diagnostics.push(argument_error(
                        "RM-GRPSTATS-OUTPUT-COUNT",
                        "requested output count must match the selected statistics",
                        2,
                    ));
                }
                finish_fixed_outputs(entry, request, outputs, diagnostics)
            } else {
                unknown_outputs(request, entry, diagnostics)
            }
        }
        Some(ValueKindFact::Unknown) | None => unknown_outputs(request, entry, diagnostics),
        Some(ValueKindFact::Object(object)) if object.runtime_class.is_none() => {
            unknown_outputs(request, entry, diagnostics)
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-GRPSTATS-DATA",
                "expected a table or numeric/logical matrix",
                0,
            ));
            unknown_outputs(request, entry, diagnostics)
        }
    }
}
