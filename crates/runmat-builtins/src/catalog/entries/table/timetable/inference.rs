use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

fn tabular_fact(class: StaticClassIdentity) -> ValueFact {
    // The object is scalar storage; its row count and variable schema are runtime values.
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn required_input(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-TIMETABLE-ARITY",
            "at least one input is required",
            0,
        ));
    }
}

fn input_class(
    request: &CallRequest,
    expected: StaticClassIdentity,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(input) = request.arguments.first() else {
        return;
    };
    match &input.kind {
        ValueKindFact::Unknown => {}
        ValueKindFact::Object(object) => {
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|class| class != &expected.owned())
            {
                diagnostics.push(argument_error(
                    "RM-TIMETABLE-INPUT-CLASS",
                    format!("expected a {} object", expected.display_name()),
                    0,
                ));
            }
        }
        _ => diagnostics.push(argument_error(
            "RM-TIMETABLE-INPUT-CLASS",
            format!("expected a {} object", expected.display_name()),
            0,
        )),
    }
}

pub(super) fn timetable(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        tabular_fact(runmat_types::standard::TIMETABLE),
        Vec::new(),
    )
}

pub(super) fn array2timetable(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    required_input(request, &mut diagnostics);
    finish_fixed(
        entry,
        request,
        tabular_fact(runmat_types::standard::TIMETABLE),
        diagnostics,
    )
}

pub(super) fn table2timetable(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    required_input(request, &mut diagnostics);
    input_class(request, runmat_types::standard::TABLE, &mut diagnostics);
    finish_fixed(
        entry,
        request,
        tabular_fact(runmat_types::standard::TIMETABLE),
        diagnostics,
    )
}

pub(super) fn timetable2table(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    required_input(request, &mut diagnostics);
    input_class(request, runmat_types::standard::TIMETABLE, &mut diagnostics);
    finish_fixed(
        entry,
        request,
        tabular_fact(runmat_types::standard::TABLE),
        diagnostics,
    )
}

pub(super) fn readtimetable(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    required_input(request, &mut diagnostics);
    if let Some(path) = request.arguments.first() {
        if !matches!(
            path.kind,
            ValueKindFact::Unknown | ValueKindFact::Character | ValueKindFact::String
        ) {
            diagnostics.push(argument_error(
                "RM-READTIMETABLE-PATH",
                "filename must be text",
                0,
            ));
        }
    }
    finish_fixed(
        entry,
        request,
        tabular_fact(runmat_types::standard::TIMETABLE),
        diagnostics,
    )
}
