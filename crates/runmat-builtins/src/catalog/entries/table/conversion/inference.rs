use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

fn table_fact() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::TABLE.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn arity(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>, exact: bool) {
    if request.arguments.is_empty() || (exact && request.arguments.len() != 1) {
        diagnostics.push(argument_error(
            "RM-TABLE-CONVERSION-ARITY",
            if exact {
                "expected exactly one input"
            } else {
                "at least one input is required"
            },
            0,
        ));
    }
}

fn table_input(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(input) = request.arguments.first() else {
        return;
    };
    match &input.kind {
        ValueKindFact::Unknown => {}
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_none_or(|class| {
                class == &runmat_types::standard::TABLE.owned()
                    || class == &runmat_types::standard::TIMETABLE.owned()
            }) => {}
        _ => diagnostics.push(argument_error(
            "RM-TABLE-CONVERSION-INPUT",
            "expected a table or timetable object",
            0,
        )),
    }
}

fn table_output(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    mut diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    arity(request, &mut diagnostics, false);
    if request.arguments.len() > 1 && request.arguments.len().is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-TABLE-CONVERSION-OPTIONS",
            "name-value options require complete pairs",
            1,
        ));
    }
    finish_fixed(entry, request, table_fact(), diagnostics)
}

pub(super) fn array2table(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    // Runtime accepts typed array-like objects (including datetime and duration) as
    // table variables. Their class alone cannot prove an invalid conversion.
    table_output(request, entry, Vec::new())
}

pub(super) fn cell2table(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(input) = request.arguments.first() {
        if !matches!(input.kind, ValueKindFact::Unknown | ValueKindFact::Cell(_)) {
            diagnostics.push(argument_error(
                "RM-CELL2TABLE-INPUT",
                "expected a cell array",
                0,
            ));
        }
    }
    table_output(request, entry, diagnostics)
}

pub(super) fn struct2table(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::Unknown | ValueKindFact::Struct(_)
        ) {
            diagnostics.push(argument_error(
                "RM-STRUCT2TABLE-INPUT",
                "expected a structure or structure array",
                0,
            ));
        }
    }
    table_output(request, entry, diagnostics)
}

pub(super) fn table2array(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics, true);
    table_input(request, &mut diagnostics);
    // Table schema is runtime data. A scalar object fact does not determine array extent or class.
    finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        diagnostics,
    )
}

pub(super) fn table2cell(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics, true);
    table_input(request, &mut diagnostics);
    let cells = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![None, None]),
        StorageFact::Dense,
    );
    finish_fixed(entry, request, cells, diagnostics)
}

pub(super) fn table2struct(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 1 | 3) {
        diagnostics.push(argument_error(
            "RM-TABLE2STRUCT-ARITY",
            "expected a table, optionally followed by 'ToScalar' and a selector",
            0,
        ));
    }
    table_input(request, &mut diagnostics);
    // The row-oriented default and ToScalar option have different shapes; preserve uncertainty.
    let structure = ValueFact::proven(
        ValueKindFact::Struct(StructFact::array(BTreeMap::new(), false, Vec::new(), false)),
        ShapeFact::Unknown,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, structure, diagnostics)
}
