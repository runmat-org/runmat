use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RunSource {
    Code,
    File,
}

pub(super) fn infer_with(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    contract: CallContract,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = contract;
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub(super) fn environment() -> ValueFact {
    let text = ValueFact::scalar(ValueKindFact::String);
    let properties = BTreeMap::from([
        ("Version".into(), text.clone()),
        ("Executable".into(), text.clone()),
        ("Library".into(), text.clone()),
        ("Home".into(), text.clone()),
        ("Status".into(), text.clone()),
        ("ExecutionMode".into(), text.clone()),
        (
            "ProcessID".into(),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ),
        ("ProcessName".into(), text),
    ]);
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::PYTHON_ENVIRONMENT_CLASS.owned()),
            properties,
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

pub(super) fn bundle(request: &CallRequest) -> ValueFact {
    let count = request.arguments.len() / 2;
    let text = ValueFact::scalar(ValueKindFact::String);
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let names = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(text.clone()),
            elements: vec![text; count],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(count)]),
        StorageFact::Opaque,
    );
    let values = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(unknown),
            elements: request
                .arguments
                .iter()
                .skip(1)
                .step_by(2)
                .cloned()
                .collect(),
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(count)]),
        StorageFact::Opaque,
    );
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::PYTHON_ARGUMENTS.owned()),
            properties: BTreeMap::from([("Names".into(), names), ("Values".into(), values)]),
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

pub(super) fn definitely_not_text(fact: &ValueFact) -> bool {
    !matches!(
        fact.kind,
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
    )
}

pub(super) fn text_at(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
    label: &str,
) {
    if request
        .arguments
        .get(index)
        .is_some_and(definitely_not_text)
    {
        diagnostics.push(argument_error(
            "RM-PYTHON-TEXT",
            format!("{label} must be text"),
            index,
        ));
    }
    if request
        .literals
        .literal_string_at(index)
        .is_some_and(|value| value.is_empty())
    {
        diagnostics.push(argument_error(
            "RM-PYTHON-EMPTY",
            format!("{label} must be non-empty"),
            index,
        ));
    }
}

pub(super) fn pairs(
    request: &CallRequest,
    start: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
    label: &str,
) {
    if request.arguments.len() < start || !(request.arguments.len() - start).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-PYTHON-PAIRS",
            format!("{label} requires name-value pairs"),
            start,
        ));
        return;
    }
    for index in (start..request.arguments.len()).step_by(2) {
        text_at(request, index, diagnostics, "Python name");
    }
}

pub(super) fn run(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    source: RunSource,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let file = source == RunSource::File;
    let name = if file { "pyrunfile" } else { "pyrun" };
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-PYTHON-ARITY",
            format!("{name} requires source"),
            0,
        ));
    } else {
        text_at(
            request,
            0,
            &mut diagnostics,
            if file { "Python file" } else { "Python code" },
        );
    }
    let outputs = request.outputs.requested.known_count();
    if outputs.is_some_and(|count| count > 0) {
        if request.arguments.len() < 2 {
            diagnostics.push(argument_error(
                "RM-PYTHON-OUTPUTS",
                format!("{name} requires Python output names when outputs are requested"),
                1,
            ));
        } else if request.arguments.get(1).is_some_and(|fact| {
            !matches!(
                fact.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::String
                    | ValueKindFact::Character
                    | ValueKindFact::Cell(_)
            )
        }) {
            diagnostics.push(argument_error(
                "RM-PYTHON-OUTPUTS",
                "Python output names must be text or a cell of text",
                1,
            ));
        }
    }
    if let Some(count) = outputs {
        let first_input = if count == 0 { 1 } else { 2 };
        if request.arguments.len() >= first_input {
            // The file adapter removes an optional Arguments pair before
            // named-input admission, so tail parity still applies to both.
            pairs(request, first_input, &mut diagnostics, name);
        }
        if count > 1
            && request.arguments.get(1).is_some_and(|fact| {
                matches!(fact.kind, ValueKindFact::String | ValueKindFact::Character)
                    && fact.is_scalar()
            })
        {
            diagnostics.push(argument_error(
                "RM-PYTHON-OUTPUTS",
                "multiple requested outputs require multiple Python output names",
                1,
            ));
        }
    }
    let mut contract = CallContract::dynamic(DynamicReason::ForeignBoundary);
    contract.outputs = vec![ValueFact::unknown(DynamicReason::ForeignBoundary)];
    infer_with(request, entry, contract, diagnostics)
}
