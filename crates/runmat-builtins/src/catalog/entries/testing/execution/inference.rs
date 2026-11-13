use std::collections::BTreeMap;

use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, DynamicReason, MutationFact, ObjectFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

const RUNNER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.unittest.TestRunner");
const SUITE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.unittest.TestSuite");
const RESULT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.unittest.TestResult");

fn object(
    class: runmat_types::StaticClassIdentity,
    shape: ShapeFact,
    properties: BTreeMap<String, ValueFact>,
    handle: bool,
) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties,
            properties_complete: false,
            handle_semantics: Some(handle),
        }),
        shape,
        StorageFact::Opaque,
    );
    if handle {
        fact.alias = AliasFact::Identity;
        fact.mutation = MutationFact::HandleSemantics;
    }
    fact
}

fn runner() -> ValueFact {
    object(
        RUNNER_CLASS,
        ShapeFact::Scalar,
        BTreeMap::from([(
            "Plugins".into(),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        )]),
        true,
    )
}

fn suite() -> ValueFact {
    object(
        SUITE_CLASS,
        ShapeFact::from(vec![Some(1), None]),
        BTreeMap::from([
            ("Name".into(), ValueFact::scalar(ValueKindFact::String)),
            (
                "ProcedureName".into(),
                ValueFact::scalar(ValueKindFact::String),
            ),
            ("TestFile".into(), ValueFact::scalar(ValueKindFact::String)),
        ]),
        false,
    )
}

fn results() -> ValueFact {
    object(
        RESULT_CLASS,
        ShapeFact::from(vec![Some(1), None]),
        BTreeMap::from([
            ("Name".into(), ValueFact::scalar(ValueKindFact::String)),
            ("Passed".into(), ValueFact::scalar(ValueKindFact::Logical)),
            ("Failed".into(), ValueFact::scalar(ValueKindFact::Logical)),
            (
                "Incomplete".into(),
                ValueFact::scalar(ValueKindFact::Logical),
            ),
            ("Details".into(), ValueFact::scalar(ValueKindFact::String)),
        ]),
        false,
    )
}

fn invalid_arity(
    request: &CallRequest,
    expected: usize,
    name: &str,
) -> Vec<runmat_types::InferenceDiagnostic> {
    if request.arguments.len() == expected {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-TESTING-EXECUTION-ARITY",
            format!("{name} expects {expected} input(s)"),
            request.arguments.len(),
        )]
    }
}

fn object_or_unknown(fact: &ValueFact) -> bool {
    matches!(
        &fact.kind,
        ValueKindFact::Object(_) | ValueKindFact::Unknown
    )
}

pub(super) fn functiontests(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = invalid_arity(request, 1, "functiontests");
    if request.arguments.first().is_some_and(|argument| {
        !matches!(
            &argument.kind,
            ValueKindFact::Cell(_) | ValueKindFact::Unknown
        )
    }) {
        diagnostics.push(argument_error(
            "RM-TESTING-FUNCTIONTESTS-CELL",
            "functiontests requires a cell array of function handles",
            0,
        ));
    }
    finish_fixed(entry, request, suite(), diagnostics)
}

pub(super) fn with_text_output(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else if !request.arguments.len().is_multiple_of(2) {
        vec![argument_error(
            "RM-TESTING-RUNNER-OPTIONS",
            "withTextOutput requires name-value pairs",
            request.arguments.len() - 1,
        )]
    } else {
        vec![argument_error(
            "RM-TESTING-UNSUPPORTED-RUNNER-OPTION",
            "withTextOutput options are not supported yet",
            0,
        )]
    };
    finish_fixed(entry, request, runner(), diagnostics)
}

pub(super) fn add_plugin(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = invalid_arity(request, 2, "addPlugin");
    if request
        .arguments
        .first()
        .is_some_and(|argument| !object_or_unknown(argument))
    {
        diagnostics.push(argument_error(
            "RM-TESTING-RUNNER-RECEIVER",
            "addPlugin requires a TestRunner handle",
            0,
        ));
    }
    if request
        .arguments
        .get(1)
        .is_some_and(|argument| !object_or_unknown(argument))
    {
        diagnostics.push(argument_error(
            "RM-TESTING-RUNNER-PLUGIN",
            "addPlugin requires a plugin object",
            1,
        ));
    }
    // Runtime checks exact class/subclass identity. Preserve proven identity
    // only for the concrete TestRunner; otherwise successful execution proves
    // the returned handle is a runner, but not the receiver's class name.
    let output = request.arguments.first().filter(|fact| matches!(&fact.kind, ValueKindFact::Object(object) if object.handle_semantics == Some(true) && object.runtime_class.as_ref().is_some_and(|class| class.is(RUNNER_CLASS)))).cloned().unwrap_or_else(runner);
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn run(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = invalid_arity(request, 2, "TestRunner.run");
    if request
        .arguments
        .first()
        .is_some_and(|argument| !object_or_unknown(argument))
    {
        diagnostics.push(argument_error(
            "RM-TESTING-RUNNER-RECEIVER",
            "run requires a TestRunner handle",
            0,
        ));
    }
    if request
        .arguments
        .get(1)
        .is_some_and(|argument| !object_or_unknown(argument))
    {
        diagnostics.push(argument_error(
            "RM-TESTING-RUNNER-SUITE",
            "run requires a TestSuite selection",
            1,
        ));
    }
    finish_fixed(entry, request, results(), diagnostics)
}
