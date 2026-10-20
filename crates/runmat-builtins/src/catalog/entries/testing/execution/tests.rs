use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, ObjectFact, OutputSelection, RequestedOutputCount,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(Vec::new()),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

#[test]
fn suite_and_results_retain_classes_but_not_a_fabricated_test_count() {
    assert_eq!(
        FUNCTIONTESTS_CATALOG_ENTRY.contract.purity,
        BuiltinPurity::DeterministicReadOnly
    );
    assert_eq!(
        FUNCTIONTESTS_CATALOG_ENTRY.contract.effects,
        &[EffectKind::MayThrow]
    );
    assert_eq!(
        FUNCTIONTESTS_CATALOG_ENTRY.contract.environment_effect,
        None
    );
    assert_eq!(FUNCTIONTESTS_CATALOG_ENTRY.contract.workspace_effect, None);
    assert!(FUNCTIONTESTS_CATALOG_ENTRY.contract.capabilities.is_empty());
    let suite = inference::functiontests(&call(vec![unknown()], 1), &FUNCTIONTESTS_CATALOG_ENTRY);
    assert!(suite.diagnostics.is_empty(), "{:#?}", suite.diagnostics);
    assert_eq!(suite.outputs[0].shape, ShapeFact::from(vec![Some(1), None]));
    let ValueKindFact::Object(object) = &suite.outputs[0].kind else {
        panic!("expected TestSuite fact")
    };
    assert!(object.runtime_class.as_ref().is_some_and(|class| class.is(
        runmat_types::StaticClassIdentity::new("matlab.unittest.TestSuite")
    )));

    let run = inference::run(
        &call(vec![unknown(), suite.outputs[0].clone()], 1),
        &TEST_RUNNER_RUN_CATALOG_ENTRY,
    );
    assert!(run.diagnostics.is_empty(), "{:#?}", run.diagnostics);
    assert_eq!(run.outputs[0].shape, ShapeFact::from(vec![Some(1), None]));
    let ValueKindFact::Object(result) = &run.outputs[0].kind else {
        panic!("expected TestResult fact")
    };
    assert!(result.runtime_class.as_ref().is_some_and(|class| class.is(
        runmat_types::StaticClassIdentity::new("matlab.unittest.TestResult")
    )));
    assert_eq!(result.properties["Passed"].kind, ValueKindFact::Logical);
    assert_eq!(result.properties["Failed"].kind, ValueKindFact::Logical);
}

#[test]
fn constructor_and_plugin_mutation_preserve_handle_semantics() {
    let runner = inference::with_text_output(&call(vec![], 1), &WITH_TEXT_OUTPUT_CATALOG_ENTRY);
    assert!(runner.diagnostics.is_empty());
    let ValueKindFact::Object(object) = &runner.outputs[0].kind else {
        panic!("expected runner")
    };
    assert_eq!(object.handle_semantics, Some(true));
    assert_eq!(runner.outputs[0].shape, ShapeFact::Scalar);
    let plugin = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::UNIT_TEST_RUNNER_PLUGIN.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: None,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let add = inference::add_plugin(
        &call(vec![runner.outputs[0].clone(), plugin], 1),
        &ADD_PLUGIN_CATALOG_ENTRY,
    );
    assert!(add.diagnostics.is_empty(), "{:#?}", add.diagnostics);
    assert_eq!(add.outputs[0], runner.outputs[0]);
}

#[test]
fn proven_invalid_forms_report_diagnostics_without_guessing_runtime_outcomes() {
    let number = ValueFact::scalar(ValueKindFact::Logical);
    assert!(
        inference::functiontests(&call(vec![number.clone()], 1), &FUNCTIONTESTS_CATALOG_ENTRY)
            .diagnostics
            .iter()
            .any(|item| item.code == "RM-TESTING-FUNCTIONTESTS-CELL")
    );
    assert!(inference::with_text_output(
        &call(vec![unknown()], 1),
        &WITH_TEXT_OUTPUT_CATALOG_ENTRY
    )
    .diagnostics
    .iter()
    .any(|item| item.code == "RM-TESTING-RUNNER-OPTIONS"));
    assert!(inference::with_text_output(
        &call(vec![unknown(), unknown()], 1),
        &WITH_TEXT_OUTPUT_CATALOG_ENTRY
    )
    .diagnostics
    .iter()
    .any(|item| item.code == "RM-TESTING-UNSUPPORTED-RUNNER-OPTION"));
    assert!(inference::add_plugin(
        &call(vec![number.clone(), number.clone()], 1),
        &ADD_PLUGIN_CATALOG_ENTRY
    )
    .diagnostics
    .iter()
    .any(|item| item.code == "RM-TESTING-RUNNER-RECEIVER"));
    assert!(inference::run(
        &call(vec![number, unknown()], 1),
        &TEST_RUNNER_RUN_CATALOG_ENTRY
    )
    .diagnostics
    .iter()
    .any(|item| item.code == "RM-TESTING-RUNNER-RECEIVER"));
}

#[test]
fn examples_assert_real_structured_pass_and_failure() {
    assert!(documentation::ADD_PLUGIN.examples[0]
        .program
        .contains("runner.addPlugin(plugin)"));
    assert!(documentation::RUN.examples[0]
        .program
        .contains("runner.run(suite)"));
    let BuiltinExampleFixture::Filesystem(fixture) = documentation::RUN.examples[0].fixture else {
        panic!("test run requires a source fixture")
    };
    assert_eq!(fixture.entries.len(), 1);
    let BuiltinFilesystemEntry::File {
        content: BuiltinFixtureContent::Utf8(source),
        ..
    } = fixture.entries[0]
    else {
        panic!("test file must be UTF-8")
    };
    assert!(source.contains("functiontests(localfunctions)"));
    assert!(source.contains("function testPass"));
    assert!(source.contains("function testFail"));
    let BuiltinExampleVerification::Assertions { source: assertions } =
        documentation::RUN.examples[0].verification
    else {
        panic!("result fields must be asserted")
    };
    assert!(assertions.contains("results(1).Passed"));
    assert!(assertions.contains("results(2).Failed"));
}
