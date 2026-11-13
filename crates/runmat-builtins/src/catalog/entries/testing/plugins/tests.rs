use super::*;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn assert_plugin_class(result: CallInference, class: runmat_types::StaticClassIdentity) {
    assert!(result.diagnostics.is_empty(), "{:#?}", result.diagnostics);
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(
        result.outputs[0].residency,
        runmat_types::ResidencyFact::Host
    );
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("plugin constructor must infer an object");
    };
    assert_eq!(object.runtime_class, Some(class.owned()));
    assert_eq!(object.handle_semantics, None);
}

#[test]
fn base_plugin_is_a_scalar_host_object_without_arguments() {
    let mut entries = Vec::new();
    extend_entries(&mut entries);
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.identity.name)
            .collect::<Vec<_>>(),
        [
            "matlab.unittest.plugins.TestRunnerPlugin",
            "matlab.unittest.plugins.CodeCoveragePlugin.forFolder"
        ]
    );
    assert_plugin_class(
        inference::test_runner_plugin(&call(vec![], vec![], 1), &TEST_RUNNER_PLUGIN_CATALOG_ENTRY),
        runmat_types::standard::UNIT_TEST_RUNNER_PLUGIN,
    );
    assert!(inference::test_runner_plugin(
        &call(vec![unknown()], vec![], 1),
        &TEST_RUNNER_PLUGIN_CATALOG_ENTRY,
    )
    .diagnostics
    .iter()
    .any(|diagnostic| diagnostic.code == "RM-TESTING-PLUGIN-ARITY"));
}

#[test]
fn folder_plugin_tracks_concrete_properties_and_option() {
    let request = call(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::String("src".into()),
            LiteralValue::String("IncludingSubfolders".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    let result =
        inference::code_coverage_for_folder(&request, &CODE_COVERAGE_FOR_FOLDER_CATALOG_ENTRY);
    assert_plugin_class(
        result.clone(),
        runmat_types::standard::UNIT_TEST_CODE_COVERAGE_PLUGIN,
    );
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("expected plugin object");
    };
    assert_eq!(
        object.properties["Folders"].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    assert_eq!(
        object.properties["IncludingSubfolders"].kind,
        ValueKindFact::Logical
    );
    let discarded = inference::code_coverage_for_folder(
        &call(
            request.arguments.clone(),
            request.literals.literal_args.clone(),
            0,
        ),
        &CODE_COVERAGE_FOR_FOLDER_CATALOG_ENTRY,
    );
    assert!(discarded.diagnostics.is_empty());
    assert!(discarded.outputs.is_empty());
}

#[test]
fn invalid_or_missing_folder_and_option_are_reported_statically_when_proven() {
    let empty = inference::code_coverage_for_folder(
        &call(vec![], vec![], 1),
        &CODE_COVERAGE_FOR_FOLDER_CATALOG_ENTRY,
    );
    assert!(empty
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TESTING-COVERAGE-FOLDER-ARITY"));

    let number = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let non_text = inference::code_coverage_for_folder(
        &call(vec![number.clone()], vec![LiteralValue::Number(3.0)], 1),
        &CODE_COVERAGE_FOR_FOLDER_CATALOG_ENTRY,
    );
    assert!(non_text
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TESTING-COVERAGE-FOLDER"));

    let empty_array = inference::code_coverage_for_folder(
        &call(
            vec![ValueFact::proven(
                ValueKindFact::String,
                ShapeFact::from(vec![Some(1), Some(0)]),
                StorageFact::Dense,
            )],
            vec![],
            1,
        ),
        &CODE_COVERAGE_FOR_FOLDER_CATALOG_ENTRY,
    );
    assert!(empty_array
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TESTING-COVERAGE-FOLDER-EMPTY"));

    let invalid_option = inference::code_coverage_for_folder(
        &call(
            vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                number,
            ],
            vec![
                LiteralValue::String("src".into()),
                LiteralValue::String("IncludingSubfolders".into()),
                LiteralValue::Number(2.0),
            ],
            1,
        ),
        &CODE_COVERAGE_FOR_FOLDER_CATALOG_ENTRY,
    );
    assert!(invalid_option
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TESTING-COVERAGE-SUBFOLDERS"));

    let char_option = inference::code_coverage_for_folder(
        &call(
            vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Character),
                ValueFact::scalar(ValueKindFact::Logical),
            ],
            vec![
                LiteralValue::String("src".into()),
                LiteralValue::Character("IncludingSubfolders".into()),
                LiteralValue::Unknown,
            ],
            1,
        ),
        &CODE_COVERAGE_FOR_FOLDER_CATALOG_ENTRY,
    );
    assert!(char_option.diagnostics.is_empty());

    let unknown_folder = inference::code_coverage_for_folder(
        &call(vec![unknown()], vec![], 1),
        &CODE_COVERAGE_FOR_FOLDER_CATALOG_ENTRY,
    );
    assert!(unknown_folder.diagnostics.is_empty());
}

#[test]
fn known_unsupported_coverage_options_are_never_inferred_as_extra_folders() {
    for name in ["Producing", "MetricLevel", "Filter"] {
        for option in [
            LiteralValue::String(name.into()),
            LiteralValue::Character(name.into()),
        ] {
            let result = inference::code_coverage_for_folder(
                &call(
                    vec![
                        ValueFact::scalar(ValueKindFact::String),
                        ValueFact::scalar(ValueKindFact::String),
                        ValueFact::scalar(ValueKindFact::String),
                    ],
                    vec![
                        LiteralValue::String("src".into()),
                        option,
                        LiteralValue::String("value".into()),
                    ],
                    1,
                ),
                &CODE_COVERAGE_FOR_FOLDER_CATALOG_ENTRY,
            );
            assert!(result
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "RM-TESTING-COVERAGE-UNSUPPORTED-OPTION"));
        }
    }

    let dangling = inference::code_coverage_for_folder(
        &call(
            vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            vec![
                LiteralValue::String("src".into()),
                LiteralValue::String("IncludingSubfolders".into()),
            ],
            1,
        ),
        &CODE_COVERAGE_FOR_FOLDER_CATALOG_ENTRY,
    );
    assert!(dangling
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TESTING-COVERAGE-SUBFOLDERS"));
}
