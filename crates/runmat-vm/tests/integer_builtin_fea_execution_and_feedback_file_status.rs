#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::Value;
use test_helpers::execute_source;

const INTEGER_CONSTRUCTORS: [&str; 8] = [
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
];

fn object_of_class<'a>(values: &'a [Value], class_name: &str) -> &'a runmat_value::ObjectInstance {
    let class_name = runmat_types::ClassIdentity::new(class_name).expect("test class identity");
    values
        .iter()
        .find_map(|value| match value {
            Value::Object(object) if object.class_name() == &class_name => Some(object),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected {class_name} object"))
}

#[test]
fn fea_trends_accepts_every_exact_integer_class() {
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            r#"
            trends = fea.trends("WindowSize", {constructor}(1));
            "#
        );
        let values = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor} FEA structural controls: {error}"));
        object_of_class(&values, "fea.Trends");
    }
}

#[test]
fn fea_results_validates_one_based_selectors_and_zero_one_flags_before_lookup() {
    use runmat_builtins::fea_contract::results_option::ResultsOption;
    for constructor in INTEGER_CONSTRUCTORS {
        for option in [
            ResultsOption::ModeIndices,
            ResultsOption::TransientSnapshotIndices,
        ] {
            let option = option.canonical_name();
            let accepted = format!(
                "value = fea.results(\"missing_integer_contract_run\", \"{option}\", {constructor}([1 2]));"
            );
            let error = execute_source(&accepted)
                .expect_err("missing run must fail after selector parsing");
            assert_eq!(
                error.identifier(),
                Some("RunMat:fea:OperationFailed"),
                "{accepted}"
            );

            let zero = format!(
                "value = fea.results(\"missing_integer_contract_run\", \"{option}\", {constructor}(0));"
            );
            let error = execute_source(&zero).expect_err("zero is not a public one-based index");
            assert_eq!(
                error.identifier(),
                Some("RunMat:fea:InvalidInput"),
                "{zero}"
            );
        }

        for flag in [
            ResultsOption::IncludeFieldValues,
            ResultsOption::IncludeDiagnostics,
            ResultsOption::IncludeModalResults,
            ResultsOption::IncludeTransientResults,
            ResultsOption::IncludeNonlinearResults,
        ] {
            let flag = flag.canonical_name();
            let accepted = format!(
                "value = fea.results(\"missing_integer_contract_run\", \"{flag}\", {constructor}(1));"
            );
            let error =
                execute_source(&accepted).expect_err("missing run must fail after flag parsing");
            assert_eq!(
                error.identifier(),
                Some("RunMat:fea:OperationFailed"),
                "{accepted}"
            );

            let invalid = format!(
                "value = fea.results(\"missing_integer_contract_run\", \"{flag}\", {constructor}(2));"
            );
            let error =
                execute_source(&invalid).expect_err("numeric predicates require zero or one");
            assert_eq!(
                error.identifier(),
                Some("RunMat:fea:InvalidInput"),
                "{invalid}"
            );
        }
        let unsupported = format!(
            "value = fea.results(\"missing_integer_contract_run\", \"IncludeElectromagneticResults\", {constructor}(1));"
        );
        assert_eq!(
            execute_source(&unsupported)
                .expect_err("unsupported selector must fail before lookup")
                .identifier(),
            Some("RunMat:fea:InvalidInput"),
            "{unsupported}"
        );
    }
}

#[test]
fn fea_text_and_typed_object_workflow_roles_never_coerce_integers() {
    for constructor in INTEGER_CONSTRUCTORS {
        for source in [
            format!("value = fea.StructuralStep.modal({constructor}(1));"),
            format!("value = fea.study({constructor}(1), {constructor}(2));"),
            format!("value = fea.sweep({constructor}(1), {constructor}(2));"),
            format!("value = fea.run({constructor}(1));"),
            format!("value = fea.validate({constructor}(1));"),
        ] {
            let error =
                execute_source(&source).expect_err("integer identity/object role must reject");
            assert_eq!(
                error.identifier(),
                Some("RunMat:fea:InvalidInput"),
                "{source}"
            );
        }
    }
}

#[test]
fn feedback_gains_and_feof_file_ids_accept_every_runmat_integer_extension_class() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            r#"
            system = tf([1], [1 1]);
            closed = feedback(system, {constructor}(1));
            atEnd = feof({constructor}(1));
            "#
        );
        let values = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor} feedback/feof extensions: {error}"));
        assert!(values
            .iter()
            .any(|value| matches!(value, Value::Bool(false))));
        assert!(values.iter().any(|value| {
            matches!(value, Value::Object(object) if object.class_name().is(runmat_types::standard::TRANSFER_FUNCTION))
        }));
    }
}

#[test]
fn feedback_and_feof_integer_extensions_have_stable_strict_mode_errors() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let system_error =
        execute_source("system = tf([1], [1 1]); value = feedback(system, int16(1));")
            .expect_err("typed integer gain is independently gated");
    assert_eq!(
        system_error.identifier(),
        Some("RunMat:compatibility:FeedbackIntegerScalarGainExtension")
    );

    let file_error = execute_source("value = feof(uint32(1));")
        .expect_err("typed integer file id is independently gated");
    assert_eq!(
        file_error.identifier(),
        Some("RunMat:compatibility:FeofIntegerIdExtension")
    );
}
