#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
use test_helpers::execute_source;

const INTEGER_CONSTRUCTORS: [&str; 8] = [
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
];

#[test]
fn compiled_exp_expm1_and_exprnd_cover_every_integer_class_in_runmat_mode() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            r#"
            a = exp({constructor}(1));
            b = expm1({constructor}(1));
            samples = exprnd({constructor}(2), {constructor}(1), {constructor}(3));
            if abs(a - exp(1)) > 1e-12 || abs(b - expm1(1)) > 1e-12 || numel(samples) ~= 3
                error('integer:numeric', 'compiled numeric semantics mismatch');
            end
            "#
        );
        execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor} compiled numeric cohort: {error}"));
    }
}

#[test]
fn compiled_erf_and_erfcinv_reject_every_integer_class() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for constructor in INTEGER_CONSTRUCTORS {
        for (name, identifier) in [
            ("erf", "RunMat:erf:InvalidInput"),
            ("erfcinv", "RunMat:erfcinv:InvalidInput"),
        ] {
            let source = format!("value = {name}({constructor}(1));");
            let error = execute_source(&source).expect_err("integer input must reject");
            assert_eq!(error.identifier(), Some(identifier), "{source}");
        }
    }
}

#[test]
fn compiled_error_formats_exact_wide_integer_and_all_empty_is_a_noop() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let error = execute_source(
        r#"
        wide = uint64(9007199254740992) + uint64(1);
        error('integer:wide', 'value %d', wide);
        "#,
    )
    .expect_err("error must throw");
    assert_eq!(error.identifier(), Some("integer:wide"));
    assert_eq!(error.message(), "value 9007199254740993");

    execute_source("error([]); value = 1;").expect("all-empty error call is a no-op");
}

#[test]
fn compiled_text_and_exist_boundaries_reject_numeric_roles_before_conversion() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "value = erasePunctuation(uint8(1));",
            "RunMat:erasePunctuation:InvalidInput",
        ),
        (
            "value = eraseURLs(uint8(1));",
            "RunMat:eraseURLs:InvalidInput",
        ),
        ("value = exist(uint8(1));", "RunMat:exist:InvalidName"),
        (
            "value = exist('name', 'mex');",
            "RunMat:compatibility:ExistSearchTypeExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("numeric or extension role must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_errorbar_accepts_every_integer_class() {
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            "handle = errorbar({constructor}([1 2]), {constructor}([3 4]), {constructor}([1 1]));"
        );
        let values = execute_source(&source)
            .unwrap_or_else(|error| panic!("{constructor} compiled errorbar: {error}"));
        assert!(
            values.iter().any(|value| matches!(value, Value::Num(_))),
            "{constructor} errorbar handle"
        );
    }
}

#[test]
fn compiled_integer_extensions_have_stable_strict_mode_identifiers() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "value = exp(uint8(1));",
            "RunMat:compatibility:ExpIntegerInputExtension",
        ),
        (
            "value = expm1(uint8(1));",
            "RunMat:compatibility:Expm1IntegerInputExtension",
        ),
        (
            "value = exprnd(uint8(1));",
            "RunMat:compatibility:ExprndIntegerMeanExtension",
        ),
        (
            "value = exprnd(1, uint8(2), uint8(2));",
            "RunMat:compatibility:ExprndIntegerSizeExtension",
        ),
        (
            "error('unqualified', 'message');",
            "RunMat:compatibility:ErrorUnqualifiedIdentifierExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict-mode extension rejection");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}
