use super::*;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "__dispatcher_category_failure",
    category = "fixture-library",
    builtin_path = "tests::category_failure_fallback"
)]
fn category_failure_fallback(_args: Vec<Value>) -> Result<Value, RuntimeError> {
    record("categorized-fallback");
    Ok(Value::Num(1.0))
}

#[runtime_builtin(
    name = "__dispatcher_category_failure",
    category = "",
    builtin_path = "tests::category_failure_override"
)]
fn category_failure_override(_args: Vec<Value>) -> Result<Value, RuntimeError> {
    record("uncategorized-error");
    Err(build_runtime_error("override already performed its effect")
        .with_identifier("RunMat:Fixture:LegacyEffect")
        .build())
}

#[runtime_builtin(
    name = "__dispatcher_category_success",
    category = "fixture-library",
    builtin_path = "tests::category_success_fallback"
)]
fn category_success_fallback(_args: Vec<Value>) -> Result<Value, RuntimeError> {
    record("categorized-fallback");
    Ok(Value::Num(1.0))
}

#[runtime_builtin(
    name = "__dispatcher_category_success",
    category = "",
    builtin_path = "tests::category_success_override"
)]
fn category_success_override(_args: Vec<Value>) -> Result<Value, RuntimeError> {
    record("uncategorized-success");
    Ok(Value::Num(2.0))
}

#[test]
fn no_category_override_error_does_not_execute_categorized_fallback() {
    reset();
    let error = call_builtin("__dispatcher_category_failure", &[]).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Fixture:LegacyEffect"));
    assert_eq!(effects(), ["uncategorized-error"]);
}

#[test]
fn no_category_success_preserves_priority_over_library_category() {
    reset();
    assert_eq!(
        call_builtin("__dispatcher_category_success", &[]).unwrap(),
        Value::Num(2.0)
    );
    assert_eq!(effects(), ["uncategorized-success"]);
}
