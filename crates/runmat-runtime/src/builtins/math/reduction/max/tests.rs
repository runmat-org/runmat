use super::*;
use futures::executor::block_on;
use runmat_builtins::NANMAX_DESCRIPTOR;
use runmat_value::{ObjectInstance, Tensor, Value};

fn max_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    block_on(super::max_builtin(None, value, rest.into()))
}

fn nanmax_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    block_on(super::nanmax_builtin(None, value, rest.into()))
}

#[test]
fn max_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = MAX_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"M = max(A)"));
    assert!(labels.contains(&"[M, I] = max(A)"));
    assert!(labels.contains(&"M = max(A, B)"));
    assert!(!labels.contains(&"[M, I] = max(A, B)"));
    assert!(labels.contains(&"M = max(A, [], dim)"));
    assert!(labels.contains(&"M = max(A, [], \"all\")"));
    assert!(labels.contains(&"M = max(A, [], \"ComparisonMethod\", method)"));
    assert!(labels.contains(&"M = max(A, B, \"ComparisonMethod\", method)"));
}

#[test]
fn nanmax_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = NANMAX_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"M = nanmax(A)"));
    assert!(labels.contains(&"[M, I] = nanmax(A)"));
    assert!(labels.contains(&"M = nanmax(A, B)"));
    assert!(!labels.contains(&"[M, I] = nanmax(A, B)"));
    assert!(labels.contains(&"M = nanmax(A, [], dim)"));
    assert!(labels.contains(&"[M, I] = nanmax(A, [], dim)"));
    assert!(labels.contains(&"M = nanmax(A, [], vecdim)"));
    assert!(labels.contains(&"M = nanmax(A, [], \"all\")"));
    assert!(labels.contains(&"M = nanmax(A, [], \"linear\")"));
    assert!(labels.contains(&"M = nanmax(A, [], \"ComparisonMethod\", method)"));
    assert!(labels.contains(&"M = nanmax(A, B, \"ComparisonMethod\", method)"));
}

#[test]
fn max_descriptor_errors_have_stable_codes() {
    assert!(MAX_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == MAX_ERROR_INVALID_ARGUMENT.code));
    assert!(MAX_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == MAX_ERROR_INVALID_INPUT.code));
    assert!(MAX_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == MAX_ERROR_SIZE_MISMATCH.code));
    assert!(MAX_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == MAX_ERROR_INTERNAL.code));
    for shared in [
        runmat_builtins::TABLE_ERROR_INVALID_ARGUMENT,
        runmat_builtins::TABLE_ERROR_INVALID_VARIABLE,
    ] {
        assert!(MAX_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == shared.code));
    }
}

#[test]
fn categorical_early_route_keeps_table_owned_argument_and_variable_errors() {
    let nonordinal = ObjectInstance::new(runmat_types::standard::CATEGORICAL);
    let error =
        max_builtin(Value::Object(nonordinal), Vec::new()).expect_err("nonordinal categorical");
    assert_eq!(
        error.identifier(),
        runmat_builtins::TABLE_ERROR_INVALID_ARGUMENT.identifier
    );

    let mut malformed_ordinal = ObjectInstance::new(runmat_types::standard::CATEGORICAL);
    malformed_ordinal.replace_property_unbounded("Ordinal".into(), Value::Bool(true));
    let error = max_builtin(Value::Object(malformed_ordinal), Vec::new())
        .expect_err("missing categorical Codes");
    assert_eq!(
        error.identifier(),
        runmat_builtins::TABLE_ERROR_INVALID_VARIABLE.identifier
    );
}

fn evaluate(value: Value, rest: &[Value]) -> BuiltinResult<MaxEvaluation> {
    block_on(super::evaluate(
        value,
        crate::call::FunctionArgumentView::from_slice(rest),
    ))
}

fn placeholder() -> Value {
    let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
    Value::Tensor(tensor)
}

mod reduction;
mod integer;
mod control;
mod nanmax;
mod pairwise;
#[cfg(feature = "wgpu")]
mod provider;
