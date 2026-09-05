use super::*;

#[test]
fn double_like_missing_prototype_errors() {
    let err =
        double_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
    assert_eq!(err.identifier(), DOUBLE_ERROR_INVALID_ARGUMENT.identifier);
    assert!(err.message().contains("expected prototype"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_like_rejects_extra_arguments() {
    let err = double_builtin(
        Value::Num(0.0),
        vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
    )
    .expect_err("expected error");
    assert_eq!(err.identifier(), DOUBLE_ERROR_INVALID_ARGUMENT.identifier);
    assert!(err.message().contains("too many input arguments"));
}
